import torch
import torch.nn as nn
from methods.mapping_param_npc import PARAMLAYER_TO_IMPLAYER_AND_LAYERTYPE


class NeuronImportanceCalculator:
    def __init__(self, delta: float = 0.99):
        self.delta = float(delta)
        self.neuron_importance = {}   # layer_name -> Tensor[out_dim]   
        self.activations = {}         # layer_name -> activation tensor
        self.hook_handles = []

    def forward_hook(self, layer_name):
        def hook(module, inputs, outputs):
            # only register hooks if gradients are enabled and the outputs require gradients
            if not torch.is_grad_enabled() or not outputs.requires_grad:
                return
            
            outputs.retain_grad()
            self.activations[layer_name] = outputs
            
            if layer_name not in self.neuron_importance:
                self.neuron_importance[layer_name] = torch.zeros(outputs.size(1), device=outputs.device)    
                
        return hook
    
    def register_hooks(self, model: nn.Module):
        self.remove_hooks()
        
        for layer_name, module in model.named_modules():
            # only on these layers because they are the ones that have parameters and are relevant for neuron importance
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
                handle = module.register_forward_hook(self.forward_hook(layer_name))
                self.hook_handles.append(handle)
                
    @torch.no_grad()
    def update_importance(self):
        for layer_name, activation in list(self.activations.items()):
            gradient = activation.grad
            if gradient is None:
                continue
                        
            if activation.dim() == 4: # conv/bn2d
                importance = (activation * gradient).abs().mean(dim=(0,2,3))
            else: # linear/bn1d
                importance = (activation * gradient).abs().mean(dim=0)
            
            # normalization
            importance = importance / importance.mean().clamp(min=1e-12)
            
            self.neuron_importance[layer_name] = importance
            
        self.activations.clear()
        
    # this function maps the importance values to the corresponding parameter layers
    def scale_for_grad(self, importance, layertype):
        
        if layertype in ("bn", "linear_b"):
            scale = importance
            
        elif layertype == "linear_w":
            scale = importance.view(-1,1)
            
        elif layertype == "conv":
            scale = importance.view(-1,1,1,1)
        
        else:
            raise ValueError(f"Unknown layertype: {layertype}")
        
        return scale
    
    # this function returns a dictionary mapping parameter layer names to their corresponding importance scales
    def get_param_scales(self):
        scale_dict = {}
                # key          # value
        for param_layer_name, (imp_layer_name, layertype) in PARAMLAYER_TO_IMPLAYER_AND_LAYERTYPE.items():

            if imp_layer_name not in self.neuron_importance:
                continue

            importance = self.neuron_importance[imp_layer_name]
            scale = self.scale_for_grad(importance, layertype)

            scale_dict[param_layer_name] = scale

        return scale_dict
    
    def print_importance(self):
        print("Anzahl Importance-Layer:", len(self.neuron_importance))
        print("=" * 80)

        for name, imp in self.neuron_importance.items():
            print(name)
            print("Shape:", imp.shape)
            print("Min:", imp.min().item())
            print("Max:", imp.max().item())
            print("Mean:", imp.mean().item())
            print("-" * 80)

    def print_param_scales(self):
        scale_dict = self.get_param_scales()

        print("Anzahl Parameter-Layer:", len(scale_dict))
        print("=" * 80)

        for param_name, scale in scale_dict.items():
            print("Param-Layer:", param_name)
            print("Scale shape:", scale.shape)
            print("Scale min:", scale.min().item())
            print("Scale max:", scale.max().item())
            print("Scale mean:", scale.mean().item())
            print("-" * 80)
            
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.activations.clear()
        
    def remove(self):
        self.remove_hooks()