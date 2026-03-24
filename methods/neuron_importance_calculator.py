import torch
import torch.nn as nn

class NeuronImportanceCalculator:
    def __init__(self, delta: float = 0.99):
        self.delta = float(delta)
        self.neuron_importance = {}   # layer_name -> Tensor[out_dim]   
        self.activations = {}         # layer_name -> activation tensor
        self.hook_handles = []

    def forward_hook(self, layer_name):
        def hook(module, inputs, outputs):
            # nur im Training / nur wenn grads existieren können
            if not torch.is_grad_enabled() or not outputs.requires_grad:
                return
            
            outputs.retain_grad()
            self.activations[layer_name] = outputs
            
            if layer_name not in self.neuron_importance:
                self.neuron_importance[layer_name] = torch.zeros(outputs.size(1), device=outputs.device)    
                
        return hook
    
    def register_hooks(self, model: nn.Module):
        # doppelte Registrierung der Hooks vermeiden
        self.remove_hooks()
        
        for layer_name, module in model.named_modules():
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
            
            # Normalisierung der importance
            importance = importance / importance.mean().clamp(min=1e-12)
            
            #EMA Update mit delta
            self.neuron_importance[layer_name] = self.delta * self.neuron_importance[layer_name] + (1 - self.delta) * importance
        
        self.activations.clear()
        
    def scale_importance_for_grad(self, importance, layertype):
        
        if layertype in ("bn", "linear_b"):
            scale = importance
            
        elif layertype == "linear_w":
            scale = importance.view(-1,1)
            
        elif layertype == "conv":
            scale = importance.view(-1,1,1,1)
        
        else:
            raise ValueError(f"Unknown layertype: {layertype}")
        
        return scale
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.activations.clear()
        
    def remove(self):
        self.remove_hooks()