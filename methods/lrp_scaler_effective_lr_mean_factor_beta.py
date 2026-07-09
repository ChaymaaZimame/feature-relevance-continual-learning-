import torch


def relevance_to_factor(scale: torch.Tensor, global_factor_value):
    return torch.full_like(scale, global_factor_value)


class LRPScaler:
    def __init__(self, beta=1.0, eps=1e-6, writer=None):
        self.beta = beta
        self.eps = eps
        self.writer = writer
        self.global_step = 0
        

        self.handles = []
        self.factor_dict = {}

    def update_scales(self, scale_dict):
        self.factor_dict = {}
        print("\n=== FACTOR STATS ===")
        
        # all selected factors from all layers will be collected to compute a global mean   
        selected_factors = []
        
        for param_name, scale in scale_dict.items():
            mask = scale >= self.beta
            if mask.any():
                factor_values = 1.0 / (scale[mask].detach() + self.eps)
                selected_factors.append(factor_values.flatten())
            
        selected_factors = torch.cat(selected_factors)
        
        # compute the global mean of the selected factors
        global_factor_value = selected_factors.mean()
        
        print(f"GLOBAL mean(1/relevance): {global_factor_value.item():.6f}")
        print(f"Number of selected neurons: {selected_factors.numel()}")        
        

        for param_name, scale in scale_dict.items():
            factor = relevance_to_factor(scale,global_factor_value)
            self.factor_dict[param_name] = factor
          
            print(
                f"{param_name}: "
                f"scale_shape={tuple(scale.shape)} | "
                f"factor_shape={tuple(factor.shape)} | "
                f"scale_min={scale.min().item():.6f} | "
                f"scale_max={scale.max().item():.6f} | "
            )
            
            print("FACTOR VALUES:")
            print(factor.flatten()[:20])  # print first 20 values of factor for debugging

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def param_hook(self, param_name):
        def hook(grad):
            if param_name not in self.factor_dict:
                print(f"Für {param_name} wurde kein Faktor im factor_dict gefunden, Gradienten bleibt unverändert")
                return grad
            
            factor = self.factor_dict[param_name]
            
            grad_before = grad.abs().mean().item()
            scaled_grad = grad * factor
            grad_after = scaled_grad.abs().mean().item()
            
            if self.writer is not None and param_name == "model.fc.weight":
                self.writer.add_scalar(f"Gradients/{param_name}_before", grad_before, self.global_step)
                self.writer.add_scalar(f"Gradients/{param_name}_after", grad_after, self.global_step)

            self.global_step += 1
            
            print(
                f"[DEBUG] HOOK {param_name} | "
                f"grad_before={grad.abs().mean().item():.8f} | "
                f"grad_after={scaled_grad.abs().mean().item():.8f} | "
                f"factor_shape={tuple(factor.shape)} | "
                f"grad_shape={tuple(grad.shape)}"
            )

            return scaled_grad

        return hook

    def register_all_params(self, model):
        self.remove()

        for param_name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            handle = p.register_hook(self.param_hook(param_name))
            self.handles.append(handle)