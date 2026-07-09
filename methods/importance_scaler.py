import torch


def importance_to_factor(scale: torch.Tensor, beta: float, eps: float = 1e-6):
    factor = torch.zeros_like(scale)
    for s, f in zip(scale, factor):
        if s >= beta:
            new_value = 1.0 / (s + eps)
            f.copy_(new_value)
    
    return factor


class ImportanceScaler:
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

        for param_name, scale in scale_dict.items():
            factor = importance_to_factor(scale, self.beta, self.eps)
            self.factor_dict[param_name] = factor
            
            nonzero = (factor != 0).sum().item()
            total = factor.numel()

            print(
                f"{param_name}: "
                f"scale_shape={tuple(scale.shape)} | "
                f"factor_shape={tuple(factor.shape)} | "
                f"scale_min={scale.min().item():.6f} | "
                f"scale_max={scale.max().item():.6f} | "
                f"factor_nonzero={nonzero}/{total}"
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
                return grad

            factor = self.factor_dict[param_name]
            
            grad_before = grad.abs().mean().item()
            scaled_grad = grad * factor
            grad_after = scaled_grad.abs().mean().item()

            if self.writer is not None and param_name == "model.fc.weight":
                self.writer.add_scalar(
                    f"Importance_Gradients/{param_name}_before",
                    grad_before,
                    self.global_step
                )
                self.writer.add_scalar(
                    f"Importance_Gradients/{param_name}_after",
                    grad_after,
                    self.global_step
                )

                self.global_step += 1
                
            print(
                f"[DEBUG] HOOK {param_name} | "
                f"grad_before={grad_before:.8f} | "
                f"grad_after={grad_after:.8f} | "
                f"factor_shape={tuple(factor.shape)} | "
                f"grad_shape={tuple(grad.shape)}"
            )
                
            return scaled_grad

        return hook

    def register_all_params(self, model):
        self.remove()

        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue

            handle = p.register_hook(self.param_hook(pname))
            self.handles.append(handle)