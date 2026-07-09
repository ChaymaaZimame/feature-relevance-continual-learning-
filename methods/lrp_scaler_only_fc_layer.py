import torch

def relevance_to_factor(scale: torch.Tensor, beta: float, eps: float = 1e-6):
    factor = torch.zeros_like(scale)

    for s, f in zip(scale, factor):
        if s >= beta:
            new_value = 1.0 / (s + eps)
            f.copy_(new_value)  

    return factor


class LRPScaler:
    def __init__(self, beta=None, eps=1e-6, writer=None):
        self.beta = beta
        self.eps = eps
        self.writer = writer
        self.global_step = 0
        
        self.handles = []
        self.factor_dict = {}

    def update_scales(self, scale_dict):
        self.factor_dict = {}
        print("\n=== FACTOR STATS ===")

        for pname, scale in scale_dict.items():
            # only consider parameters that belong to the fc layer
            if not "fc" in pname:
                continue
            
            factor = relevance_to_factor(scale, self.beta, self.eps)
            self.factor_dict[pname] = factor
            
            nonzero = (factor != 0).sum().item()
            total = factor.numel()
            
            print(
                f"{pname}: "
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

        for pname, p in model.named_parameters():
            if not "fc" in pname:
                print(f"SKIP: {pname}")
                continue
            if not p.requires_grad:
                print(f"SKIP (no grad): {pname}")
                continue
            
            handle = p.register_hook(self.param_hook(pname))
            print(f"HOOK: {pname}")
            self.handles.append(handle)