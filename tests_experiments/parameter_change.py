""" THIS SCRIPT WAS IMPLEMENTED WITH AI ASSISTANCE """
import torch



before = torch.load(
    "/home/zimame/ma-chaymaa-zimame/results_experiments/baseline_control_lrp_batch_global_0.1/model_after_expand_before_training.pth",
    map_location="cpu"
)

after = torch.load(
    "/home/zimame/ma-chaymaa-zimame/results_experiments/baseline_control_lrp_batch_global_0.1/best_model_control_batch_global_0.1.pth",
    map_location="cpu"
)

stats = {
    "backbone_weight": [0, 0],
    "backbone_bias": [0, 0],
    "fc_weight": [0, 0],
    "fc_bias": [0, 0],
}

fc_weight_name = None
fc_bias_name = None

for name in before:
    if not torch.is_tensor(before[name]):
        continue

    a = before[name].cpu()
    b = after[name].cpu()

    changed = (a != b).sum().item()
    total = a.numel()

    part = "fc" if "fc" in name else "backbone"
    typ = "weight" if "weight" in name else "bias"

    stats[f"{part}_{typ}"][0] += total
    stats[f"{part}_{typ}"][1] += changed

    if part == "fc" and typ == "weight":
        fc_weight_name = name
    if part == "fc" and typ == "bias":
        fc_bias_name = name


for key in stats:
    total = stats[key][0]
    changed = stats[key][1]
    print(key.upper())
    print("Parameter :", total)
    print("Geändert  :", changed)
    print(f"Prozent    : {100 * changed / total:.6f}%")
    print()


print("========== FC OUTPUTS ==========")

w_old = before[fc_weight_name].cpu()
w_new = after[fc_weight_name].cpu()

b_old = before[fc_bias_name].cpu()
b_new = after[fc_bias_name].cpu()

for label, start, end in [
    ("ERSTE 7 OUTPUTS", 0, 7),
    ("LETZTE 3 OUTPUTS", 7, 10),
]:
    w_changed = (w_old[start:end] != w_new[start:end]).sum().item()
    w_total = w_old[start:end].numel()

    b_changed = (b_old[start:end] != b_new[start:end]).sum().item()
    b_total = b_old[start:end].numel()

    print(label)
    print(f"Weights: {w_changed} / {w_total} ({100*w_changed/w_total:.6f}%)")
    print(f"Biases : {b_changed} / {b_total} ({100*b_changed/b_total:.6f}%)")
    print()