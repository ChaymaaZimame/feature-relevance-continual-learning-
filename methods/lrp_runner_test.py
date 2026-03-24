import torch

from model.resnet18 import ResNet18
from methods.lrp_runner import LRPRunner

from torch.utils.data import DataLoader
from data.dataset import ImageWoofDataset

model = ResNet18(num_classes=7)

# Gewichte des besten Modells laden
model.load_state_dict(torch.load("/home/zimame/ma-chaymaa-zimame/checkpoints/best_model_test0_sgd5_2_test.pth"))

# Gerät festlegen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modell auf Gerät verschieben und in den Evaluierungsmodus setzen
model = model.to(device).eval()

lrp = LRPRunner(model, device=device)

# Datensatz mit 7 Klassen laden
val_dataset_7 = ImageWoofDataset(
    img_path="/home/zimame/data/imagewoof2",
    split="val",
    class_indices=[0, 1, 2, 3, 4, 5, 6]
)

# DataLoader für den Validierungsdatensatz erstellen, mit BS=8
val_loader = DataLoader(val_dataset_7, batch_size=8, shuffle=False)

# nur der erste Batch wird geladen (8 Bilder)
for images, labels in val_loader:
    break

# Bilder aufs Gerät verschieben
input_images = images.to(device)

# prüfen ob get_layer_relevances() sinnvolle LRP-Layer liefert
print("\n########## Relevances per LRP-Layer ##########")    
rel_dict = lrp.get_layer_relevances(input_images, topk=1)
#lrp.print_relevances(rel_dict)
assert len(rel_dict) > 0

# prüfen ob get_param_scales() funktioniert
# enthalten alle noch 1,1,1 weil
# ich noch nicht die echte param.grad.shape einsetze
# sondern nur einen vorbereitenden Skalierungstensor baue
print("\n########## Param-Scales ##########")
scale_dict = lrp.get_param_scales(input_images, topk=1)
assert len(scale_dict) > 0
# einfach Shapes anzeigen
for name, scale in scale_dict.items():
    print(name, scale.shape)
    
print("\n########## Detailed Param Scales ##########")
lrp.print_param_scales(input_images) #'--diese Funtkion mit print_relevances kombinieren?'
    
