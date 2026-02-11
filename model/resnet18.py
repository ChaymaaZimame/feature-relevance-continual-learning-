
import torchvision
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # lade ResNet18 Modell ohne vortrainierte Gewichte
        self.model = torchvision.models.resnet18(weights=None)
        # letzte Schicht je nach Anzahl Klassen anpassen
        # self.model.fc.in_features gibt die Anzahl der Inputsmerkmale der letzten fc-Schicht zurück,
        # die standardmäßig 512 für ResNet18 ist und schon definiert ist in torchvision.models.resnet18
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    model = ResNet18(num_classes=7)
    #print(model)
    
    for layer_name, layer in model.model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            print(f"Layer Name: {layer_name}, Layer Type: {type(layer).__name__}")