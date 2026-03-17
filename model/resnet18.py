
import torch
import torchvision
import torch.nn as nn

torch.set_printoptions(threshold=float("inf"))


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
            #https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
            in_features=self.model.fc.in_features,
            out_features=num_classes
        )
    # x ist ein Batch von Bildern
    # das Ergebnis ist logits [B,num_classes]
    def forward(self, x):
        return self.model(x)
    
    # Methode um Klassenzahl zu erweitern
    def expand_head(self, new_num_classes: int):
        # alte fc-Schicht
        old_fc = self.model.fc
        # Anzahl alte Klassen
        old_num_classes = old_fc.out_features
        #Anzahl alte Inputs
        in_features = old_fc.in_features
        
        # neue fc-Schicht mit alten Input-Features aber neue Klassenanzahl      
        new_fc = nn.Linear(in_features, new_num_classes)
        # wegen: RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_addmm)
        # to(old_fc.weight.device) verschiebt der neue Layer auf dasselbe Device vom alten Layer
        new_fc = new_fc.to(old_fc.weight.device)
        # selbe Datentyp wie alte Layer
        new_fc = new_fc.to(dtype=old_fc.weight.dtype)
                                                            
        # kein Gradient Tracking
        with torch.no_grad():
            
            #alte Gewichte kopieren
            """
            new_fc.weight:
            
            Zeile 0 → Klasse 0
            Zeile 1 → Klasse 1
            Zeile 2 → Klasse 2
            Zeile 3 → Klasse 3
            Zeile 4 → Klasse 4
            Zeile 5 → Klasse 5
            Zeile 6 → Klasse 6
            Zeile 7 → Klasse 7 (neu)
            Zeile 8 → Klasse 8 (neu)
            Zeile 9 → Klasse 9 (neu)
            """
            # new_fc.weight[:7] bedeutet nimm die ersten 7 Zeilen vom Tensor weights BIS INDEX 7
            # .copy_(old_fc.weight) kopiert die alten Gewcihte old_fc.weight in diese 7 Zeilen
            new_fc.weight[:old_num_classes].copy_(old_fc.weight)
            new_fc.bias[:old_num_classes].copy_(old_fc.bias)

            #https://docs.pytorch.org/docs/stable/generated/torch.Tensor.normal_.html
            new_fc.weight[old_num_classes:].normal_(mean=old_fc.weight.mean().item(), std= old_fc.weight.std().item())
            # oder einfach nicht initilaiseren, geht auch...
     
            # new_fc.weight[7:] bedeutet nimm alle Zeile ab Index 7, d.h. die letzte 3
            # https://codemia.io/knowledge-hub/path/how_do_i_initialize_weights_in_pytorch
            # Kaiming Initialization nutzlich wenn man ReLU-Aktivierungsfunktionen hat 
            #nn.init.kaiming_normal_(new_fc.weight[old_num_classes:])
            # new_fc.bias[7:] bedeutet nimm alle Zeile ab Index 7, d.h. die letzte 3
            # nn.init.zeros_ füllt den Tensor mit Nullen (https://docs.pytorch.org/docs/stable/nn.init.html)
            nn.init.zeros_(new_fc.bias[old_num_classes:])

        # alte Schicht durch die neue ersetzen
        self.model.fc = new_fc
        self.num_classes = new_num_classes
    
if __name__ == "__main__":
    model = ResNet18(num_classes=7)
    print(model.model.fc)
    old_w = model.model.fc.weight.detach().clone()
    old_b = model.model.fc.bias.detach().clone()
    #print("old weights:", old_w)
       
    
    model.expand_head(new_num_classes=10)
    print(model.model.fc)
    new_w = model.model.fc.weight.detach()
    new_b = model.model.fc.bias.detach()
    #print("new weights:", new_w)
    
    #print("old part weights equal:", torch.equal(new_w[:7], old_w))
    #print("old part bias equal:", torch.equal(new_b[:7], old_b))
   
    last_three_classes_w = model.model.fc.weight.detach()[7:]
    last_three_classes_b = model.model.fc.bias.detach()[7:]
    
    print("new bias all zero:", last_three_classes_b)
    #print("new weights:", last_three_classes_w)
    
    
    
    
    import torch
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet18(num_classes=7).to(device)
    model.eval()
    
    # Random data
    x = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        out_before = model(x)  # [2,7]
        print("out_before:", out_before)

    # expand am selben Modell
    model.expand_head(10)
    model.eval()

    with torch.no_grad():
        out_after = model(x)   # [2,10]
        print("out_after:", out_after)


    print("old and new output equal:", torch.equal(out_before, out_after[:, :7]))
    