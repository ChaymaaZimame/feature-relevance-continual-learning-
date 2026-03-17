# imageWoof Datatset Full Size 
# zunächst mit wget https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz heruntergeladen
# dann mit tar -xvf imagewoof2.tgz entpackt
#ndrin sind c. 13000 Bilder von Hunden in 10 Klassen
# im train-set sind cirka 9300 Bilder und im valid-set cirka 3930 Bilder


#Quelle: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

# unterschiedliche Transformationen für Trainings- und Validierungsdaten
# so wie hier empfohlen https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(224,224)),
        transforms.RandomHorizontalFlip(),
        #transforms.RandAugment(num_ops=2, magnitude=9),  # starke Augmentierung
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5022, 0.4623, 0.3986], std=[0.2532, 0.2454, 0.2540])  
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5022, 0.4623, 0.3986], std=[0.2532, 0.2454, 0.2540])  
    ])
}

'''
def transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])  
    ])
'''
class ImageWoofDataset(Dataset):
    def __init__(self, 
                 img_path, #/home/zimame/data/imagewoof2
                 split='train',
                 transform = None,
                 class_indices = None
        ):
        self.img_path = Path(img_path)
        self.split = split
        # beim normalen Training wird keine Transformation übergeben
        # dann werden die standard Transformationen aus data_transforms verwendet
        # beim mean/std berechnen wird eine Transformation übergeben (da ohne normalization übergeben)
        if transform is None:
            self.transform = data_transforms[split]
        else:
            self.transform = transform
        
        if split == 'train':
            split_dir = self.img_path / 'train' # d.h./home/zimame/data/imagewoof2/train
        else:
            split_dir = self.img_path / 'val'    
       
        # alle Klassenordner in Liste speichern
        #self.all_class_names = []
        # interdir() listet alle Unterordner vom split_dir (train oder val)    
        #is_dir überprüft ob es ein Ordner ist
        #for d in split_dir.iterdir():
         #   if d.is_dir():
          #      self.all_class_names.append(d.name)
                
        self.all_class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        
        if class_indices is None:
            self.class_names = self.all_class_names
        else:
            #for i in class_indices:
             #   self.class_names = self.all_class_names[i]
            self.class_names = [self.all_class_names[i] for i in class_indices]
        
        # class_name zu Label-ID mappen
        # später als label für jedes Bild verwenden
        self.class_to_label = {}
        for idx, class_name in enumerate(self.class_names):
            self.class_to_label[class_name] = idx # z.B. {"n02086240": 0} - 0-9
        
        #alle Bildpfade und Labels in Liste speichern    
        # je nach split_dir (train oder val)         
        self.subsets = self.build_train_val_subsets(split_dir)
        
        #print(f"Dataset initialized with {len(self.subsets)} images and {len(self.class_names)} classes for split '{self.split}'")          
         
    def build_train_val_subsets(self, split_dir):
        # liste von Tupeln (img_path, label) initialisiert
        #images = []
        # numpy array statt liste für schnelleren zugriff
        images_array = []
            
        for class_name in self.class_names:
            # ich greife auf den jeweiligen Klassenordner zu
            class_dir = split_dir / class_name
            #interdir() listet in dem Fall alle Bilder in jedem Klassenordner
            for img_path in sorted(class_dir.iterdir()):
                    # das label wird vom dict class_to_label geholt
                    label = self.class_to_label[class_name] #idx
                    
                    images_array.append((img_path, label))
                    #images.append((img_path, label))
        return np.array(images_array)       
          
        
    def __len__(self):
        return len(self.subsets)
        
    def __getitem__(self, idx):
        #self.subsets ist eine Liste von Tupeln (img_path, label)
        # labels werden automatisch von dataloader in tensor umgewandelt
        img_path, label = self.subsets[idx]
        with Image.open(img_path) as im:
            image = im.convert("RGB")
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

if __name__ == "__main__": 
    
    train_dataset_7 = ImageWoofDataset(img_path="/home/zimame/data/imagewoof2", split='train', class_indices = [0,1,2,3,4,5,6])
    val_dataset_7 = ImageWoofDataset(img_path="/home/zimame/data/imagewoof2", split='val', class_indices = [0,1,2,3,4,5,6])
    
    print(f"Training-Samples (7 classes): {len(train_dataset_7)}")
    print(f"Validation-Samples (7 classes): {len(val_dataset_7)}")

    train_dataset_3 = ImageWoofDataset(img_path="/home/zimame/data/imagewoof2", split='train', class_indices = [7,8,9])
    val_dataset_3 = ImageWoofDataset(img_path="/home/zimame/data/imagewoof2", split='val', class_indices = [7,8,9])
    
    print(f"Training-Samples (3 classes): {len(train_dataset_3)}")
    print(f"Validation-Samples (3 classes): {len(val_dataset_3)}")
    
    train_dataloader_7 = DataLoader(train_dataset_7, batch_size=32, shuffle=True)
    val_dataloader_7 = DataLoader(val_dataset_7, batch_size=32, shuffle=False)
    train_dataloader_3 = DataLoader(train_dataset_3, batch_size=32, shuffle=True)
    val_dataloader_3 = DataLoader(val_dataset_3, batch_size=32, shuffle=False)
    
    print("Beispiel Batch aus dem Trainings-Dataloader (7 Klassen):")
    images, labels = next(iter(train_dataloader_3))
    print(f"Images Shape: {images.shape}") #torch.Size([32, 3, 224, 224])
    print(f"Labels: {labels}")                                  
    print("Images tensor example: ", images[0])

    