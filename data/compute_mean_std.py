import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data.dataset import ImageWoofDataset



def compute_mean_std(img_path, batch_size: int = 32):
    print("Start computing mean and std...")
    
	# transform without normalization
    basic_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # create training dataset
    train_dataset = ImageWoofDataset(img_path="/home/zimame/data/imagewoof2", split='train', transform=basic_transform)

    print(f"Full training dataset size: {len(train_dataset)}")

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # initialize variables for calculation
    channel_sum = torch.zeros(3) #to save the results of each channel
    channel_squared_sum = torch.zeros(3)
    total_pixel_count = 0

    for images, _ in tqdm(train_loader, desc="Computing mean/std"):
        B, C, H, W = images.shape
        total_pixel_count += B * H * W

        channel_sum += images.sum(dim=[0, 2, 3]) # E[x] before dividing by total pixel count; sum(B,H,W)=[pixels in each channel]
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3]) #E[X**2] before dividing by total pixel count

    mean = channel_sum / total_pixel_count #E[x]
    std = torch.sqrt(channel_squared_sum / total_pixel_count - mean ** 2) # sqrt(E[X**2]-E[x]**2)

    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")

    return mean, std

if __name__ == "__main__":
    compute_mean_std(img_path="/home/zimame/data/imagewoof2", batch_size=64)
