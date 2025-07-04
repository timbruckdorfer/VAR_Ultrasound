import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from vqvae import VQVAE

# Directories for all preprocessed datasets
DATASET_DIRS = [
    'datasets_preprocessed/Kidney_preprocessed',
    'datasets_preprocessed/Fetal_head_preprocessed',
    'datasets_preprocessed/Thyroid_preprocessed/DDTI_preprocessed',
    'datasets_preprocessed/Thyroid_preprocessed/Gland_preprocessed',
    'datasets_preprocessed/Thyroid_preprocessed/Nodules_preprocessed',
]

# Number of images to sample from each dataset for testing
SAMPLE_PER_DATASET = 20

class ImageFolderSample(Dataset):
    def __init__(self, root_dir, sample_size=20, image_size=256):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(root_dir, ext)))
        self.files = random.sample(files, min(sample_size, len(files)))
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)

def get_test_dataloader(batch_size=4, sample_per_dataset=20, image_size=256):
    datasets = []
    for d in DATASET_DIRS:
        if os.path.exists(d):
            datasets.append(ImageFolderSample(d, sample_per_dataset, image_size))
    if not datasets:
        raise RuntimeError('No datasets found!')
    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=2)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_test_dataloader()
    model = VQVAE(test_mode=False).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('Starting test training loop...')
    for epoch in range(1):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            rec, usages, vq_loss = model(imgs)
            rec_loss = torch.nn.functional.mse_loss(rec, imgs)
            loss = rec_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch} Iter {i}: rec_loss={rec_loss.item():.4f}, vq_loss={vq_loss.item():.4f}, total={loss.item():.4f}')
            if i >= 2:  # Only a few iterations for test
                break
    print('Test training loop finished.')

if __name__ == '__main__':
    main() 