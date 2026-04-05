"""
Generate PACS dataset for PFLlib federated learning.
PACS: 4 domains (Photo, Art_painting, Cartoon, Sketch), 7 classes.
Each domain = 1 client. Images resized to 224x224 for ResNet-18.

Usage:
  1. Download PACS from: https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd
     or use the torchvision/huggingface version.
  2. Extract to dataset/PACS/rawdata/PACS/ so the structure is:
     dataset/PACS/rawdata/PACS/
       ├── art_painting/
       │   ├── dog/
       │   ├── elephant/
       │   ├── giraffe/
       │   ├── guitar/
       │   ├── horse/
       │   ├── house/
       │   └── person/
       ├── cartoon/
       ├── photo/
       └── sketch/
  3. Run: cd dataset && python generate_PACS.py
"""

import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PACSDataset(Dataset):
    def __init__(self, data_paths, data_labels, transform=None):
        super(PACSDataset, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)


# Class name to label mapping (alphabetical order, standard PACS)
CLASS_NAMES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']


def load_pacs_domain(root_path, domain_name):
    """Load all images from a single PACS domain."""
    domain_path = os.path.join(root_path, domain_name)
    data_paths = []
    data_labels = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(domain_path, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found, skipping.")
            continue
        label = CLASS_TO_IDX[class_name]
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                data_paths.append(fpath)
                data_labels.append(label)

    print(f"  Domain '{domain_name}': {len(data_paths)} images, "
          f"{len(set(data_labels))} classes")
    return data_paths, data_labels


def get_pacs_loader(root_path, domain_name):
    """Create DataLoader for a PACS domain with standard transforms."""
    data_paths, data_labels = load_pacs_domain(root_path, domain_name)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = PACSDataset(data_paths, data_labels, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    return loader


random.seed(1)
np.random.seed(1)
data_path = "PACS/"
dir_path = "PACS/"


def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = os.path.join(data_path, "rawdata", "PACS")

    if not os.path.exists(root):
        print(f"\nError: PACS raw data not found at {root}")
        print("Please download PACS dataset and extract it so the structure is:")
        print(f"  {root}/")
        print("    ├── art_painting/")
        print("    ├── cartoon/")
        print("    ├── photo/")
        print("    └── sketch/")
        print("\nDownload link: https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd")
        return

    X, y = [], []
    print("\nLoading PACS domains...")
    for domain in DOMAINS:
        loader = get_pacs_loader(root, domain)

        for _, (images, labels) in enumerate(loader):
            pass  # Load all data in one batch

        X.append(images.cpu().detach().numpy())
        y.append(labels.cpu().detach().numpy())

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'\nNumber of labels per domain: {labelss}')
    print(f'Number of clients (domains): {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    for client in range(num_clients):
        print(f"Client {client} ({DOMAINS[client]})\t Size: {len(X[client])}\t "
              f"Labels: {np.unique(y[client])}")
        print(f"\t\t Samples per label: {[i for i in statistic[client]]}")
        print("-" * 60)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data,
              num_clients, max(labelss), statistic, None, None, None)

    print(f"\nPACS dataset generated at {dir_path}")
    print(f"  Clients: {num_clients} (one per domain)")
    print(f"  Classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"  Domains: {', '.join(DOMAINS)}")


if __name__ == "__main__":
    generate_dataset(dir_path)
