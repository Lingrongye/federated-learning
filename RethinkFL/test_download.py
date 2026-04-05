import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torchvision.datasets import MNIST, USPS, SVHN
import numpy as np

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'dataset')
os.makedirs(data_dir, exist_ok=True)
print('Data dir:', data_dir)

print('Downloading MNIST...')
d = MNIST(data_dir, train=True, download=True)
print('MNIST train size:', len(d))
img, label = d[0]
arr = np.array(img)
print('MNIST img shape:', arr.shape, arr.dtype)

print('Downloading USPS...')
d = USPS(data_dir, train=True, download=True)
print('USPS train size:', len(d))

print('Downloading SVHN...')
d = SVHN(data_dir, split='train', download=True)
print('SVHN train size:', len(d))

print('All downloads complete!')
