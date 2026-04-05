import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchvision.datasets import MNIST, USPS, SVHN
from utils.conf import data_path
import numpy as np

dp = data_path()
print('Data path:', dp)

d = MNIST(dp, train=True, download=False)
img, label = d[0]
print('MNIST type:', type(img), 'label:', label)

d = USPS(dp, train=True, download=False)
img, label = d[0]
print('USPS type:', type(img), type(label))
arr = np.array(img)
print('USPS array shape:', arr.shape, arr.dtype)

d = SVHN(dp, split='train', download=False)
img, label = d[0]
print('SVHN type:', type(img), type(label))
arr = np.array(img)
print('SVHN array shape:', arr.shape, arr.dtype)
