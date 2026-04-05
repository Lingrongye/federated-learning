import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.digits import MyDigits
from utils.conf import data_path
import torchvision.transforms as transforms

dp = data_path()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

d = MyDigits(dp, train=True, download=False, transform=transform, data_name='mnist')
print('MyDigits(mnist) len:', len(d.dataset))
print('Has targets:', hasattr(d.dataset, 'targets'))
img, label = d[0]
print('Item shape:', img.shape, 'label:', label)

d2 = MyDigits(dp, train=True, download=False, transform=transform, data_name='usps')
print('MyDigits(usps) len:', len(d2.dataset))
img2, label2 = d2[0]
print('USPS Item shape:', img2.shape, 'label:', label2)

transform_svhn = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
d3 = MyDigits(dp, train=True, download=False, transform=transform_svhn, data_name='svhn')
print('MyDigits(svhn) len:', len(d3.dataset))
print('Has labels:', hasattr(d3.dataset, 'labels'))
img3, label3 = d3[0]
print('SVHN Item shape:', img3.shape, 'label:', label3)
