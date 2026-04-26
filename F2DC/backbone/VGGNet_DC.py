"""Stub for missing VGGNet_DC in F2DC release.
Only needed so dataset/__init__.py import does not crash. We do not actually
use AlexNet variants; ResNet10_DC is the F2DC default for all 3 datasets.
"""
def alexnet_dc(num_classes=7, gum_tau=0.1):
    raise NotImplementedError("alexnet_dc not in F2DC release; use resnet10_dc")
def alexnet_dc_office(num_classes=10, gum_tau=0.1):
    raise NotImplementedError("alexnet_dc_office not in F2DC release")
def alexnet_dc_digits(num_classes=10, gum_tau=0.1):
    raise NotImplementedError("alexnet_dc_digits not in F2DC release")
