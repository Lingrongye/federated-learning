import torchvision.transforms as transforms
from utils.conf import data_path
from datasets.utils.federated_dataset import FederatedDataset, partition_pacs_domain_skew_loaders
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNet import resnet10, resnet12, resnet18, resnet34
from backbone.WRN import wrn_28_10
from backbone.ResNet_DC import resnet10_dc, resnet34_dc
from backbone.VGGNet_DC import alexnet_dc
from backbone.efficientnet import EfficientNetB0
from backbone.googlenet import GoogLeNet
from backbone.mobilnet_v2 import MobileNetV2, mobile_dc
from torchvision.datasets import ImageFolder, DatasetFolder
from backbone.VGGNet import vggnet
from backbone.mobileNet import mobilenet


class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None,subset_train_num=7,subset_capacity=10):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + 'PACS_7/' + self.data_name + '/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + 'PACS_7/' + self.data_name + '/', self.transform, self.target_transform)

        all_data = self.imagefolder_obj.samples
        self.train_index_list=[]
        self.test_index_list=[]
        for i in range(len(all_data)):
            if i%subset_capacity<=subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):

        if self.train:
            used_index_list=self.train_index_list
        else:
            used_index_list=self.test_index_list

        path = self.imagefolder_obj.samples[used_index_list[index]][0]
        target = self.imagefolder_obj.samples[used_index_list[index]][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaPACS(FederatedDataset):
    NAME = 'fl_pacs'
    SETTING = 'domain_skew'
    DOMAINS_LIST = ['photo', 'art', 'cartoon', 'sketch']
    # 每个domain 按 30% 的训练样本给client使用 
    percent_dict = {'photo':0.3, 'art':0.3, 'cartoon':0.3, 'sketch':0.3} # 30% data size

    N_SAMPLES_PER_Class = None
    N_CLASS = 7
    Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.RandomCrop(128, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    def get_data_loaders(self, selected_domain_list):

        using_list = self.DOMAINS_LIST if (selected_domain_list is None or len(selected_domain_list) == 0) else list(selected_domain_list)

        nor_transform = self.Nor_TRANSFORM
        train_dataset_list = []
        test_dataset_list = []
        test_transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor(), self.get_normalization_transform()])

        for _, domain in enumerate(using_list):
            train_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=True,
                                               transform=nor_transform)

            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            test_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=False,
                                              transform=test_transform)
            test_dataset_list.append(test_dataset)
        traindls, testdls = partition_pacs_domain_skew_loaders(train_dataset_list, test_dataset_list, self.args.model, self)
        return traindls, testdls

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaPACS.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name):
        nets_dict = {'resnet10': resnet10, 'resnet12': resnet12, 'resnet18': resnet18, 'resnet34': resnet34,
                     'efficient': EfficientNetB0, 'mobilnet': MobileNetV2,'googlenet':GoogLeNet}
        nets_list = []
        if names_list == None:
            if model_name=='f2dc':
                for j in range(parti_num):
                    nets_list.append(resnet10_dc(num_classes=FedLeaPACS.N_CLASS,
                                                 gum_tau=FedLeaPACS.model_args.gum_tau))
            elif model_name in ('f2dc_pg', 'f2dc_pgv33', 'f2dc_pg_lab'):
                from backbone.ResNet_DC_PG import resnet10_dc_pg
                args_obj = FedLeaPACS.model_args
                pw = getattr(args_obj, 'pg_proto_weight', 0.3)
                tau = getattr(args_obj, 'pg_attn_temperature', 0.3)
                for j in range(parti_num):
                    nets_list.append(resnet10_dc_pg(num_classes=FedLeaPACS.N_CLASS,
                                                     gum_tau=FedLeaPACS.model_args.gum_tau,
                                                     proto_weight=pw,
                                                     attn_temperature=tau))
            elif model_name == 'f2dc_pg_ml':
                from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml
                args_obj = FedLeaPACS.model_args
                pw = getattr(args_obj, 'pg_proto_weight', 0.3)
                tau = getattr(args_obj, 'pg_attn_temperature', 0.3)
                lc = getattr(args_obj, 'ml_lite_channel', 32)
                lt = getattr(args_obj, 'ml_lite_tau', 0.1)
                rho = getattr(args_obj, 'ml_main_rho', 0.0)
                for j in range(parti_num):
                    nets_list.append(resnet10_dc_pg_ml(num_classes=FedLeaPACS.N_CLASS,
                                                       gum_tau=FedLeaPACS.model_args.gum_tau,
                                                       proto_weight=pw,
                                                       attn_temperature=tau,
                                                       ml_lite_channel=lc,
                                                       ml_lite_tau=lt,
                                                       ml_main_rho=rho))
            elif model_name == 'f2dc_dse':
                from backbone.ResNet_DC_F2DC_DSE import resnet10_f2dc_dse
                args_obj = FedLeaPACS.model_args
                dse_red = getattr(args_obj, 'dse_reduction', 8)
                for j in range(parti_num):
                    nets_list.append(resnet10_f2dc_dse(num_classes=FedLeaPACS.N_CLASS,
                                                        gum_tau=FedLeaPACS.model_args.gum_tau,
                                                        dse_reduction=dse_red))
            elif model_name=='fdse':
                from backbone.ResNet_FDSE import resnet10_fdse_pacs
                for j in range(parti_num):
                    nets_list.append(resnet10_fdse_pacs(num_classes=FedLeaPACS.N_CLASS))
            else:
                for j in range(parti_num):
                    nets_list.append(resnet10(FedLeaPACS.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaPACS.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform