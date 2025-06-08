import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader
import tonic
from torch.utils.data import DataLoader, Subset, random_split
from tonic import DiskCachedDataset

import nnt_cli.utils as nu
from nnt_cli.templates.gen_temp import GeneralTemplate
from nnt_cli.utils.collate_fn import PadTensors
from nnt_cli.utils.settin.gen_settin import get_num_workers

class _Project_Template(GeneralTemplate):
    """
    Project template for specific project.
    """
    
    def init_params(self):
        """
        **Essential**
        Set the global parameters for the class, including settings.
        """
        super().init_params()

        self.infer_size=0.1
        # The size of the infer set. If it is 0.1, then the infer set will be 10% of the test set.
        # If it is 1, then the infer set will be the same as the test set.
    
    # def set_iter(self):
    #     """
    #     Set the iteration variables and values.
    #     The variables set here will be used for iteration training if you call the iteration training methods.
    #     `self.variable_name` will be used first when `iter` (sweep) mode is called.
    #     `self.variable_name` must be consistent with the variable names in the class defined.
    #     """
    #     self.vary_list=[1,2,4,8]
    #     self.variable_name="bit_width"

    #     self.vary_list2=[64,128,256,512]
    #     self.variable_name2="num_hiddens"
    
    def set_dataset(self, preset=False):
        """
        **Essential**
        Set the dataset.
        `self.train_dataset` and `self.test_dataset` must be set.
        """
        self.sensor_size = tonic.datasets.NMNIST.sensor_size

        self.pre_transform = tonic.transforms.Compose([
                            tonic.transforms.Denoise(filter_time=10000),
                            tonic.transforms.ToFrame(
                                sensor_size=self.sensor_size,
                                time_window=5000)
                        ])
        
        
        train_post_transform = transforms.Compose([
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10, 10]),
        ])

        test_post_transform = transforms.Compose([
            torch.from_numpy,

        ])

        if preset is True:
            # For analysis plotting, we need to use the original dataset.
            self.train_preset = tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=True)
            self.test_preset = tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=False)

        
        if self.with_cache == "2stage":
            # Using 2-stage transform to save cache space.
            # Only first stage transformed data will be saved.
            trainset = tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=True)
            testset = tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=False)

            self.train_dataset = DiskCachedDataset(trainset, transform=train_post_transform, cache_path=f'{self.cache_path}/train')
            self.test_dataset = DiskCachedDataset(testset, transform=test_post_transform, cache_path=f'{self.cache_path}/test')
            print("Using two-stage transform to save cache space!")
        elif self.with_cache == "Full":
            # Full cache will use a lot of space, but it will be faster because data is already transformed.
            train_full_transform=tonic.transforms.Compose([self.pre_transform, train_post_transform])
            test_full_transform=tonic.transforms.Compose([self.pre_transform, test_post_transform])
            trainset=tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=train_full_transform, train=True)
            testset=tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=test_full_transform, train=False)
            self.train_dataset = DiskCachedDataset(trainset, cache_path=f'{self.cache_path}/train')
            self.test_dataset = DiskCachedDataset(testset, cache_path=f'{self.cache_path}/test')

            print("Using full transform to achieve full speed!")
        elif self.with_cache == "False":
            train_full_transform=tonic.transforms.Compose([self.pre_transform, train_post_transform])
            test_full_transform=tonic.transforms.Compose([self.pre_transform, test_post_transform])
            self.train_dataset=tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=train_full_transform, train=True)
            self.test_dataset=tonic.datasets.NMNIST(save_to=f'{self.parent_path}/dataset', transform=test_full_transform, train=False)
            print("Not using cache!")
        else:
            raise ValueError("with_cache must be one of '2stage', 'Full', 'False'")
        
        if self.vali_fr_train is True:
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [1-self.val_size, self.val_size])
        else:
            self.val_dataset = self.test_dataset
        
        if self.quick_debug is True:
                # For quick debug
                trainset_size = range(int(len(self.train_dataset)*self.qb_dataset_size))
                testset_size = range(int(len(self.test_dataset)*self.qb_dataset_size))
                self.train_dataset = Subset(self.train_dataset, trainset_size)
                self.test_dataset = Subset(self.test_dataset, testset_size)

    def set_dataloader(self):
        """
        **Essential**
        Set the dataloader.
        `self.train_loader` and `self.test_loader` must be set. If you don't need inference, self.infer_loader = None.
        """
        if self.para_mode is True:
            num_workers=get_num_workers("Dist",dist_num=4)

        elif self.debug_mode is True:
            num_workers=get_num_workers("Half")

        else:
            num_workers=get_num_workers("Full")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                       collate_fn=PadTensors(batch_first=False),
                                       shuffle=True, num_workers=num_workers,pin_memory=True)

        if self.infer_size >=1:
            infer_subset=self.test_dataset
            test_subset=self.test_dataset
        else:
            test_subset, infer_subset = random_split(self.test_dataset, [1-self.infer_size, self.infer_size])

        self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, 
                                      collate_fn=PadTensors(batch_first=False),
                                      shuffle=False,num_workers=num_workers,pin_memory=True)
        self.infer_loader = DataLoader(infer_subset, batch_size=self.batch_size, 
                                       collate_fn=PadTensors(batch_first=False),
                                       shuffle=False, num_workers=num_workers,pin_memory=True)
        
        if hasattr(self, 'val_dataset'):
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, 
                                         collate_fn=PadTensors(batch_first=False),
                                         shuffle=False, num_workers=num_workers,pin_memory=True)

    def set_name(self):
        """
        If inputs are None, here are default output file name.
        Setting at GeneralTemplate training methods.
        By using `match_name`and adding index can make sure that multi times results will not be overwritten.
        Every time you run the code (not iteration), it will create a new folder with the next index.
        You can rewrite this method in subclass to set your own name.
        """
        self.match_name=f"{self.notebook_name}_qnn_FMNIST_"
        self.remark=f"2_linear_layers"

    # def set_model(self):
    #     """
    #     **Essential**
    #     But not necessary to set the model here. I suggest to set the model in another .ipynb file
    #     Set the training model.
    #     `self.net`, `self.loss`, `self.optimizer` must be set.
    #     In subclass, do NOT use super() to inherit optimizer!!!
    #     """
    #     num_inputs =28*28
    #     num_hiddens=self.num_hiddens
    #     num_outputs = 10

    #     self.net = nn.Sequential(
    #             # unnamed layer, use layer1.parameters() or layer1.weight to access
    #             # named layer, use net.<name>.param to access
    #             OrderedDict([
    #                 ('flaten', nn.Flatten()),
    #                 ('input_sav1', nu.layer.gen_layer.InputSaviorLayer(f"{self.dir_path}/data/{self.vary_title}","input_af_quant",squeeze=False)),
    #                 ('linear_in', nn.Linear(num_inputs, num_hiddens,bias=False,device=self.device,)),                                
    #                 ('batch_norm1', nn.BatchNorm1d(num_hiddens)),
    #                 ('relu1',nn.ReLU()),

    #                 ('linear2', nn.Linear(num_hiddens, num_hiddens,bias=False,device=self.device)),                                
    #                 ('batch_norm2', nn.BatchNorm1d(num_hiddens)),
    #                 ('relu2',nn.ReLU()),

    #                 ('linear3', nn.Linear(num_hiddens, num_hiddens,bias=False,device=self.device)),                                
    #                 ('batch_norm3', nn.BatchNorm1d(num_hiddens)),
    #                 ('relu3',nn.ReLU()),

    #                 ('linear_out',nn.Linear(num_hiddens, num_outputs,bias=False,device=self.device)),
    #             ])
    #         ).to(self.device)
        
    #     self.loss = nn.CrossEntropyLoss()

    #     self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
    
    def init_net(self):
        """
        Init net if you need to.
        """
        print("No init of the network!")

    def set_train(self):
        """
        Set the training function.
        If you want to use auto plot for later results,
        `self.results` must be set at [train_l_list, train_acc_list, test_acc_list, infer_acc_list] format.
        """
        num_steps = 0
        # Not used

        if self.load_cp_flag:
            self.results=nu.train.snn_train.train_snn(self.net,self.train_loader,self.val_loader,self.loss,self.num_epochs,self.optimizer,
                                    num_steps,infer_loader=self.infer_loader, SF_funct=True, in2spk=False, 
                                    forward=True, eve_in=True,device=self.device,debug_mode=self.debug_mode,
                                    checkpoint_path=self.checkpoint_path,checkpoint=self.checkpoint,cp_add_sav_dict=self.cp_add_dict,
                                    test_loader=self.test_loader)
        else:
            self.results=nu.train.snn_train.train_snn(self.net,self.train_loader,self.val_loader,self.loss,self.num_epochs,self.optimizer,
                                    num_steps,infer_loader=self.infer_loader, SF_funct=True, in2spk=False, 
                                    forward=True, eve_in=True,device=self.device,debug_mode=self.debug_mode,
                                    checkpoint_path=self.checkpoint_path,cp_add_sav_dict=self.cp_add_dict,
                                    test_loader=self.test_loader)

        # self.results=nu.train.gen_train.train_funct(self.net, self.train_loader, self.test_loader, self.loss, self.num_epochs, 
        #     self.optimizer,device=self.device,quant_tensor=False, infer_iter=self.infer_loader)

