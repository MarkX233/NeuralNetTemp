import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader

import nnt_cli.utils as nu
from nnt_cli.templates.gen_temp import GeneralTemplate

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
    
    def set_iter(self):
        """
        Set the iteration variables and values.
        The variables set here will be used for iteration training if you call the iteration training methods.
        `self.variable_name` will be used first when `iter` (sweep) mode is called.
        `self.variable_name` must be consistent with the variable names in the class defined.
        """
        self.vary_list=[1,2,4,8]
        self.variable_name="bit_width"

        self.vary_list2=[64,128,256,512]
        self.variable_name2="num_hiddens"
    
    def set_dataset(self):
        """
        **Essential**
        Set the dataset.
        `self.train_dataset` and `self.test_dataset` must be set.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))    # y=(x-mean)/std    Transform the value from 0 to 1 to -1 to 1 to fit binary quantizer.
            ])

        self.train_dataset = torchvision.datasets.FashionMNIST(root=f'{self.parent_path}/dataset/FashionMNIST',
                                                        train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.FashionMNIST(root=f'{self.parent_path}/dataset/FashionMNIST', 
                                                        train=False, download=True, transform=transform)
    def set_dataloader(self):
        """
        **Essential**
        Set the dataloader.
        `self.train_loader` and `self.test_loader` must be set. If you don't need inference, self.infer_loader = None.
        """
        if self.para_mode is True:
            num_workers=nu.settin.gen_settin.get_num_workers("Dist",dist_num=4)

        elif self.debug_mode is True:
            num_workers=nu.settin.gen_settin.get_num_workers("Half")

        else:
            num_workers=nu.settin.gen_settin.get_num_workers("Full")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        # self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=num_workers)

        test_subset, infer_subset = nu.settin.gen_settin.spilt_dataset(self.infer_size, self.test_dataset)

        self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False,num_workers=num_workers)
        self.infer_loader = DataLoader(infer_subset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def set_name(self):
        """
        If inputs are None, here are default output file name.
        Setting at GeneralTemplate training methods.
        By using `match_name`and adding index can make sure that multi times results will not be overwritten.
        Every time you run the code (not iteration), it will create a new folder with the next index.
        """
        self.match_name=f"{self.notebook_name}_qnn_FMNIST_"
        self.remark=f"2_linear_layers"

    def set_model(self):
        """
        **Essential**
        Set the training model.
        `self.net`, `self.loss`, `self.optimizer` must be set.
        In subclass, do NOT use super() to inherit optimizer!!!
        """
        num_inputs =28*28
        num_hiddens=self.num_hiddens
        num_outputs = 10

        self.net = nn.Sequential(
                # unnamed layer, use layer1.parameters() or layer1.weight to access
                # named layer, use net.<name>.param to access
                OrderedDict([
                    ('flaten', nn.Flatten()),
                    ('input_sav1', nu.layer.gen_layer.InputSaviorLayer(f"{self.dir_path}/data/{self.vary_title}","input_af_quant",squeeze=False)),
                    ('linear_in', nn.Linear(num_inputs, num_hiddens,bias=False,device=self.device,)),                                
                    ('batch_norm1', nn.BatchNorm1d(num_hiddens)),
                    ('relu1',nn.ReLU()),

                    ('linear2', nn.Linear(num_hiddens, num_hiddens,bias=False,device=self.device)),                                
                    ('batch_norm2', nn.BatchNorm1d(num_hiddens)),
                    ('relu2',nn.ReLU()),

                    ('linear3', nn.Linear(num_hiddens, num_hiddens,bias=False,device=self.device)),                                
                    ('batch_norm3', nn.BatchNorm1d(num_hiddens)),
                    ('relu3',nn.ReLU()),

                    ('linear_out',nn.Linear(num_hiddens, num_outputs,bias=False,device=self.device)),
                ])
            ).to(self.device)
        
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
    
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
        self.results=nu.train.gen_train.train_funct(self.net, self.train_loader, self.test_loader, self.loss, self.num_epochs, 
            self.optimizer,device=self.device,quant_tensor=False, infer_iter=self.infer_loader)

