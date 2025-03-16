import os
import torch
import torchvision
from torch import nn
from torch.nn import init
# import torchvision.transforms as transforms
import numpy as np
import sys

current_file = __file__
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.utils.data import random_split

import snntorch as snn
import tonic
from torch.utils.data import DataLoader, Subset
from tonic import DiskCachedDataset
from snntorch import functional as SF

import brevitas.nn as qnn

import torchvision.transforms as transforms
from collections import OrderedDict

import nnt_cli.utils as utils
from nnt_cli.templates.gen_temp import GeneralTemplate


class QSNN_Template(GeneralTemplate):

    def __init__(
        self,
        notebook_name,
        para_mode,
        debug_mode,
        batch_size,
        num_epochs,
        lr,
        beta,
        bit_width,
        num_hiddens,
        reset_mode,
        match_name=None,
        remark=None,
        learn_beta_threshold=True,
        
    ):
        """
        Args, are the parameters that need to be set in the certain cell with `parameters` tag in the notebook,
        which means, these parameters can be set by script.
        """

        self.para_mode = para_mode
        self.debug_mode = debug_mode
        self.batch_size = batch_size
        self.notebook_name = notebook_name
        self.num_epochs = num_epochs
        self.learning_rate = lr
        self.beta = beta
        self.bit_width=bit_width
        self.num_hiddens=num_hiddens
        self.reset_mode=reset_mode
        self.match_name=match_name
        self.remark=remark
        self.learn_beta_threshold=learn_beta_threshold

    def init_params(self):
        """
        The parameters here that don't need to be changed by script, only by initial setting.
        """
        super().init_params()

        if hasattr(self,'bit_width') and isinstance(self.bit_width, list):
            self.bit_width_1st=self.bit_width[0]
        elif hasattr(self,'bit_width') and isinstance(self.bit_width, int):
            self.bit_width_1st=self.bit_width

        self.infer_size=0.1
        # The size of the infer set. If it is 0.1, then the infer set will be 10% of the test set.
        # If it is 1, then the infer set will be the same as the test set.
        # The quantized data of infer_dataset will be saved.

        self.with_cache="False"
        # `with_cache` must be one of '2stage', 'Full', 'False'.

        self.quick_debug=False
        # Use quick_debug mode for a quick run with reduced sized dataset to verify the model.
        self.qb_dataset_size=0.3
        # The size of dataset when you are using quick_debug mode.

        self.n_mels=64
        # Number of mel filters.

        self.order=2
        # The delta order/dimension number of DeltaCalculator in pre_transform, that is added to feature.

        self.window=2
        # The delta window size of DeltaCalculator in pre_transform.

        if hasattr(self,'bit_width') and isinstance(self.bit_width, list):
            self.bit_width_1st=self.bit_width[0]
        elif hasattr(self,'bit_width') and isinstance(self.bit_width, int):
            self.bit_width_1st=self.bit_width
    
    def set_path(self):
        super().set_path()
        self.cache_path=f'{self.notebook_path}/cache/shd'

    def set_iter(self):
        self.vary_list=[1,2,4,8]
        self.variable_name="bit_width"
        # Must be consistent with the variable names in the class defined.

        self.vary_list2=[1024,2048,4096]
        self.variable_name2="num_hiddens"

        

    def set_dataset(self, preset=False):
        """
        The main features of SHD data sets
        Input data type: consisting of pulse sequence (spike trains) instead of static images or continuous signals.
        Data format: The data contains time information. Each input sample is a collection of pulse events on multiple time steps.
        Pulse encoding: The data has been encoded in a pulse sequence, and no additional pulse coding is required.
        Dimension:
        Input layer: 700 channels (that is, 700 auditory nerves).
        Output category: 20 categories (numbers 0-9, in English and German).
        Data source: pulse data converted based on TIMIT voice dataset.
        Training set/test set division:
        Training set: 8156 samples
        Test set: 2264 samples
        """
            
        sensor_size = tonic.datasets.SHD.sensor_size


        self.pre_transform = tonic.transforms.Compose([
                            tonic.transforms.ToFrame(
                                sensor_size=sensor_size,
                                time_window=10000), # [T,1,700]

                            utils.transform.gen_trans.FlattenTransform(), # [T,700]  # Flatten for Mel Matrix Multiplication
                            utils.transform.snn_trans.FixedTimeStepTrans(target_frames=85),  # [T_fixed,700]
                        ])
        
        train_post_transform = transforms.Compose([                                 

            utils.transform.snn_trans.MelSpectrogramEncoder(n_mels=self.n_mels), # [T_fixed,n_mels]
            utils.transform.snn_trans.DeltaCalculator(order=self.order,window=self.window), # [T_fixed,(order+1)*n_mels]

            # Data enhance
            utils.transform.snn_trans.RandomTimeWarp(max_shift=4),
            utils.transform.gen_trans.RandomChannelDropout(dropout_prob=0.1),

            utils.transform.gen_trans.ChannelPercentileNorm(percentile=97, abs=True), 
            # Because of the existence of logarithmic calculation, the result is negative.
            # And the value of this channel is negative in the entire time domain window, 
            # which may cause the Percentile value to be in the denominator and the absolute value 
            # to be very small, resulting in a negative extreme value.
            # Therefore, calculate abs value before PercentileNorm.
            torch.from_numpy,
            utils.transform.gen_trans.ToFloat(),
            
        ])

        test_post_transform = transforms.Compose([

            utils.transform.snn_trans.MelSpectrogramEncoder(n_mels=self.n_mels),
            utils.transform.snn_trans.DeltaCalculator(order=self.order,window=self.window),

            utils.transform.gen_trans.ChannelPercentileNorm(percentile=97, abs=True), 
            # Because of the existence of logarithmic calculation, the result is negative.
            # And the value of this channel is negative in the entire time domain window, 
            # which may cause the Percentile value to be in the denominator and the absolute value 
            # to be very small, resulting in a negative extreme value.
            # Therefore, calculate abs value before PercentileNorm.
            torch.from_numpy,
            utils.transform.gen_trans.ToFloat(),
            
        ])

        if not hasattr(self,"bit_width_1st"):
            train_post_transform=tonic.transforms.Compose([train_post_transform,
                                                            utils.transform.gen_trans.AddChannel(), # Add channel for Normalize
                                                            transforms.Normalize((0.5,), (0.5,)) # Only one channel here in SHD
                                                            ])
            test_post_transform=tonic.transforms.Compose([test_post_transform,
                                                            utils.transform.gen_trans.AddChannel(),  
                                                            transforms.Normalize((0.5,), (0.5,))
                                                            ])

        elif self.bit_width_1st==1:
            # It needs normalize when the bit_width is 1.
            train_post_transform=tonic.transforms.Compose([train_post_transform,
                                                            utils.transform.gen_trans.AddChannel(), # Add channel for Normalize
                                                            transforms.Normalize((0.0001,), (0.01,)) # Only one channel here in SHD
                                                            ])
            test_post_transform=tonic.transforms.Compose([test_post_transform,
                                                            utils.transform.gen_trans.AddChannel(),  
                                                            transforms.Normalize((0.0001,), (0.01,))
                                                            ])
        # It only needs to convert 0 to -1 when 1 bit is used, while other values ​​remain positive.
        else:
            train_post_transform=tonic.transforms.Compose([train_post_transform,
                                                            utils.transform.gen_trans.AddChannel(), # Add channel for Normalize
                                                            transforms.Normalize((0,), (0.01,)) # Only one channel here in SHD
                                                            ])
            test_post_transform=tonic.transforms.Compose([test_post_transform,
                                                            utils.transform.gen_trans.AddChannel(),  
                                                            transforms.Normalize((0,), (0.01,))
                                                            ])

        self.train_full_transform=tonic.transforms.Compose([self.pre_transform, train_post_transform])
        self.test_full_transform=tonic.transforms.Compose([self.pre_transform, test_post_transform])

        if preset is True:
            self.train_preset = tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=True)
            self.test_preset = tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=False)

        
        if self.with_cache == "2stage":
            # Using 2-stage transform to save cache space.
            # The first stage is to denoise and to frame the data, and data form is numpy, which consume less space than tensor form.
            trainset = tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=True)
            testset = tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.pre_transform, train=False)

            self.train_dataset = DiskCachedDataset(trainset, transform=train_post_transform, cache_path=f'{self.cache_path}/train')
            self.test_dataset = DiskCachedDataset(testset, transform=test_post_transform, cache_path=f'{self.cache_path}/test')
            print("Using two-stage transform to save cache space!")
        elif self.with_cache == "Full":
            # Full cache will use more than 100 GB space
            
            trainset=tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.train_full_transform, train=True)
            testset=tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.test_full_transform, train=False)
            self.train_dataset = DiskCachedDataset(trainset, cache_path=f'{self.cache_path}/train')
            self.test_dataset = DiskCachedDataset(testset, cache_path=f'{self.cache_path}/test')

            print("Using full transform to achieve full speed!")
        elif self.with_cache == "False":

            self.train_dataset=tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.train_full_transform, train=True)
            self.test_dataset=tonic.datasets.SHD(save_to=f'{self.parent_path}/dataset', transform=self.test_full_transform, train=False)
            print("Not using cache!")
        else:
            raise ValueError("with_cache must be one of '2stage', 'Full', 'False'")
        
        if self.quick_debug is True:
                # For quick debug
                trainset_size = range(int(len(self.train_dataset)*self.qb_dataset_size))
                testset_size = range(int(len(self.test_dataset)*self.qb_dataset_size))
                self.train_dataset = Subset(self.train_dataset, trainset_size)
                self.test_dataset = Subset(self.test_dataset, testset_size)

    def set_dataloader(self):
        if self.para_mode is True:
            num_workers=utils.settin.gen_settin.get_num_workers("Dist",dist_num=4)

        elif self.debug_mode is True:
            num_workers=utils.settin.gen_settin.get_num_workers("Half")

        else:
            num_workers=utils.settin.gen_settin.get_num_workers("Full")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                       collate_fn=utils.collate_fn.TensorTransposeBatch(batch_first=False),
                                       shuffle=True, num_workers=num_workers,pin_memory=True)

        if self.infer_size >=1:
            infer_subset=self.test_dataset
            test_subset=self.test_dataset
        else:
            test_subset, infer_subset = random_split(self.test_dataset, [1-self.infer_size, self.infer_size])

        self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, 
                                      collate_fn=utils.collate_fn.TensorTransposeBatch(batch_first=False),
                                      shuffle=False,num_workers=num_workers,pin_memory=True)
        self.infer_loader = DataLoader(infer_subset, batch_size=self.batch_size, 
                                       collate_fn=utils.collate_fn.TensorTransposeBatch(batch_first=False),
                                       shuffle=False, num_workers=num_workers,pin_memory=True)
        # There is no pad, because the time steps are already unified in transforms of dataset.

    def set_name(self):
        """
        For output filename
        By using `match_name`and adding index can make sure that multi times results will not be overwritten.
        Every time you run the code (not iteration), it will create a new folder with the next index.
        """
        self.match_name=f"{self.notebook_name}_qsnn_SHD_"
        self.remark=f"2_linear_layers"

    def set_model(self):
        """In subclass, do NOT use super() to inherit optimizer!!!"""
        num_inputs =2*34*34
        num_hiddens=self.num_hiddens
        num_outputs = 10

        act_quant=utils.settin.qnn_settin.get_act_quant(self.bit_width)
        weight_quant=utils.settin.qnn_settin.get_weight_quant(self.bit_width)
        self.net = nn.Sequential(
                # unnamed layer, use layer1.parameters() or layer1.weight to access
                # named layer, use net.<name>.param to access
                OrderedDict([
                    ('flaten', nn.Flatten()),
                    ('quant_ident', qnn.QuantIdentity(act_quant=act_quant,return_quant_tensor=True,
                                                    )),
                    ('input_sav1', utils.layer.gen_layer.InputSaviorLayer(f"{self.sav_data_path}/{self.vary_title}","input_af_quant",squeeze=False)),
                    ('quant_linear_in', qnn.QuantLinear(num_inputs, num_hiddens,bias=False,device=self.device,return_quant_tensor=True,
                                                    weight_quant=weight_quant,)),                                
                    ('batch_norm1', nn.BatchNorm1d(num_hiddens)),
                    ('leaky1', snn.Leaky(beta=self.beta, init_hidden=True,reset_mechanism="zero", 
                                         learn_beta=self.learn_beta_threshold, learn_threshold=self.learn_beta_threshold)),
                    ('quant_relu1',qnn.QuantReLU(bit_width=self.bit_width,return_quant_tensor=True,)),
                    ('quant_linear_out',qnn.QuantLinear(num_hiddens, num_outputs,bias=False,device=self.device,return_quant_tensor=True,
                                                    weight_quant=weight_quant,)),
                    ('leaky2', snn.Leaky(beta=self.beta, init_hidden=True, output=True, 
                                         learn_beta=self.learn_beta_threshold, learn_threshold=self.learn_beta_threshold))
                ])
            ).to(self.device)
        
        self.loss=SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

        self.scheduler=None
    
    def init_net(self):

        print("No init of the network!")
        # No init, because I'm not sure how to init the quantized layer.


    def set_train(self):
        num_steps = 0
        # Not used
        if self.load_cp_flag:
            self.results=utils.train.snn_train.train_snn(self.net,self.train_loader,self.test_loader,self.loss,self.num_epochs,self.optimizer,
                                    num_steps,infer_loader=self.infer_loader, SF_funct=True, in2spk=False, 
                                    forward=True, eve_in=True,device=self.device,debug_mode=self.debug_mode,
                                    checkpoint_path=self.checkpoint_path,mid_results=self.cp_mid_results,
                                    scheduler=self.scheduler)
        else:
            self.results=utils.train.snn_train.train_snn(self.net,self.train_loader,self.test_loader,self.loss,self.num_epochs,self.optimizer,
                                    num_steps,infer_loader=self.infer_loader, SF_funct=True, in2spk=False, 
                                    forward=True, eve_in=True,device=self.device,debug_mode=self.debug_mode,
                                    checkpoint_path=self.checkpoint_path,scheduler=self.scheduler)
    

    
    