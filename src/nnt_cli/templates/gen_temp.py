import os
from pathlib import Path
import torch

# import torchvision.transforms as transforms

import sys

current_file = __file__
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tqdm import tqdm

import random
import numpy as np

import shutil

import nnt_cli.utils as nu


class GeneralTemplate():
    """
    The None parts are the parts that need to be implemented.
    If you use the default structure, you only need to implement:
        `set_dataset`, `set_dataloader`, `set_model`, `set_train`.
    
    It is recommended to implement `__init__` and `init_params` to set the needed parameters,
    especially the ones need to be changed by script, if you want to do different training with different params.

    If you want to do iteration train, you need to implement `set_iter`. The store and plot functions are already implemented
    and will automatically deal with the data.
    """
    def __init__(self,**kwargs):
        """
        Args, are the parameters that need to be set in the certain cell with `parameters` tag in the notebook,
        which means, these parameters can be set by script.
        """

        for key, value in kwargs.items():
            setattr(self, key, value)

    def init_params(self):
        """
        The parameters here that don't need to be changed by script, only by initial setting.
        """
        self.iter_flag=False
        self.onetime_flag=False
        self.double_iter_flag=False
        self.load_cp_flag=False

        self.sav_final = True
        self.sav_paras = True
        self.sav_state = True
        self.sav_checkpoint = True
        # Save switch

        self.record={}

        self.infer_size=0.1
        # The size of the infer set. If it is 0.1, then the infer set will be 10% of the test set.
        # If it is 1, then the infer set will be the same as the test set.


    def set_iter(self):
        """
        Set the iteration parameters here.
        The vary_list is the list of the values that need to be iterated.
        The variable_name is the name of the variable that need to be iterated
        The first one will be used in 'iter' mode, both will be used in 'diter' mode.
        """
        self.vary_list=[1,2,4,8]
        self.variable_name="bit_width"
        # Must be consistent with the variable names in the class defined.

        self.vary_list2=[64,128,256,512]
        self.variable_name2="num_hiddens"

    def load_checkpoint(self,cp_fpath,no_plot=False):
        self.init_params()
        self.load_cp_flag=True
        self.set_path()
        self.set_dataset()
        if self.match_name is None or self.remark is None:
            self.set_name()
        self.set_dir()
        self.set_results_path()
        self.set_model()
        self.init_net()
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            _,self.cp_mid_results=nu.sl.load_checkpoint(self.net,self.optimizer,cp_fpath, scheduler=self.scheduler)
        else:
            _,self.cp_mid_results=nu.sl.load_checkpoint(self.net,self.optimizer,cp_fpath)
        # Here we don't need loss, 'cause set_model has defined loss.
        self.set_train()
        self.set_store_onetime()
        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()



    def train_double_iter(self,no_plot=False):
        self.init_params()
        self.double_iter_flag=True
        self.set_path()
        self.set_dataset()
        self.set_dataloader()
        if self.match_name is None or self.remark is None:
            self.set_name()
        self.set_dir()
        self.set_results_path()
        self.set_iter()
        for vari2 in tqdm(self.vary_list2,desc="Second Iteration"):
            setattr(self, self.variable_name2, vari2)
            self.vari2=vari2
            # For saving record
            for vari in tqdm(self.vary_list,desc="First Iteration"):
                setattr(self, self.variable_name, vari)
                self.vari1=vari
                print("====================================================")
                print(f"Training {self.variable_name}={getattr(self,self.variable_name)} and {self.variable_name2}={getattr(self,self.variable_name2)}")
                print(f"Progress:{self.vary_list2.index(vari2)+1} / {len(self.vary_list2)} of {self.vary_list.index(vari)+1} / {len(self.vary_list)}")
                print("====================================================")
                self.set_vary_in_iter()
                self.set_model()
                self.init_net()
                self.set_train()
                self.set_store_onetime()

        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()

    def train_iter(self,no_plot=False):
        self.init_params()
        self.iter_flag=True
        self.set_path()
        self.set_dataset()
        self.set_dataloader()
        if self.match_name is None or self.remark is None:
            self.set_name()
        self.set_dir()
        self.set_results_path()
        self.set_iter()
        for vari in tqdm(self.vary_list,desc="Iteration"):
            setattr(self, self.variable_name, vari)
            self.vari1=vari
            # For saving record
            print("====================================================")
            print(f"Training {self.variable_name}={getattr(self,self.variable_name)}")
            print(f"Progress:{self.vary_list.index(vari)+1} / {len(self.vary_list)}")
            print("====================================================")
            self.set_vary_in_iter()
            self.set_model()
            self.init_net()
            self.set_train()
            self.set_store_onetime()
        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()
    
    def train_onetime(self,no_plot=False):
        self.init_params()
        self.onetime_flag=True
        self.set_path()
        self.set_dataset()
        self.set_dataloader()
        if self.match_name is None or self.remark is None:
            self.set_name()
        self.set_dir()
        self.set_results_path()
        self.set_vary_in_iter()
        self.set_model()
        self.init_net()
        self.set_train()
        self.set_store_onetime()
        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()
    
    def train_continue(self,no_plot=False):
        """
        Do another training round after one time training.
        Used in debug and testing
        """
        self.set_train()
        self.set_store_onetime()
        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()



    def set_path(self):

        self.notebook_path = Path().resolve()

        self.parent_path = os.path.join(self.notebook_path, "..")


    def set_dataset(self):
        """
        Set the dataset here.
        """
        self.train_dataset=None
        self.test_dataset=None
        print("`set_dataset` is not implemented.")

    def set_dataloader(self):
        if self.para_mode is True:
            num_workers=nu.settin.gen_settin.get_num_workers("Dist",dist_num=4)

        elif self.debug_mode is True:
            num_workers=nu.settin.gen_settin.get_num_workers("Half")

        else:
            num_workers=nu.settin.gen_settin.get_num_workers("Full")


        self.test_loader = None
        self.infer_loader = None
        print("`set_dataloader` is not implemented.")

    def set_name(self):
        """
        For output filename
        By using `match_name`and adding index can make sure that multi times results will not be overwritten.
        Every time you run the code (not iteration), it will create a new folder with the next index.

        Example::

            self.match_name=f"{self.notebook_name}_qsnn_NMNIST_"
            self.remark=f"2_linear_layers"
        """
        print("`set_name` is not implemented.")
        pass

    def set_dir(self):
        self.findex=nu.sl.get_next_demo_index(f"{self.notebook_path}/results",self.match_name,"dir")

        self.dirname=f"{self.match_name}{self.findex}"

        self.suptitle=f"{self.dirname}-{self.remark}"

        self.dir_path=f"{self.notebook_path}/results/{self.dirname}"

        os.makedirs(self.dir_path)
        print(f"Current work directory: {self.dir_path}")
        # Make dir first, in case multi task running at the same time.

    def set_results_path(self):

        self.sav_data_path = f"{self.dir_path}/data"
        self.res_path = f"{self.dir_path}/rec"
        self.paras_path = f"{self.dir_path}/params"
        self.state_path = f"{self.dir_path}/state"
        if self.sav_checkpoint:
            self.checkpoint_path=f"{self.dir_path}/checkpoint"
        else:
            self.checkpoint_path=None


    def set_vary_in_iter(self):
        """
        The variables need to be set in the iteration (when there is one).
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


        if self.onetime_flag is False:
            if self.double_iter_flag is True:
                self.vary_title=f"{self.variable_name}{getattr(self,self.variable_name)}-{self.variable_name2}{getattr(self,self.variable_name2)}"
            elif self.iter_flag is True:
                self.vary_title=f"{self.variable_name}{getattr(self,self.variable_name)}"
        else:
            self.vary_title="Onetime"
        # `vary_title` is for the name of the final results.


    def set_model(self):
        """Set the model here."""
        self.net = None
        
        self.loss=None

        self.optimizer = None

        self.scheduler= None

        print("`set_model` is not implemented.")
    
    def init_net(self):

        """Initialize the network here."""
        pass

        # for _, layer in self.net.named_modules():
        #     if isinstance(layer, nn.Linear):
        #         init.normal_(layer.weight, mean=0, std=0.01)
        #         if layer.bias is not None:
        #             init.constant_(layer.bias, val=0)
            # if isinstance(layer, snn.Leaky):
            #     init.normal_(layer.threshold, mean=0.5, std=0.01)
            #     init.normal_(layer.beta, mean=0.3, std=0.01)

    def set_train(self):
        """Set the training process here. 
        The results should be a list containing `Train Loss`, `Train Accuracy`, `Test Accuracy`, `Infer Accuracy`."""

        if self.load_cp_flag is True:
        # If you want to use load checkpoint, you should complete this branch.
            self.results=None
        else:
            self.results=None

        print("`set_train` is not implemented.")
    
    # def load_cp_and_train(self,cp_fpath):
    #     self.results=None

    #     print("`load_cp_and_train` is not implemented.")
        
    def set_store_onetime(self):
        
        self.record[f"{self.vary_title}"] = {
                "Train Loss": self.results[0],
                "Train Accuracy": self.results[1],
                "Test Accuracy": self.results[2],
                "Infer Accuracy": self.results[3],
             }

        if self.sav_final is True and self.onetime_flag is False:
            if self.double_iter_flag is True:
                # `store_final_results` will store the final results in the csv file. It meat to be used to store for one time final result.
                # So vary_x, vary_y and vary_z should be the value, not the list!
                nu.sl.store_final_results(self.results,self.vary_title,
                                       vary_x=self.vari1,vary_x_name=self.variable_name,
                                        vary_y=self.vari2, vary_y_name=self.variable_name2, 
                                        total_name=f"{self.variable_name}-{self.variable_name2}",
                                        path=self.res_path,
                                        vary_z=self.dirname, vary_z_name="Match Name",
                                        single_file=False)
            elif self.iter_flag is True:
                nu.sl.store_final_results(self.results,self.vary_title,
                                       vary_x=self.vari1,vary_x_name=self.variable_name,
                                        vary_y=None, vary_y_name=None, 
                                        total_name=f"{self.variable_name}",
                                        path=self.res_path,
                                        vary_z=self.dirname, vary_z_name="Match Name",
                                        single_file=False)
            
            
        
        if self.sav_paras is True:
            nu.sl.sav_lin_net_paras(self.net,self.paras_path,
                                 self.vary_title,overwrite=True)

        if self.sav_state is True:
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                nu.sl.sav_net_state(self.net,self.state_path,
                             self.vary_title,optimizer=self.optimizer,loss=self.loss,scheduler=self.scheduler)
            else:
                nu.sl.sav_net_state(self.net,self.state_path,
                             self.vary_title,optimizer=self.optimizer,loss=self.loss)
        
    
    def set_store_record(self):
        nu.sl.store_record(self.record,self.remark,path=self.res_path,HPID_en=False)

    def plot_final(self):
        
        if self.onetime_flag is False:
            if self.double_iter_flag is True:
                nu.sl.load_final_csv_and_plot(f"{self.res_path}/{self.variable_name}-{self.variable_name2}_total_results.csv",
                                           xname=self.variable_name,yname=self.variable_name2)
            elif self.iter_flag is True:
                nu.sl.load_final_csv_and_plot(f"{self.res_path}/{self.variable_name}_total_results.csv",
                                           xname=self.variable_name,yname=None)
        else:
            print("No final result plot because you are running in onetime mode.")
    
    def plot_record(self):
        nu.sl.load_record_and_plot(f"{self.res_path}/{self.remark}.csv")

    def set_seed(self,seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
        torch.backends.cudnn.deterministic = True

        #this flag enables cudnn for some operations such as conv layers and RNNs, 
        # which can yield a significant speedup.
        torch.backends.cudnn.enabled = False

        # This flag enables the cudnn auto-tuner that finds the best algorithm to use
        # for a particular configuration. (this mode is good whenever input sizes do not vary)
        torch.backends.cudnn.benchmark = False

        # I don't know if this is useful, look it up.
        #os.environ['PYTHONHASHSEED'] = str(seed)

    def check_dataset(self,cache_mode=None):
        """
        Check if the dataset is is complete.
        Directly callable.
        """
        self.init_params()
        if cache_mode is not None and isinstance(cache_mode,str):
            self.with_cache=cache_mode

        self.set_path()
        self.set_dataset()
        self.set_dataloader()

        for idx in tqdm(range(len(self.train_dataset))):
            try:
                sample = self.train_dataset[idx]
            except KeyError as e:
                print(f"Error with index {idx}: {e}")
    
    def remove_cache(self):
        """
        If you are having trouble with the dataset while training, for example,
            KeyError: Caught KeyError in DataLoader worker process 0.
            ...
            KeyError: "Unable to open object (object 'data' doesn't exist)"
        Call check_dataset(with_cache=True) to see if the dataset is complete.
        If you see errors like:
            Error with index xxxxx: "Unable to open object (object 'data' doesn't exist)"
        Then you can try to remove the cache to fix the error.
        """
        try:
            self.set_path()
            shutil.rmtree(self.cache_path)
            print(f"Cache removed scuccessfully!")
        except FileNotFoundError:
            print(f"Cache not found!")

    def dataset_analyze(self):
        """
        Get the global average, standard deviation, min and max value per channel of all time steps, all batches.

        Make sure you set proper pre_transform first.

        For manually call.
        """
        self.init_params()
        self.set_path()
        self.set_dataset()

        # If the following functions are using dataloader and dataloader is using collate_fn, 
        # the results could have deviation because of padding or interception.

        trainset_mean,trainset_std=nu.data_analysis.compute_mean_std_loader(self.train_loader, no_channel=True)   # If you use flatten in dataset transform, here will be no channel.
        testset_mean,testset_std=nu.data_analysis.compute_mean_std_loader(self.test_loader, no_channel=True)
        # Use dataloader to calculate
        # Need less RAM

        print(f"Trainset mean and std, calculated by dataloader: {trainset_mean}, {trainset_std}")
        print(f"Testset mean and std, calculated by dataloader: {testset_mean}, {testset_std}")

        # trainset_mean2,trainset_std2=du.compute_mean_std_dataset(trainset)
        # testset_mean2,testset_std2=du.compute_mean_std_dataset(testset)
        # # Use dataset to calculate
        # # Need more RAM

        # print(f"Trainset, calculated by dataset: {trainset_mean2}, {trainset_std2}")
        # print(f"Testset, calculated by dataset: {testset_mean2}, {testset_std2}")

        trainset_min, trainset_max=nu.data_analysis.compute_min_max_loader(self.train_loader, no_channel=True)
        testset_min, testset_max=nu.data_analysis.compute_min_max_loader(self.test_loader, no_channel=True)

        print(f"Trainset min and max, calculated by dataloader: {trainset_min}, {trainset_max}")
        print(f"Testset min and max, calculated by dataloader: {testset_min}, {testset_max}")

        time_len_dict=nu.data_analysis.analyze_time_length(self.train_dataset)

        print(f"Time length: {time_len_dict}")
    
    def get_visual_frame_distribution(self, dataset_num=0, quantile=0.99):
        """
        Plot frame value distribution from dataset
        """

        self.init_params()
        self.set_path()
        try:
            self.set_dataset(preset=True)
        except RuntimeError:
            print("It seems that dataset doesn't have preset kwarg.")
            self.set_dataset()

        if self.train_preset is not None:
            frame0, _ = self.train_preset[dataset_num]

            nu.plot.anal_plot.plot_time_step_distribution(self.train_preset)

            nu.plot.anal_plot.visualize_event_distribution(frame0,title="Pre-Frames Event Distribution Analysis")
            nu.plot.anal_plot.visualize_value_distribution(frame0,title="Pre-Frames Value Distribution Analysis")
        else:
            print("Warning! `train_preset` is not set yet.")

        if self.train_dataset is not None:

            frame,label = self.train_dataset[dataset_num]

            nu.plot.anal_plot.visualize_value_distribution(frame,title="Full Transformed Frames Value Distribution Analysis")
            nu.plot.anal_plot.visualize_distribution_robust(frame,title="Full Transformed Frames Filter Value Distribution Analysis", quantile=quantile)
        else:
            print("Warning! `train_dataset` is not set yet.")
    
    def check_gradient_norm(self):
        """
        Callable only after training.
        """
        grad_norms = [p.grad.norm().item() for p in self.net.parameters()]
        print(f"Gradient norm: {np.mean(grad_norms):.3e} ± {np.std(grad_norms):.3e}, the normal value should be between 1E-6 and 1E-3")