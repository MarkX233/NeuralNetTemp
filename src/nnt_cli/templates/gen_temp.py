from abc import abstractmethod
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
import optuna

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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        self.iter_flag=False 
        self.onetime_flag=False
        self.double_iter_flag=False
        self.load_cp_flag=False
        self.optuna_flag=False

        self.train_method="one"

        self.vali_fr_train=True
        self.val_size=0.2
        # If True, the validation set will be split from the training set.
        # If False, the validation set will be the same as the test set.

        self.with_cache="False"
        # Using cache to store the dataset with transforms.
        # Setup in `set_dataset`.

        self.quick_debug=False
        self.qb_dataset_size=0.1
        # Used for quick debugging.
        # If True, the dataset will be a small subset of the original dataset.

        # Save switch
        self.sav_final = True
        self.sav_paras = True
        self.sav_state = False # Recommended to be False, use sav_checkpoint instead.
        self.sav_checkpoint = True
        

        self.record={}

        self.infer_size=0.1
        # The size of the infer set. If it is 0.1, then the infer set will be 10% of the test set.
        # If it is 1, then the infer set will be the same as the test set.
        # The data of infer set will be extracted.


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

    def perform_training(self, train_method='one'):
        match train_method:
            case "one":
                self.train_onetime(no_plot=True)
            case "iter":
                self.train_iter(no_plot=True)
            case "diter":
                self.train_double_iter(no_plot=True)
            case "cp":
                self.load_checkpoint(self.cp_fpath,no_plot=True)
            case "opt":
                self.optuna_optimize(study_name=self.notebook_name, db_url=self.db_url, n_trials=self.n_trials)
            case _:
                raise ValueError("Invalid train method!")

    def load_checkpoint(self,cp_fpath,no_plot=False):
        self.init_params()
        self.load_cp_flag=True
        self.train_method="cp"
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
        if hasattr(self, 'scheduler'):
            self.checkpoint=nu.sl.load_checkpoint(cp_fpath,model=self.net,optimizer=self.optimizer, scheduler=self.scheduler, device=self.device)
        else:
            self.checkpoint=nu.sl.load_checkpoint(cp_fpath,model=self.net,optimizer=self.optimizer, device=self.device)

        if 'train_method' not in self.checkpoint or self.checkpoint['train_method'] == 'one':
            self.set_train()
            self.set_store_onetime()
            self.set_store_record()
            if no_plot is False:
                self.plot_final()
                self.plot_record()
        elif self.checkpoint['train_method'] == 'iter':
            self._load_cp_params()
            self.vary_list=self.vary_list[self.cur_vari_index:]

            self.train_iter(no_plot=no_plot)
        elif self.checkpoint['train_method'] == 'diter':
            self._load_cp_params()
            self.vary_list=self.vary_list[self.cur_vari_index:]
            self.vary_list2=self.vary_list2[self.cur_vari_index2:]

            self.train_double_iter(no_plot=no_plot)



    def train_double_iter(self,no_plot=False):
        self.init_params()
        if self.load_cp_flag is False:
            self.double_iter_flag=True
            self.train_method="diter"
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
            self.cur_vari_index2=self.vary_list2.index(vari2)
            # For saving record
            for vari in tqdm(self.vary_list,desc="First Iteration"):
                setattr(self, self.variable_name, vari)
                self.vari1=vari
                self.cur_vari_index=self.vary_list.index(vari)
                print("====================================================")
                print(f"Training {self.variable_name}={getattr(self,self.variable_name)} and {self.variable_name2}={getattr(self,self.variable_name2)}")
                print(f"Progress:{self.cur_vari_index2+1} / {len(self.vary_list2)} of {self.cur_vari_index+1} / {len(self.vary_list)}")
                print("====================================================")
                self.set_vary_in_iter()
                self.set_model()
                self.init_net()
                self.set_additional_cp_dict()
                self.set_train()
                self.set_store_onetime()

        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()

    def train_iter(self,no_plot=False):
        self.init_params()
        if self.load_cp_flag is False:
            self.iter_flag=True
            self.train_method="iter"
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
            self.cur_vari_index=self.vary_list.index(vari)
            print("====================================================")
            print(f"Training {self.variable_name}={getattr(self,self.variable_name)}")
            print(f"Progress:{self.cur_vari_index+1} / {len(self.vary_list)}")
            print("====================================================")
            self.set_vary_in_iter()
            self.set_model()
            self.init_net()
            self.set_additional_cp_dict()
            self.set_train()
            self.set_store_onetime()
        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()
    
    def train_onetime(self,no_plot=False):
        self.init_params()
        self.onetime_flag=True
        self.train_method="one"
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
        self.set_additional_cp_dict()
        self.set_train()
        self.set_store_onetime()
        self.set_store_record()
        if no_plot is False:
            self.plot_final()
            self.plot_record()
    
    def train_optuna(self, trial):
        self.init_params()
        self.set_optuna(trial)
        self.optuna_flag=True
        self.train_method="opt"
        self.set_path()
        self.set_dataset()
        self.set_dataloader()
        if self.match_name is None or self.remark is None:
            self.set_name()
        self.set_dir(mkdir=False)
        self.set_results_path()
        self.set_vary_in_iter()
        self.set_model()
        self.init_net()
        self.set_train()

        return self.results[2][-1] # Test Accuracy
        # The return value is the test accuracy, which will be used by optuna to optimize the parameters.
    
    def optuna_optimize(self, study_name, db_url, n_trials=100, unlock=True, mk_db=True):
        """
        Optimize the hyperparameters using optuna.
        
        Args:
            study_name: The name of the study.
            db_url: The url of the database. For example, db_url = "sqlite:////path/to/your/study.db"
            n_trials: The number of trials.
            unlock: If True, unlock the database, when KeyboardInterrupt.
            mk_db: If True, create the database if it doesn't exist.
        """
        from optuna.storages import RDBStorage, RetryFailedTrialCallback

        from nnt_cli.utils.settin.gen_settin import ensure_db_exists, cleanup_sqlite_locks

        # Check if the database is locked, if so, unlock it.
        cleanup_sqlite_locks(db_url)

        # Check if the database exists, if not, create it.
        ensure_db_exists(db_url,mk_db=mk_db)

        storage = RDBStorage(
            url=db_url,
            engine_kwargs={
                "connect_args": {"timeout": 30},
                "pool_size": 20
            },
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3), # Retry failed trials
        )
        try:
            study = optuna.create_study(
                                        study_name=study_name,
                                        storage=storage,
                                        direction='maximize',
                                        load_if_exists=True, # Load the study if it exists.
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.MedianPruner())
        except optuna.exceptions.DuplicateStudyError: # in case the study is created by another process
            study = optuna.load_study(
                                        study_name=study_name,
                                        storage=storage)
        try:
            study.optimize(self.train_optuna, n_trials=n_trials)
        except KeyboardInterrupt:
            print("Optimization stopped by user.")
            if unlock:
                cleanup_sqlite_locks(db_url)
            

    def train_continue(self,no_plot=False):
        """
        Do another training round after one time training.
        Used only in debug and testing.
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

    @abstractmethod
    def set_dataset(self):
        """
        Set the dataset here.
        """
        self.train_dataset=None
        self.test_dataset=None
        print("`set_dataset` is not implemented.")

    @abstractmethod
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

    @abstractmethod
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

    def set_dir(self,mkdir=True):
        """
        Set the directory here.
        The directory will be created in the `results` folder in the current notebook path.
        """
        self.findex=nu.sl.get_next_demo_index(f"{self.notebook_path}/results",self.match_name,"dir")

        self.dirname=f"{self.match_name}{self.findex}"

        self.suptitle=f"{self.dirname}-{self.remark}"

        self.dir_path=f"{self.notebook_path}/results/{self.dirname}"

        if mkdir is True:
            os.makedirs(self.dir_path)
        print(f"Current work directory: {self.dir_path}")

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
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        if self.onetime_flag is True:
            self.vary_title="Onetime"
        elif self.optuna_flag is True:
            self.vary_title="Optuna"
        elif self.load_cp_flag is True:
            self.vary_title="Load Checkpoint"
        elif self.double_iter_flag is True:
            self.vary_title=f"{self.variable_name}{getattr(self,self.variable_name)}-{self.variable_name2}{getattr(self,self.variable_name2)}"
        elif self.iter_flag is True:
            self.vary_title=f"{self.variable_name}{getattr(self,self.variable_name)}"
            
        # `vary_title` is for the name of the final results.

    @abstractmethod
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

    def set_additional_cp_dict(self):
        """
        Set additional checkpoint dictionary to save.
        """
        self.cp_add_dict={
                "train_method":self.train_method,
                "dirname": self.dirname,
                "remark": self.remark,
                "match_name": self.match_name,
                "suptitle": self.suptitle,
                # "dir_path": self.dir_path,
                "res_path": self.res_path,
                "paras_path": self.paras_path,
                "vary_title": self.vary_title,
            }
        if hasattr(self, 'variable_name'):
            self.cp_add_dict["variable_name"]=self.variable_name
        if hasattr(self, 'vary_list'):
            self.cp_add_dict["vary_list"]=self.vary_list
        if hasattr(self, 'variable_name2'):
            self.cp_add_dict["variable_name2"]=self.variable_name2
        if hasattr(self, 'vary_list2'):
            self.cp_add_dict["vary_list2"]=self.vary_list2
        if hasattr(self, 'cur_vari_index'):
            self.cp_add_dict["cur_vari_index"]=self.cur_vari_index
        if hasattr(self, 'cur_vari_index2'):
            self.cp_add_dict["cur_vari_index2"]=self.cur_vari_index2

    @abstractmethod
    def set_train(self):
        """Set the training process here. 
        The results should be a list containing `Train Loss`, `Train Accuracy`, `Validation Accuracy`, `Infer Accuracy`, 'Test Accuracy.
        Or a dictionary with the same key
        These key names and corresponding data will be used for plotting.
        If you use a dictionary, all the other data besides these key name will also be saved.
        """

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
        
        if isinstance(self.results, list): # For old version compatibility
            self.record[f"{self.vary_title}"] = {
                    "Train Loss": self.results[0],
                    "Train Accuracy": self.results[1],
                    "Validation Accuracy": self.results[2],
                    "Infer Accuracy": self.results[3],
                }
            if len(self.results) > 4:
                self.record["Test Accuracy"] = self.results[4]
        elif isinstance(self.results, dict): # After version 0.4.10
            self.record[f"{self.vary_title}"] = {
                    "Train Loss": self.results.get('Train Loss'),
                    "Train Accuracy": self.results.get('Train Accuracy'),
                    "Validation Accuracy": self.results.get('Validation Accuracy'),
                    "Infer Accuracy": self.results.get('Infer Accuracy'),
                    "Test Accuracy": self.results.get('Test Accuracy'),
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
            if hasattr(self, 'scheduler'):
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

        # # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
        torch.backends.cudnn.deterministic = True

        # #this flag enables cudnn for some operations such as conv layers and RNNs, 
        # # which can yield a significant speedup.
        # torch.backends.cudnn.enabled = False

        # # This flag enables the cudnn auto-tuner that finds the best algorithm to use
        # # for a particular configuration. (this mode is good whenever input sizes do not vary)
        # torch.backends.cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)

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
    
    def get_visual_frame_distribution(self, dataset_num=0, quantile=0.99, time_distri=True, font_scale=1.0):
        """
        Plot frame value distribution from dataset
        """

        self.init_params()
        self.set_path()
        try:
            self.set_dataset(preset=True)
        except TypeError:
            print("It seems that dataset doesn't have preset kwarg.")
            self.set_dataset()

        if hasattr(self, 'train_preset') and self.train_preset is not None:
            frame0, _ = self.train_preset[dataset_num]

            if time_distri:
                nu.plot.anal_plot.plot_time_step_distribution(self.train_preset, font_scale=font_scale)

            nu.plot.anal_plot.visualize_event_distribution(frame0,title="Pre-Transformed Frames Event Distribution Analysis", font_scale=font_scale)
            nu.plot.anal_plot.visualize_value_distribution(frame0,title="Pre-Transformed Frames Value Distribution Analysis", font_scale=font_scale)
        else:
            print("Warning! `train_preset` is not set yet.")

        if hasattr(self, 'train_preset') and self.train_dataset is not None:

            frame,label = self.train_dataset[dataset_num]

            nu.plot.anal_plot.visualize_value_distribution(frame,title="Full Transformed Frames Value Distribution Analysis", font_scale=font_scale)
            nu.plot.anal_plot.visualize_distribution_robust(frame,title="Full Transformed Frames Filter Value Distribution Analysis", quantile=quantile)
        else:
            print("Warning! `train_dataset` is not set yet.")
    
    def check_gradient_norm(self):
        """
        Callable only after training.
        """
        grad_norms = [p.grad.norm().item() for p in self.net.parameters()]
        print(f"Gradient norm: {np.mean(grad_norms):.3e} ± {np.std(grad_norms):.3e}, the normal value should be between 1E-6 and 1E-3")

    @abstractmethod
    def set_optuna(self, trial):
        self.sav_final = False
        self.sav_paras = False
        self.sav_state = False
        self.sav_checkpoint = False

    def _load_cp_params(self):
        setattr(self, "dirname", self.checkpoint['dirname'])
        setattr(self, "remark", self.checkpoint['remark'])
        setattr(self, "match_name", self.checkpoint['match_name'])
        setattr(self, "suptitle", self.checkpoint['suptitle'])
        setattr(self, "vary_title", self.checkpoint['vary_title'])

        setattr(self, "variable_name", self.checkpoint.get('variable_name', None))
        setattr(self, "vary_list", self.checkpoint.get('vary_list', None))
        setattr(self, "cur_vari_index", self.checkpoint.get('cur_vari_index', None))
        setattr(self, "variable_name2", self.checkpoint.get('variable_name2', None))
        setattr(self, "vary_list2", self.checkpoint.get('vary_list2', None))
        setattr(self, "cur_vari_index2", self.checkpoint.get('cur_vari_index2', None))