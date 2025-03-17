import torch
from snntorch import utils
from tqdm import tqdm

from nnt_cli.utils.layer.gen_layer import InputSaviorLayer
from nnt_cli.utils.sl import save_checkpoint

def rate_encoding_images(images, time_steps=10):
    """
    Encode images to spike train and reshape.

    Args:
        images (list): 
            [batch_size, 1, 28, 28]
        time_steps (int): 
            Number of time steps
    Returns: 
        lists:
            [time_steps, batch_size, 784]
    """
    # images, _ = next(iter(data_loader))
    batch_size, _, height, width = images.shape
    images = images.view(batch_size, -1)  # [batch_size, 784]
    spike_sequence = torch.bernoulli(images.unsqueeze(0).repeat(time_steps, 1, 1))  # [time_steps, batch_size, 784]
    return spike_sequence.view(time_steps, batch_size, height*width)


class forward_pass():
    """
    For the net that doesn't contain Time Iteration.

        spkin: The input is already coded spikes and length is defined as num_steps.

        valin: The input is value.

        evein: The input is event-based frame, the length is uncertain.
    """
    def __init__(self,net,num_steps,data,sav_layer_en):
        self.mem_rec = []
        self.spk_rec = []
        self.num_steps=num_steps
        self.data=data
        utils.reset(net)  # resets hidden states for all LIF neurons in net
        self.net=net

        self.sav_layer_en=sav_layer_en
        
        
    def spkin(self):
        for step in range(self.num_steps):
            if self.sav_layer_en:
                self.enable_sav_layer(step)
            spk_out, mem_out = self.net(self.data[step])
            self.spk_rec.append(spk_out)
            self.mem_rec.append(mem_out)

        return torch.stack(self.spk_rec), torch.stack(self.mem_rec)
    def valin(self):
        for step in range(self.num_steps):
            if self.sav_layer_en:
                self.enable_sav_layer(step)
            spk_out, mem_out = self.net(self.data)
            self.spk_rec.append(spk_out)
            self.mem_rec.append(mem_out)

        return torch.stack(self.spk_rec), torch.stack(self.mem_rec)
    
    def evein(self):
        for step in range(self.data.size(0)):
            if self.sav_layer_en:
                self.enable_sav_layer(step)
            spk_out, mem_out = self.net(self.data[step])
            self.spk_rec.append(spk_out)
            self.mem_rec.append(mem_out)

        return torch.stack(self.spk_rec), torch.stack(self.mem_rec)
    
    def enable_sav_layer(self,step):
        for _, layer in self.net.named_modules():
                    if isinstance(layer, InputSaviorLayer):
                        layer.time_count=step   # The setting here must correspond to du.InputSaviorLayer.
                        layer.last_one=False
                        if step == self.data.size(0)-1:
                            layer.last_one=True

def train_snn(net, train_loader, test_loader, loss, num_epochs, optimizer, num_steps,
               forward=True, eve_in=False, SF_funct=False,  infer_loader=None, in2spk=False, 
               device="cpu",sampler=None,debug_mode=False,checkpoint_path=None,mid_results=None,
               scheduler=None):
    """
    A general training function for Spiking Neural Networks (SNN).

    Args:
        net (nn.Module): The SNN model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        loss (function): Loss function used for training (e.g., CrossEntropyLoss).
        num_epochs (int): Number of training epochs.
        optimizer (Optimizer or list): Optimizer(s) for updating network parameters.
        num_steps (int): Number of timesteps for SNN simulation.
        forward (bool): Whether to use a forward pass (useful when input data can't directly be sent into network).
        eve_in (bool): Whether the input is an event-based frame.
        SF_funct (bool): Whether to apply a loss function from snntorch.functional (e.g., SF.mse_count_loss).
        infer_loader (DataLoader, optional): DataLoader for an additional inference dataset.
            If it is None, the results of inference accuracy will return a list of 0.
        in2spk (bool): Whether to encode input as spikes (e.g., using rate encoding).
        device (str): The device to run the training on (e.g., "cpu" or "cuda").
        sampler (Sampler, optional): Sampler for the training dataset, used for DDP.
        debug_mode (bool): Whether to print debug information during training.
        checkpoint_path (str, optional): Path to save the checkpoint during training. If it is None, there will be no saving of checkpoint.
        mid_results (list, optional): List containing the state of training to resume from a checkpoint.\
            The list should contain [last_epoch_num, train_l_list, train_acc_list, test_acc_list, infer_acc_list].

    Returns:
        list: Training loss, training accuracy, test accuracy, and inference accuracy lists.
    """

    checkpoint_sav_period= 5 # unit: epoch

    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

    if mid_results is None:
        train_l_list=[]
        train_acc_list=[]
        infer_acc_list=[]
        test_acc_list=[]
        epoch_range=range(num_epochs)
    else:
        if len(mid_results) != 5:
            raise ValueError("Wrong input length of mid_results!")
        
        last_epoch_num=mid_results[0]
        train_l_list=mid_results[1]
        train_acc_list=mid_results[2]
        test_acc_list=mid_results[3]
        infer_acc_list=mid_results[4]
        if last_epoch_num < num_epochs:
            epoch_range=range(last_epoch_num+1,num_epochs)
        else:
            raise ValueError("The epoch loaded from checkpoint is not smaller than the num_epoch settings.")



    for epoch in epoch_range:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for frames, label in tqdm(train_loader, desc=f"Training: Epoch {epoch+1}/{num_epochs}", unit="batch",leave=False): 
        # For NMNIST, ToFrame and no `batch_first`:
        # frames: feature tensor, shape [time_length, batch_size, 2*34*34]. label: label tensor, shape [batch_size].
        # 1 batch size of dataset for one time
            frames=frames.to(device)
            label=label.to(device)

            utils.reset(net)  # resets hidden states for all LIF neurons in net
            net.train()

            if isinstance(optimizer,list):
                for opt in optimizer:
                    # print(opt)
                    opt.zero_grad()
            else:
                optimizer.zero_grad()

            # print(frames.shape)

            spk_rec=torch.tensor([],device=device)
            mem_rec=torch.tensor([],device=device)
            
            spk_rec,mem_rec=net_run(net,frames,num_steps,in2spk,forward,eve_in)
            
            _, label_hat_id=spk_rec.sum(dim=0).max(1) # Output: Rate coding
            
            # initialize the total loss value
            loss_val = torch.zeros((1), dtype=torch.float, device=device)

            if SF_funct is True:
                loss_val=loss(spk_rec, label)
            else:
                for step in range(num_steps):
                    step_loss = loss(mem_rec[step], label)      # cross entrophy.
                    loss_val = loss_val + step_loss
                # loss_val = loss(y_hat, label).sum().to(device)    # .sum(), input is a list or array
            
            loss_val.backward()
            
            if isinstance(optimizer,list):
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()
            # opt_i.step()    # update params
            train_l_sum = train_l_sum + loss_val.item()
            train_acc_sum = train_acc_sum + (label_hat_id == label).float().sum().item()
            n = n + label.shape[0]

            del loss_val, frames, label, spk_rec, mem_rec
            # release memory

        if scheduler is not None:
            scheduler.step()
        # Update the learning rate at the end of each epoch

        test_acc = eval_acc(test_loader, net, num_steps, device=device,in2spk=in2spk,forward=forward,eve_in=eve_in)
        train_l_list.append(train_l_sum / n)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f \n'
        #         % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

        flag_last_epoch=False
        if epoch==num_epochs-1:
            flag_last_epoch=True

        if infer_loader is not None:
            infer_acc = eval_acc(infer_loader, net, num_steps, device=device,in2spk=in2spk,
                                 forward=forward,eve_in=eve_in,sav_layer_en=flag_last_epoch)
            infer_acc_list.append(infer_acc)
        else:
            infer_acc=0
            infer_acc_list.append(infer_acc)    # For plot

        if checkpoint_path is not None and epoch % checkpoint_sav_period == 0:
            if scheduler is not None:
                save_checkpoint(net,optimizer,epoch,loss,train_l_list,train_acc_list,test_acc_list,infer_acc_list,checkpoint_path, scheduler=scheduler)
            # For now the supported amount of optimizer is one.
            else:
                save_checkpoint(net,optimizer,epoch,loss,train_l_list,train_acc_list,test_acc_list,infer_acc_list,checkpoint_path)

        if debug_mode is True:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_l_sum / n:.4f}, Train Acc: {train_acc_sum / n:.3f}, Test Acc: {test_acc:.3f}, Infer Acc: {infer_acc:.3f}")

        

        if isinstance(device,str) and device.startswith("cuda") or \
            isinstance(device,torch.device) and device.type == "cuda":
            torch.cuda.empty_cache()
    if isinstance(device,str) and device.startswith("cuda") or \
        isinstance(device,torch.device) and device.type == "cuda":     
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return [train_l_list, train_acc_list, test_acc_list, infer_acc_list]


def eval_acc(data_iter, net, num_steps, device="cpu",in2spk=False,forward=False,eve_in=False,sav_layer_en=False):
    acc_sum, n = 0.0, 0
    if isinstance(net, torch.nn.Module):
        for _, layer in net.named_modules():
            if isinstance(layer, InputSaviorLayer) and sav_layer_en is True:
                layer.sav_en = True
                layer.with_time_scale=True

    net.eval()
    with torch.no_grad():
        
        # if isinstance(net, torch.nn.Module):
        #     # for _, layer in net.named_modules():
        #     #     if isinstance(layer, DropoutLayer):
        #     #         layer.training=False

        for frames, label in tqdm(data_iter, desc="Evaluating", unit="batch", leave=False):
            frames=frames.to(device)
            label=label.to(device)

            spk_rec,_=net_run(net,frames,num_steps,in2spk,forward,eve_in,sav_layer_en=sav_layer_en)
            
            _, y_hat_id=spk_rec.sum(dim=0).max(1)
            
            acc_sum = acc_sum + (y_hat_id == label).float().sum().item()
            n = n + label.shape[0]

        # if isinstance(net, torch.nn.Module):
        #     # for _, layer in net.named_modules():
        #     #     if isinstance(layer, DropoutLayer):
        #     #         layer.training=True
        # net.train()

    if isinstance(net, torch.nn.Module):
        for _, layer in net.named_modules():
            if isinstance(layer, InputSaviorLayer) and layer.sav_en is True:
                layer.sav_en = False
        

    return acc_sum / n
    
    
    
def net_run(net,frames,num_steps,in2spk,forward,eve_in,sav_layer_en=False):
    if in2spk is True and eve_in is True:
        raise ValueError("Error! Wrong settings of in2spk and eve_in! These two can't stay True together!")
            
    if in2spk is True and forward is False:
        raise ValueError("Error! Wrong settings of in2spk and forward! When `in2spk` is set to True, `forward` must also be set to True.")
    
    if eve_in is True and forward is False:
        raise ValueError("Error! Wrong settings of eve_in and forward! When `eve_in` is set to True, `forward` must also be set to True.")


    if in2spk is True and forward is True:
        spk_in=rate_encoding_images(frames,num_steps)
        # spk_in.to(device)
        # explicit rate encoding
        # if forward is True:
        # When net contains no time step iteraion, especially using nn.Sequential and etc. pack.
        spk_rec, mem_rec=forward_pass(net,num_steps,spk_in,sav_layer_en=sav_layer_en).spkin()
        # else:
        #     spk_rec, mem_rec=net(spk_in)
    elif eve_in is True and forward is True:
        spk_rec, mem_rec=forward_pass(net,num_steps,frames,sav_layer_en=sav_layer_en).evein()
    
    elif forward is True:
        spk_rec, mem_rec=forward_pass(net,num_steps,frames,sav_layer_en=sav_layer_en).valin()
    else:
        # frames=frames.view(frames.shape[0], -1)
        # print(frames.shape)
        spk_rec, mem_rec=net(frames)

    return spk_rec, mem_rec