from brevitas.quant_tensor.base_quant_tensor import QuantTensor
from tqdm import tqdm
import torch

from nnt_cli.utils.sl import save_checkpoint
from nnt_cli.utils.data_trans import float_quant_tensor2int


def train_funct(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs,
    optimizer,
    infer_iter=None,
    device="cpu",
    quant_tensor=False,
    checkpoint_path=None,
    mid_results=None
):
    
    """
    A general train function to train network not involving time.

    Args:
        quant_tensor (bool):
            Whether to transform network output from quant tensor of brevitas to int tensor, before calculating loss.
    """
    checkpoint_sav_period= 5 # unit: epoch

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



    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

        net.train()
        for X, y in tqdm(train_iter, desc=f"Training: Epoch {epoch+1}/{num_epochs}", unit="batch",leave=False):
            # X: feature tensor, shape [batch_size, 1*28*28]. y: label tensor, shape [batch_size].
            # 1 batch size of dataset for one time
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X)
            
            if isinstance(optimizer, list):
                for opt in optimizer:
                    # print(opt)
                    opt.zero_grad()
            else:
                optimizer.zero_grad()
            
            if quant_tensor is True:
                y_hat_int=float_quant_tensor2int(y_hat)
            else:
                y_hat_int=y_hat
            
            l = loss(y_hat_int, y).to(device)  # .sum(), input is a list or array

            l.backward()

            if isinstance(optimizer, list):
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()

            train_l_sum += l.item()

            if isinstance(y_hat_int, QuantTensor):
                y_hat_int=float_quant_tensor2int(y_hat_int)
            # Transform quant tensor to int tensor to calculate accuracy

            train_acc_sum += (
                (y_hat_int.argmax(dim=1) == y).sum().item()
            )  # argmax returns index
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, device=device,quant_tensor=quant_tensor)
        train_l_list.append(train_l_sum / n)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f \n'
        #         % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        last_epoch=False
        if epoch==num_epochs-1:
            last_epoch=True

        if infer_iter is not None:
            infer_acc = evaluate_accuracy(infer_iter, net, device=device,quant_tensor=quant_tensor, sav_layer_en=last_epoch)
            infer_acc_list.append(infer_acc)
        else:
            infer_acc=0
            infer_acc_list.append(infer_acc)    # For plot

        if checkpoint_path is not None and epoch % checkpoint_sav_period == 0:
            # For now the supported amount of optimizer is one.
            save_checkpoint(net,optimizer,epoch,loss,train_l_list,train_acc_list,test_acc_list,infer_acc_list,checkpoint_path)

        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_l_sum / n:.4f}, Train Acc: {train_acc_sum / n:.3f}, Test Acc: {test_acc:.3f}, Infer Acc: {infer_acc:.3f}")

    return [train_l_list, train_acc_list, test_acc_list, infer_acc_list]

def evaluate_accuracy(
    data_iter, net, params=None, device="cpu", quant_tensor=False, sav_layer_en=False
):  
    """
    Evaluating accuracy for training function.

    """
    from layer.gen_layer import InputSaviorLayer
    acc_sum, n = 0.0, 0
    if isinstance(net, torch.nn.Module):
        for _, layer in net.named_modules():
            if isinstance(layer, InputSaviorLayer) and sav_layer_en is True:
                layer.sav_en = True
        net.eval()
    with torch.no_grad():
        for X, y in tqdm(data_iter, desc="Evaluating", unit="batch",leave=False):
            X = X.to(device)
            y = y.to(device)
            if params is not None:
                acc_sum += (net(X, params=params).argmax(dim=1) == y).float().sum().item()
            else:
                y_hat=net(X)
                if quant_tensor is True:
                    y_hat_int=float_quant_tensor2int(y_hat)
                    acc_sum += (y_hat_int.argmax(dim=1) == y).float().sum().item()
                else:
                    if isinstance(y_hat, QuantTensor):
                        y_hat_int=float_quant_tensor2int(y_hat)
                    else:
                        y_hat_int=y_hat
                    acc_sum += (y_hat_int.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]

    if isinstance(net, torch.nn.Module):
        for _, layer in net.named_modules():
            if isinstance(layer, InputSaviorLayer) and layer.sav_en is True:
                layer.sav_en = False
        net.train()

    return acc_sum / n