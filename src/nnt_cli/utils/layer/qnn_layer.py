import snntorch as snn
import torch
import torch.nn as nn

from nnt_cli.utils.settin.qnn_settin import FloatQuantizerBeta, FloatQuantizerThreshold

class QuantLeaky(snn.Leaky):
    """
    Quantized version of Leaky layer.
    If using learnable beta, the beta parameter will be quantized using FloatQuantizerBeta before preforming forward().
    If using learnable threshold, the threshold parameter will be quantized using FloatQuantizerThreshold before preforming forward().
    """
    def __init__(self, beta, init_hidden=False,reset_mechanism="subtract",
                  learn_beta=False, learn_threshold=False,output=False,
                  
                  ):
        
        # if isinstance(beta, float):
        #         beta=torch.tensor(beta)
        
        # with torch.no_grad():
        #     beta = FloatQuantizerBeta.tensor_quant(beta)

        super().__init__(beta=beta, init_hidden=init_hidden, reset_mechanism=reset_mechanism, 
                         learn_beta=learn_beta, learn_threshold=learn_threshold, output=output)
        
        self.learn_beta=learn_beta
        self.learn_threshold=learn_threshold
        

    def forward(self, input_, mem=None):
        if self.learn_beta:
            # print(f"Beta in device: {self.beta.device}", 
            #       f"Quantizer in device: {FloatQuantizerBeta.flag_tensor.device}")
            if self.beta.device != FloatQuantizerBeta.flag_tensor.device:
                beta_temp=self.beta.clone().detach().requires_grad_(False).to(FloatQuantizerBeta.device)
                # beta_temp = self.beta
                with torch.no_grad():
                    beta_temp = FloatQuantizerBeta.tensor_quant(beta_temp)
                    temp = beta_temp[0].to(input_.device)
                    self.beta = nn.Parameter(temp).to(input_.device)
                # print("Betas are in different devices",self.beta)
            else:
                self.beta = nn.Parameter(FloatQuantizerBeta.tensor_quant(self.beta)[0]).to(input_.device)
                # print("Betas are in same device",self.beta)
        if self.learn_threshold:
            if self.beta.device != FloatQuantizerThreshold.flag_tensor.device:
                threshold_temp=self.threshold.clone().detach().requires_grad_(False).to(FloatQuantizerThreshold.device)
                with torch.no_grad():
                    threshold_temp = FloatQuantizerThreshold.tensor_quant(threshold_temp)
                    temp = threshold_temp[0].to(input_.device)
                    self.threshold = nn.Parameter(temp).to(input_.device)
            else:
                self.threshold = nn.Parameter(FloatQuantizerThreshold.tensor_quant(self.threshold)[0]).to(input_.device)
            # print(self.threshold)
        return super().forward(input_, mem)