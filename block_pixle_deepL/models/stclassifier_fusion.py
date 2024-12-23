#

#STCLASSIFIER BLOCK
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, copy
from datetime import datetime

from models.pse_fusion import PixelSetEncoder 
from models.tae_fusion import TemporalAttentionEncoder 
from models.convlstm_fusion import convlstm 

from models.decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim_s1=2,input_dim_s2=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=False,
                 extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=55,
                 positions=None,
                 mlp4=[128, 64, 32, 12], fusion_type=None,hidden_dim=32, kernel_size=3,input_neuron = 128, output_dim=128):
        
        super(PseTae, self).__init__()
        

        self.s1_max_len = len_max_seq
        self.s2_max_len = len_max_seq
        
        self.early_seq_mlp1 = copy.deepcopy(mlp1)
        self.early_seq_mlp1[0] = input_dim_s1+input_dim_s2      
        self.positions = positions 
        

        # ----------------early fusion        
        self.spatial_encoder_earlyFusion = PixelSetEncoder(input_dim=self.early_seq_mlp1[0], mlp1=self.early_seq_mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra, extra_size=extra_size)    
        

        self.temporal_encoder_earlyFusion = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                        T=T, len_max_seq=self.s2_max_len, positions=positions)

        self.convlstm_earlyFusion = convlstm(input_dimc=1, hidden_dim=32, kernel_size=3,input_neuron = 128, output_dim=128,bias=False)  
         

        # ----------------pse fusion
        self.mlp1_s1 = copy.deepcopy(mlp1)
        self.mlp1_s1[0] = input_dim_s1 
        self.mlp3_pse = [1024, 512, 256]  
          

        self.spatial_encoder_s2 =  PixelSetEncoder(input_dim_s2, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        
        self.spatial_encoder_s1 = PixelSetEncoder(input_dim_s1, mlp1=self.mlp1_s1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                       extra_size=extra_size)
    

        self.temporal_encoder_pseFusion = TemporalAttentionEncoder(in_channels=mlp2[-1]*2, n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=self.mlp3_pse, dropout=dropout,
                                                        T=T, len_max_seq=self.s2_max_len, positions=positions) 
        
        
        # ------------------tsa fusion
        self.temporal_encoder_s2 = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=self.s2_max_len, positions=positions) 
        
        self.temporal_encoder_s1 = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=self.s1_max_len, positions=positions)
                                                        
        
        self.decoder = get_decoder(mlp4)
        
        # ------------------softmax averaging
        #self.decoder_tsa_fusion = get_decoder([128*2, 64, 32, 19])

        self.name = fusion_type
        self.fusion_type = fusion_type

        
    def forward(self, input_s1, input_s2, dates): 
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        start = datetime.now()
        
        if self.fusion_type == 'pse':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s2 = self.spatial_encoder_s2(input_s2)  
            out = torch.cat((out_s1, out_s2), dim=2)
            out = self.temporal_encoder_pseFusion(out, dates[1]) #indexed for sentinel-2 dates 
            out = self.decoder(out) 
            
            
        elif self.fusion_type == 'tsa':
            out_s1 = self.spatial_encoder_s1(input_s1)            
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) #indexed for sentinel-1 dates             
            out_s2 = self.spatial_encoder_s2(input_s2)            
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) #indexed for sentinel-2 dates             
            out = torch.cat((out_s1, out_s2), dim=1)
            out = self.decoder(out)
               
            
            
        elif self.fusion_type == 'softmax_norm':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) 
            out_s1 = self.decoder(out_s1)
            
            out_s2 = self.spatial_encoder_s2(input_s2)
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) 
            out_s2 = self.decoder(out_s2)
            
            out = torch.divide(torch.multiply(out_s1, out_s2), torch.sum(torch.multiply(out_s1, out_s2)))

        elif self.fusion_type == 'softmax_avg':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) 
            out_s1 = self.decoder(out_s1)
            
            out_s2 = self.spatial_encoder_s2(input_s2)
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) 
            out_s2 = self.decoder(out_s2)
            
            out = torch.divide(torch.add(out_s1, out_s2), 2.0)

        elif self.fusion_type == 'early':
                                                       
            input_s11,extra_fe = input_s1
            input_s22,extra_fe = input_s2                      
            
            data_s1, mask_s1 = input_s11
            data_s2, _ = input_s22
           
            data_s12 = torch.cat((data_s1, data_s2), dim=2)            
           
            input_s1122 =[data_s12, mask_s1] # mask_s1 = mask_s2
            
            out = [input_s1122,extra_fe]
                        
            out = self.spatial_encoder_earlyFusion(out)
            out = self.temporal_encoder_earlyFusion(out, dates[1]) #indexed for sentinel-2 dates
            out = self.decoder(out)
            
        
        
        elif self.fusion_type == 'convlstm':
                                                       
            input_s11,extra_fe = input_s1
            input_s22,extra_fe = input_s2                      
            
            data_s1, mask_s1 = input_s11
            data_s2, _ = input_s22
           
            data_s12 = torch.cat((data_s1, data_s2), dim=2)            
           
            input_s1122 =[data_s12, mask_s1] # mask_s1 = mask_s2
            
            out = [input_s1122,extra_fe]
                        
            out = self.spatial_encoder_earlyFusion(out)
            out = self.convlstm_earlyFusion(out) #indexed for sentinel-2 dates
            out = self.decoder(out)
            
        return out       


    def param_ratio(self):
        if self.fusion_type == 'pse':
            s = get_ntrainparams(self.spatial_encoder_s1)  + get_ntrainparams(self.spatial_encoder_s2)
            t = get_ntrainparams(self.temporal_encoder_pseFusion)
            c = get_ntrainparams(self.decoder)
            total = s + t + c
            
        elif self.fusion_type == 'tsa':
            s = get_ntrainparams(self.spatial_encoder_s1)  + get_ntrainparams(self.spatial_encoder_s2)
            t = get_ntrainparams(self.temporal_encoder_s1) + get_ntrainparams(self.temporal_encoder_s2)
            c = get_ntrainparams(self.decoder)
            total = s + t + c
            
        elif self.fusion_type == 'softmax_avg' or self.fusion_type == 'softmax_norm':
            s = get_ntrainparams(self.spatial_encoder_s1)  + get_ntrainparams(self.spatial_encoder_s2)
            t = get_ntrainparams(self.temporal_encoder_s1) + get_ntrainparams(self.temporal_encoder_s2)
            c = get_ntrainparams(self.decoder) * 2
            total = s + t + c
        
        elif self.fusion_type == 'early':  
            s = get_ntrainparams(self.spatial_encoder_earlyFusion)
            t = get_ntrainparams(self.temporal_encoder_earlyFusion)
            c = get_ntrainparams(self.decoder)
            total = s + t + c
            
        elif self.fusion_type == 'convlstm':  
            s = get_ntrainparams(self.spatial_encoder_earlyFusion)
            t = get_ntrainparams(self.convlstm_earlyFusion)
            c = get_ntrainparams(self.decoder)
            total = s + t + c

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Transformer {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
