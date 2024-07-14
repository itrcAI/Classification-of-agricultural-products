#

#STCLASSIFIER BLOCK
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, copy
from datetime import datetime

from models.pse_fusion import PixelSetEncoder 
from models.tae_fusion import TemporalAttentionEncoder 
from models.decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=False,
                 extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 positions=None,
                 mlp4=[128, 64, 32, 12], fusion_type=None):
        
        super(PseTae, self).__init__()
        

        self.s1_max_len = 75
        self.s2_max_len = 27
        self.early_seq_mlp1 = [12, 32, 64]
        self.positions = positions 
        

        # ----------------early fusion        
        self.spatial_encoder_earlyFusion = PixelSetEncoder(input_dim=self.early_seq_mlp1[0], mlp1=self.early_seq_mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra, extra_size=extra_size)    
        

        self.temporal_encoder_earlyFusion = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                        T=T, len_max_seq=self.s2_max_len, positions=positions)

        # ----------------pse fusion
        self.mlp1_s1 = copy.deepcopy(mlp1)
        self.mlp1_s1[0] = 2 
        self.mlp3_pse = [1024, 512, 256]  
          

        self.spatial_encoder_s2 =  PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        
        self.spatial_encoder_s1 = PixelSetEncoder(self.mlp1_s1[0], mlp1=self.mlp1_s1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
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

            data_s1, mask_s1 = input_s1
            data_s2, _ = input_s2

            data = torch.cat((data_s1, data_s2), dim=2)
             
            out = (data, mask_s1) # mask_s1 = mask_s2
            out = self.spatial_encoder_earlyFusion(out)
            out = self.temporal_encoder_earlyFusion(out, dates[1]) #indexed for sentinel-2 dates
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

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##########starts er
class PseTae_pretrained(nn.Module):

    def __init__(self, weight_folder, hyperparameters, device='cuda'):
        """
        Pretrained PseTea classifier.
        The class can either load the weights of a single fold or aggregate the predictions of the different sets of
        weights obtained during k-fold cross-validation and produces a single prediction.
        Args:
            weight_folder (str): Path to the folder containing the different sets of weights obtained during each fold
            (res_dir of the training script)
            hyperparameters (dict): Hyperparameters of the PseTae classifier
            device (str): Device on which the model should be loaded ('cpu' or 'cuda')
        """
        super(PseTae_pretrained, self).__init__()
        self.weight_folder = weight_folder
        self.hyperparameters = hyperparameters     
        
        self.model_instances = []

        m = PseTae(**hyperparameters)
        if device == 'cpu':
            map_loc = 'cpu'
        else:
            map_loc = 'cuda:{}'.format(torch.cuda.current_device())
            m = m.cuda()
        d = torch.load(os.path.join(weight_folder, 'model.pth.tar'), map_location=map_loc)
        m.load_state_dict(d['state_dict'])
        self.model_instances.append(m)
        print('Successfully loaded {} model instances')
########################################er end but doeasnot test
    def forward(self, input_s1, input_s2, dates):
        """ Returns class logits
        Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            outputs = [F.log_softmax(m(input_s1, input_s2, dates), dim=1) for m in self.model_instances]
            outputs = torch.stack(outputs, dim=0).mean(dim=0)
        return outputs

    def predict_class(self, input_s1, input_s2, dates):
        """Returns class prediction
                Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            pred = self.forward(input_s1, input_s2, dates).argmax(dim=1)
        return pred


