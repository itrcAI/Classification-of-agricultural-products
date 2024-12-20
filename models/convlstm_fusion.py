#
#TESTING FOR TSA ENCODING - final
#TSA BLOCK


"""

aaaa

"""
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy
from datetime import datetime


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dimc, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        

        """

        super(ConvLSTMCell, self).__init__()

        self.input_dimc = input_dimc
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        
        self.conv = nn.Conv1d(in_channels=self.input_dimc + self.hidden_dim,
                            out_channels=4*self.hidden_dim,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            bias=self.bias)

    
    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis
        combined_conv = self.conv(combined)
      
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
class convlstm(nn.Module):
    def __init__(self, input_dimc=1, hidden_dim=32, kernel_size=3,input_neuron = 128, output_dim=128,bias=False ):

        """
        input_dimc: must be 1
        hidden_dim: number of filtter. it must be power of 2 and same or biger than 16
        kernel_size: Size of kernel
        input_neuron: must be equal with last mlp2 layer
        output_dim:must be equal with first mlp4 layer
        bias: must be false
        """

        super(convlstm, self).__init__()


        self.inconv = nn.Conv2d(input_dimc,hidden_dim,(1,kernel_size))

        self.cell = ConvLSTMCell( 
                                input_dimc=hidden_dim,
                                hidden_dim=hidden_dim,
                                kernel_size=kernel_size,
                                bias=bias)
        

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dim * input_neuron, int((input_neuron*hidden_dim)/2)) 
        self.fc2 = nn.Linear(int((input_neuron*hidden_dim)/2), int((input_neuron*hidden_dim)/4)) 
        self.fc3 = nn.Linear(int((input_neuron*hidden_dim)/4), int((input_neuron*hidden_dim)/8)) 
        self.fc4 = nn.Linear(int((input_neuron*hidden_dim)/8), int((input_neuron*hidden_dim)/16)) 
        self.fc5 = nn.Linear(int((input_neuron*hidden_dim)/16), output_dim) 

    def forward(self, x, hidden=None, state=None):

        
        x = x.unsqueeze(2)
        # (b x t x c x h ) -> (b x c x t x h )
        x = x.permute(0,2,1,3)
        x = torch.nn.functional.pad(x, (1, 1), 'constant', 0)
        x = self.inconv.forward(x)
      
        b, c, t, h = x.shape

        if hidden is None:
            hidden = torch.zeros((b, c, h))
        if state is None:
            state = torch.zeros((b, c, h))
        
        if torch.cuda.is_available():
            hidden = hidden.cuda()
            state = state.cuda()
 
        for iter in range(t):
            hidden, state = self.cell.forward(x[:, :, iter, :], (hidden, state))

        x = state
        x = self.flatten(x)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        

        return x
