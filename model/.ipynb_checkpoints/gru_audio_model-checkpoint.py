
import torch
import torch.nn as nn
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class GRUAudioConfig:
    input_size: int= 1
    cond_size: int = 3
    hidden_size: int = 48
    num_layers: int = 4
    output_size: int = 256
    dropout: float = 0.1


class RNN(nn.Module):
     # input size - the number of "classes"
    def __init__(self, config: GRUAudioConfig):
        super(RNN, self).__init__()
        self.config = config
        
        self.input_size = config.input_size
        self.cond_size = config.cond_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.num_layers = config.num_layers #no. of stacked GRU layers
        
        self.i2h = nn.Linear(self.input_size+self.cond_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, 
                          dropout=config.dropout if config.num_layers > 1 else 0.0)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

 
    # input and cv are each one sequence element 
    def forward(self, input, hidden, batch_size=1):
        #print("input size is " + str((input.size())))
        
        h1 = self.i2h(input)
        #print("size of h1 is " + str(h1.size()))
        
        h_out, hidden = self.gru(h1.view(batch_size,1,-1), hidden)
        #print("h_out"+str(h_out.size()))
        
        logits = self.decoder(h_out.view(batch_size,-1))
        #print("output2"+str(output.size()))
        
        return logits, hidden
 
    # initialize hiddens for each minibatch
    def init_hidden(self,batch_size=1):
        return .1*torch.rand(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.gru.weight_hh_l0.device)-.05


