import torch
import torch.nn as nn
from torch import Tensor

class ListenAttendSpell(nn.Module):

    def __init__(self, encoder, decoder):
        super(ListenAttendSpell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, input_lengths, targets, teacher_forcing_ratio, return_decode_dict):
        
        output, hidden = self.encoder(inputs, input_lengths)
        
        result = self.decoder(targets, output, teacher_forcing_ratio, return_decode_dict)

        return result