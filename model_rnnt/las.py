import torch
import torch.nn as nn
from torch import Tensor

class ListenAttendSpell(nn.Module):

    def __init__(self, encoder, decoder, mode):
        super(ListenAttendSpell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mode = mode

    def forward(self, inputs, input_lengths, targets, teacher_forcing_ratio, return_decode_dict):
        enc_state, hidden = self.encoder(inputs, input_lengths)
        result2= []
        output_sequence = []

        if self.mode == "2nd_beam": # beam search inference
            result = self.decoder(targets, enc_state, 5)

        elif self.mode == "res":
            result = self.decoder(inputs, input_lengths, targets, enc_state, 1)
        
        else:
            result, output_sequence, result2 = self.decoder(targets, enc_state, teacher_forcing_ratio, return_decode_dict)
        
        return result, output_sequence, result2