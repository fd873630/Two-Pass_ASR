import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

class TopK_RNN_T_Decoder(nn.Module):
    def __init__(self, rnn_t, beam_size, device):
        super(TopK_RNN_T_Decoder, self).__init__()
        self.ongoing_beams = None
        self.cumulative_ps = None
        self.encoder = rnn_t.encoder
        self.decoder = rnn_t.decoder
        #self.eos_id = rnn_t.decoder.eos_id 
        self.joint = rnn_t.joint
        self.project_layer = rnn_t.project_layer
        self.device = device

        self.beam_size = beam_size
        self.vocab_size = rnn_t.decoder.vocab_size
        self.beam_search = rnn_t.new_beam_search

    def forward(self, inputs, inputs_lengths, W):
        batch_size, hidden, attn = inputs.size(0), None, None
    
        enc_state, _ = self.encoder(inputs, inputs_lengths)
        enc_state = self.project_layer(enc_state)

        total = torch.zeros(batch_size * W, 55)
        best_total = torch.zeros(batch_size, 55)

        hypothesis = []
        for batch_idx in range(batch_size):
            
            B = self.beam_search(enc_state[batch_idx, :, :], W, self.device)
            
            hypothesis.append(B)

        start_index = 0
        count = 0
        for i, unit in enumerate(hypothesis):
            
            for index in range(W):
                a = unit[index].k[1:]
                a.append(53)
                target_len = len(a)
                
                if index == 0:
                    best_total[count, :target_len] = torch.LongTensor(a)
                    count += 1

                total[start_index + index, :target_len] = torch.LongTensor(a)
                
            start_index += W
        
        total = total.long()
        best_total = best_total.long()
        
        return total, best_total