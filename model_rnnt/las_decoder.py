import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor, LongTensor
import torch.nn.init as init

class Speller(nn.Module):
    KEY_ATTENTION_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE_SYMBOL = 'sequence_symbol'

    def __init__(self, vocab_size, embedding_size, max_length, 
                hidden_dim, pad_id, sos_id, eos_id, attention_head, 
                num_layers, projection_dim, dropout_p, device, beam_mode):
        super(Speller, self).__init__()

        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.input_dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.attention = nn.MultiheadAttention(embed_dim=projection_dim,
                                                num_heads=attention_head)
        self.projection = nn.Linear(hidden_dim, projection_dim, bias=True)
        #self.projection1 = nn.Linear(hidden_dim, projection_dim, bias=True)
        self.generator = nn.Linear(projection_dim, vocab_size, bias=True)
        self.beam_mode = beam_mode
        self.device = device

        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p)

    def forward_step(self, input_var, hidden, encoder_outputs, attn):        
        batch_size, output_lengths = input_var.size(0), input_var.size(1)
        
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        
        #if self.training:        
        self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)
        output = self.projection(output)

        output = output.transpose(0, 1).contiguous()

        encoder_outputs = encoder_outputs.transpose(0, 1)
    
        context, attn = self.attention(output, encoder_outputs, encoder_outputs)

        attn_score = torch.zeros(attn.size(0),1).to(self.device)
        
        for batch_idx in range(attn.size(0)):
            attn_score[batch_idx] = attn[batch_idx][torch.where(attn[batch_idx] > 0.5)].sum()

        context = context.transpose(0, 1).contiguous()
        #output = self.projection(context)
        output = context
        
        output = self.generator(output)
        step_output = F.log_softmax(output, dim=-1)
        normal_step_outputs = F.softmax(output, dim=-1)
        
        return step_output, hidden, attn, normal_step_outputs

    def forward(self, inputs, encoder_outputs, teacher_forcing_ratio, return_decode_dict):
        hidden, attn = None, None
        result, decode_dict = list(), dict()      
        result2 = list()
        if not self.training:
            decode_dict[Speller.KEY_ATTENTION_SCORE] = list()
            decode_dict[Speller.KEY_SEQUENCE_SYMBOL] = list()

        inputs, batch_size, max_length = self.validate_args(inputs, encoder_outputs, teacher_forcing_ratio) # inference를 위한거 크게 신경 x
        
        inputs_add_sos = torch.LongTensor([self.sos_id]*batch_size).view(batch_size, 1)
        
        if inputs.is_cuda: inputs_add_sos = inputs_add_sos.cuda()
        
        inputs = inputs.long()        
        inputs = torch.cat((inputs_add_sos, inputs), dim=1)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        lengths = np.array([max_length] * batch_size)
        
        if use_teacher_forcing:
            output_sequence = None
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
            step_outputs, hidden, attn, normal_step_outputs = self.forward_step(inputs, hidden, encoder_outputs, attn)
             
            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                result.append(step_output)
                normal_step_output = normal_step_outputs[:, di, :]
                result2.append(normal_step_output)

        else:
            result2 = []
            output_sequence = None
            input_var = inputs[:, 0].unsqueeze(1)
            count = 0
            for di in range(max_length):

                step_output, hidden, attn, normal_step_outputs = self.forward_step(input_var, hidden, encoder_outputs, attn)
                    
                result.append(step_output.squeeze())
                input_var = result[-1].topk(1)[1]
                    
                #학습 아닐때    
                if not self.training:
                    decode_dict[Speller.KEY_ATTENTION_SCORE].append(attn)
                    decode_dict[Speller.KEY_SEQUENCE_SYMBOL].append(input_var)
                    eos_batches = input_var.data.eq(self.eos_id)
                        
                    if eos_batches.dim() > 0:
                        eos_batches = eos_batches.cpu().view(-1).numpy()
                        update_idx = ((lengths > di) & eos_batches) != 0
                        lengths[update_idx] = len(decode_dict[Speller.KEY_SEQUENCE_SYMBOL])
            
        del decode_dict

        return result, output_sequence, result2

    def validate_args(self, inputs, encoder_outputs, teacher_forcing_ratio):
        # input script
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        # input이 없을때만 발동 -> 즉 inference할때만
        if inputs is None:  # inference
            inputs = LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")

        else:
            max_length = inputs.size(1)  # minus the start of sequence symbol

        return inputs, batch_size, max_length
