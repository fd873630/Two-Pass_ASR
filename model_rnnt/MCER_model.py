import torch
import torch.nn as nn
import numpy as np
from model_rnnt.hangul import moasseugi
from model_rnnt.eval_distance import eval_cer

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue
            
            index, char = line.split('   ')
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char

def compute_cer(preds, labels):
    char2index, index2char = load_label('./label,csv/hangul.labels')

    count = 0    
    total_cer = 0
    total_cer_len = 0

    units = []
    units_pred = []
    for a in labels:
        if a == 53: # eos
            break
        units.append(index2char[a])
            
    for b in preds:
        if b == 53: # eos
            break
        units_pred.append(index2char[b])
            
    label = moasseugi(units)
    pred = moasseugi(units_pred)

    cer = eval_cer(pred, label)
    cer_len = len(label.replace(" ", ""))
        
    return cer, cer_len

class MCER_model(nn.Module):
    def __init__(self, rnn_t, las, eos, device):
        super(MCER_model, self).__init__()

        self.rnn_t = rnn_t
        self.las = las
        self.device = device
        self.eos = eos

    def forward(self, inputs, input_lengths, targets, teacher_forcing_ratio, beam_width, return_decode_dict):   
        
        # 이거 다했는데 output이 gather이 안됨 그런까 beam result를 tensor로 바꾸고 train 함수에서 계산하기 

        beam_result = self.rnn_t.MCER_beam_search(inputs, input_lengths, W=beam_width)
        print(beam_result.shape)
        beam_cer = 0 
        beam_len = 0
        beam_probability = 0 # 
        loss_MWER = 0

        targets = targets.squeeze()

        for j in range(beam_width):
            probability = 1

            inference_target = torch.zeros(1, 55).to(self.device) # 55는 max len
            inference = beam_result[j].k[1:] # RNN-T beam list
            
            inference.append(self.eos)           
            inference = torch.tensor(inference) 
            seq_length = inference.size(0)
                
            inference_target[0,:seq_length] = inference
            inference_target.long()
                
            _, _, normal_logits = self.las(inputs, input_lengths, inference_target, 1, False)
            
            normal_logits = torch.stack(normal_logits, dim=1).to(self.device)
            hypothesis_p, hypothesis = normal_logits.max(-1) # hypothesis_p : P(yi|x), hypothesis : ym
            
            result = [] # ym
            
            for i, q in enumerate(hypothesis.squeeze()):
                probability *= hypothesis_p.squeeze()[i].item() # P(ym|x)     
                result.append(q.item())

                if q.item() == self.eos:
                    break
            
            beam_probability += probability # sum yi<Hm P(yi|x)

            cer, cer_len = compute_cer(result, targets.cpu().tolist())
                        
            beam_cer += cer
            beam_len += cer_len
    
        beam_mean_cer = beam_cer/beam_width # W-(y*,Hm) : mean number of word errors for Hm

        for l in range(beam_width):
            
            inference_target = torch.zeros(1, 55).to(self.device) # 55는 max len
            inference = beam_result[l].k[1:] # RNN-T beam list

            inference.append(self.eos)           
            inference = torch.tensor(inference) 
            seq_length = inference.size(0)

            inference_target[0,:seq_length] = inference
            inference_target.long()

            _, _, normal_logits = self.las(inputs, input_lengths, inference_target, 1, False)
            
            normal_logits = torch.stack(normal_logits, dim=1).to(self.device)
            hypothesis_p, hypothesis = normal_logits.max(-1) # hypothesis_p : P(yi|x), hypothesis : ym
            
            result = [] # ym
            
            for i, q in enumerate(hypothesis.squeeze()):
                probability *= hypothesis_p.squeeze()[i].item() # P(ym|x)     
                result.append(q.item())

                if q.item() == self.eos:
                    break
            
            P_hat = probability / beam_probability # P^(ym|x,Hm) = P(ym|x) / sum yi<Hm P(yi|x)
            
            cer, cer_len = compute_cer(result, targets.cpu().tolist())
            
            W_hat = cer - beam_mean_cer # W^(y*,ym) = W(y*,ym) - W-(y*,Hm)

            loss_MWER += P_hat * W_hat


        return loss_MWER

