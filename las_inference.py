import os
import time
import yaml
import random
import shutil
import argparse
import datetime
import editdistance
import scipy.signal
import numpy as np 

# torch 관련
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from warprnnt_pytorch import RNNTLoss

from model_rnnt.eval_distance import eval_wer, eval_cer
from model_rnnt.model import Transducer, JointNet
from model_rnnt.encoder import BaseEncoder
from model_rnnt.decoder import BaseDecoder
from model_rnnt.las_decoder import Speller
from model_rnnt.las import ListenAttendSpell
from model_rnnt.topk_decoder import TopKDecoder
from model_rnnt.hangul import moasseugi
from model_rnnt.data_loader_deepspeech import SpectrogramDataset, AudioDataLoader, AttrDict

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        with torch.no_grad():
            label_smoothed = torch.zeros_like(logit)
            label_smoothed.fill_(self.smoothing / (self.vocab_size - 1))
            label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
            label_smoothed[target == self.ignore_index, :] = 0
            
        return torch.sum(-label_smoothed * logit)

def compute_cer(preds, labels):
    char2index, index2char = load_label('./label,csv/hangul.labels')

    count = 0    
    total_wer = 0
    total_cer = 0

    total_wer_len = 0
    total_cer_len = 0

    for label, pred in zip(labels, preds):
        units = []
        units_pred = []
        for a in label:
            if a == 53: # eos
                break
            units.append(index2char[a])
            
        for b in pred:
            if b == 53: # eos
                break
            units_pred.append(index2char[b])
            
        label = moasseugi(units)
        pred = moasseugi(units_pred)

        wer = eval_wer(pred, label)
        cer = eval_cer(pred, label)
        
        wer_len = len(label.split())
        cer_len = len(label.replace(" ", ""))

        total_wer += wer
        total_cer += cer

        total_wer_len += wer_len
        total_cer_len += cer_len
        
    return total_wer, total_cer, total_wer_len, total_cer_len

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

def las_inference(model, val_loader, device):
    model.eval()

    eos_id = 53
    total_num = 0

    total_cer = 0
    total_cer_len = 0

    with open("./las_inference.txt", "w") as f:
        f.write('\n')
        f.write("inference 시작")
        f.write('\n')

    start_time = time.time()
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets, inputs_lengths, targets_lengths = data
            total_num += sum(targets_lengths)
            inputs_lengths = torch.IntTensor(inputs_lengths)
            targets_lengths = torch.IntTensor(targets_lengths)

            inputs = inputs.to(device) # (batch_size, time, freq)
            targets = targets.to(device)
            inputs_lengths = inputs_lengths.to(device)
            targets_lengths = targets_lengths.to(device)

            logits, _, _ = model(inputs, inputs_lengths, targets, 0, False)
            logits = torch.stack(logits, dim=1).to(device)

            hypothesis = logits.max(-1)[1]

            hypothesis = hypothesis.squeeze()
            _, cer, _, cer_len = compute_cer(hypothesis.cpu().numpy(),targets.cpu().numpy())        
            '''
            for a, b in zip(targets,hypothesis):
                chars = []
                predic_chars = []
                
                for w in a:
                    if w == 53: # eos
                        break
                    chars.append(index2char[w])

                for y in b:
                    if y == 53: # eos
                        break
                    predic_chars.append(index2char[y])
                
                with open("./las_inference.txt", "a") as f:
                    f.write('\n')
                    f.write(moasseugi(chars))
                    f.write('\n')
                    f.write(moasseugi(predic_chars))
                    f.write('\n')
            '''
            total_cer += cer
            total_cer_len += cer_len
    
    final_cer = (total_cer / total_cer_len) * 100

    return final_cer

def main():
    
    yaml_name = "/home/jhjeong/jiho_deep/two_pass/label,csv/Two_Pass.yaml"

    with open("./las_only_train.txt", "w") as f:
        f.write(yaml_name)
        f.write('\n')
        f.write('\n')
        f.write("학습 시작")
        f.write('\n')

    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    windows = { 'hamming': scipy.signal.hamming,
                'hann': scipy.signal.hann,
                'blackman': scipy.signal.blackman,
                'bartlett': scipy.signal.bartlett
                }

    SAMPLE_RATE = config.audio_data.sampling_rate
    WINDOW_SIZE = config.audio_data.window_size
    WINDOW_STRIDE = config.audio_data.window_stride
    WINDOW = config.audio_data.window

    audio_conf = dict(sample_rate=SAMPLE_RATE,
                        window_size=WINDOW_SIZE,
                        window_stride=WINDOW_STRIDE,
                        window=WINDOW)

    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    #-------------------------- Model Initialize --------------------------
    #Prediction Network
    enc = BaseEncoder(input_size=config.model.enc.input_size,
                      hidden_size=config.model.enc.hidden_size, 
                      output_size=config.model.enc.output_size,
                      n_layers=config.model.enc.n_layers, 
                      dropout=config.model.dropout, 
                      bidirectional=config.model.enc.bidirectional) 

    las_dec = Speller(vocab_size=config.model.vocab_size,
                      embedding_size=config.model.las_dec.embedding_size,
                      max_length=config.model.las_dec.max_length, 
                      hidden_dim=config.model.las_dec.hidden_size, 
                      pad_id=0, 
                      sos_id=52, 
                      eos_id=53, 
                      attention_head=config.model.las_dec.attention_head, 
                      num_layers=config.model.las_dec.n_layers,
                      projection_dim=config.model.las_dec.projection_dim, 
                      dropout_p=config.model.dropout, 
                      device=device,
                      beam_mode=False)
    
    #top_dec = TopKDecoder(las_dec, 1)
    las_model = ListenAttendSpell(enc, las_dec, "false").to(device)
    las_model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/las_model_save_end.pth"))
    #las_model = nn.DataParallel(las_model).to(device)    

    #-------------------------- Data load --------------------------   
    #val dataset
    val_dataset = SpectrogramDataset(audio_conf, 
                                     config.data.val_path, 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                 shuffle=True,
                                 num_workers=config.data.num_workers,
                                 batch_size=2,
                                 drop_last=True)

    print(" ")
    print("las_inference를 진행합니다.")
    print(" ")

    cer = las_inference(las_model, val_loader, device)
    
    print("cer")
    print(cer)

if __name__ == '__main__':
    main()