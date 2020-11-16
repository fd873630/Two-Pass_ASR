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
from warprnnt_pytorch import RNNTLoss

from model_rnnt.eval_distance import eval_wer, eval_cer
from model_rnnt.model import Transducer, JointNet
from model_rnnt.encoder import BaseEncoder
from model_rnnt.decoder import BaseDecoder
from model_rnnt.las_decoder import Speller
from model_rnnt.las import ListenAttendSpell
from model_rnnt.hangul import moasseugi
from model_rnnt.data_loader_deepspeech import SpectrogramDataset, AudioDataLoader, AttrDict

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
            if a == 54: # eos
                break
            units.append(index2char[a])
            
        for b in pred:
            if b == 54: # eos
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

def las_inference(model, val_loader, criterion, device):
    model.eval()

    eos_id = 54
    total_loss = 0

    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

    with open("./second_inference.txt", "w") as f:
        f.write('\n')
        f.write("inference 시작")
        f.write('\n')
    
    char2index, index2char = load_label('./label,csv/hangul.labels')

    start_time = time.time()
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets, inputs_lengths, targets_lengths = data
        
            inputs_lengths = torch.IntTensor(inputs_lengths)
            targets_lengths = torch.IntTensor(targets_lengths)

            inputs = inputs.to(device) # (batch_size, time, freq)
            targets = targets.to(device)
            inputs_lengths = inputs_lengths.to(device)
            targets_lengths = targets_lengths.to(device)

            logits = model(inputs, inputs_lengths, targets, 0, False)
            logits = torch.stack(logits, dim=1).to(device)

            hypothesis = logits.max(-1)[1]

            hypothesis = hypothesis.squeeze()
            wer, cer, wer_len, cer_len = compute_cer(hypothesis.cpu().numpy(),targets.cpu().numpy())        

            total_wer += wer
            total_cer += cer
                    
            total_wer_len += wer_len
            total_cer_len += cer_len

            for a, b in zip(targets.cpu(), hypothesis.cpu()):
                chars = []
                predic_chars = []
                
                for w in a:
                    if w.item() == 54: # eos
                        break
                    chars.append(index2char[w.item()])

                for y in b:
                    if y.item() == 54: # eos
                        break
                    predic_chars.append(index2char[y.item()])
                
                with open("./second_inference.txt", "a") as f:
                    f.write('\n')
                    f.write(moasseugi(chars))
                    f.write('\n')
                    f.write(moasseugi(predic_chars))
                    f.write('\n')
                
    val_loss = total_loss / total_batch_num

    final_wer = (total_wer / total_wer_len) * 100
    final_cer = (total_cer / total_cer_len) * 100

    return val_loss, final_cer, final_wer

def main():
    
    yaml_name = "/home/jhjeong/jiho_deep/two_pass/label,csv/Two_Pass.yaml"

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
    
    #학습된 enc 불러오기
    enc.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/first_train_enc_save.pth"))
  
    #Transcription Network
    rnnt_dec = BaseDecoder(embedding_size=config.model.rnn_t_dec.embedding_size,
                           hidden_size=config.model.rnn_t_dec.hidden_size, 
                           vocab_size=config.model.vocab_size, 
                           output_size=config.model.rnn_t_dec.output_size, 
                           n_layers=config.model.rnn_t_dec.n_layers, 
                           dropout=config.model.dropout)

    #Joint Network
    joint = JointNet(input_size=config.model.enc.output_size, 
                     inner_dim=config.model.joint.inner_dim, 
                     vocab_size=config.model.vocab_size)

    #Transducer
    rnnt_model = Transducer(encoder=enc, 
                            decoder=rnnt_dec, 
                            joint=joint,
                            enc_hidden=config.model.enc.hidden_size,
                            enc_projection=config.model.enc.output_size) 

    las_dec = Speller(vocab_size=config.model.vocab_size,
                      embedding_size=config.model.las_dec.embedding_size,
                      max_length=config.model.las_dec.max_length, 
                      hidden_dim=config.model.las_dec.hidden_size, 
                      pad_id=0, 
                      sos_id=53, 
                      eos_id=54, 
                      attention_head=config.model.las_dec.attention_head, 
                      num_layers=config.model.las_dec.n_layers,
                      projection_dim=config.model.las_dec.projection_dim, 
                      dropout_p=config.model.dropout, 
                      device=device)

    las_dec.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/second_las_dec_save_tf_1.pth"))

    las_model = ListenAttendSpell(enc, las_dec)
    
    las_model = nn.DataParallel(las_model).to(device)
    #-------------------------- Loss Initialize --------------------------
    rnnt_criterion = RNNTLoss().to(device)
    """
        acts: Tensor of [batch x seqLength x (labelLength + 1) x outputDim] containing output from network
        (+1 means first blank label prediction)
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
    """
    las_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    #-------------------- Model Pararllel & Optimizer --------------------
    

    las_optimizer = optim.AdamW(las_model.module.parameters(), 
                                 lr=config.optim.lr, 
                                 weight_decay=config.optim.weight_decay)

    las_scheduler = optim.lr_scheduler.MultiStepLR(las_optimizer, 
                                                    milestones=config.optim.milestones, 
                                                    gamma=config.optim.decay_rate)
    
    #-------------------------- Data load --------------------------
    #val dataset
    val_dataset = SpectrogramDataset(audio_conf, 
                                     config.data.val_path, 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                 shuffle=False,
                                 num_workers=config.data.num_workers,
                                 batch_size=config.data.batch_size,
                                 drop_last=True)

    print(" ")
    print("second step inference를 학습합니다.")
    print(" ")

    print('{} 평가 시작'.format(datetime.datetime.now()))
    eval_time = time.time()
    _, cer, wer = las_inference(las_model, val_loader, las_criterion, device)
    eval_total_time = time.time()-eval_time
    
    print("final_wer -> ")
    print(wer)
    print("final_cer -> ")
    print(cer)



if __name__ == '__main__':
    main()