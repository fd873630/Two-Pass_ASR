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
from model_rnnt.model import Transducer
from model_rnnt.encoder import BaseEncoder
from model_rnnt.decoder import BaseDecoder
from model_rnnt.las_decoder import Speller
from model_rnnt.las import ListenAttendSpell
from model_rnnt.hangul import moasseugi
from model_rnnt.data_loader_deepspeech import SpectrogramDataset, AudioDataLoader, AttrDict

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
    
    total_wer = 0
    total_cer = 0

    total_wer_len = 0
    total_cer_len = 0

    for label, pred in zip(labels, preds):
        units = []
        units_pred = []
        for a in label:
            units.append(index2char[a])
            
        for b in pred:
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

def train(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        inputs, targets, inputs_lengths, targets_lengths = data

        inputs_lengths = torch.IntTensor(inputs_lengths)
        targets_lengths = torch.IntTensor(targets_lengths)

        inputs = inputs.to(device) # (batch_size, time, freq)
        targets = targets.to(device)
        inputs_lengths = inputs_lengths.to(device)
        targets_lengths = targets_lengths.to(device)

        logits = model(inputs, inputs_lengths, targets, 1, False)
        logits = torch.stack(logits, dim=1).to(device)
        
        hypothesis = logits.max(-1)[1]

        wer, cer, wer_len, cer_len = compute_cer(hypothesis.cpu().numpy(),targets.cpu().numpy())
        
        total_wer += wer
        total_cer += cer
                
        total_wer_len += wer_len
        total_cer_len += cer_len
        
        loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))
        
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        max_inputs_length = inputs_lengths.max().item()
        max_targets_length = targets_lengths.max().item()
        
        total_loss += loss.item()

        if i % 1000 == 0:
            print('{} train_batch: {:4d}/{:4d}, train_loss: {:.4f}, train_cer: {:.2f} train_time: {:.2f}'
                  .format(datetime.datetime.now(), i, total_batch_num, loss.item(), (cer/cer_len)*100, time.time() - start_time))
            start_time = time.time()
            
    final_wer = (total_wer / total_wer_len) * 100
    final_cer = (total_cer / total_cer_len) * 100

    train_loss = total_loss / total_batch_num

    return train_loss, final_cer, final_wer

def eval(model, val_loader, criterion, device):
    model.eval()
    
    total_loss = 0
    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

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

            loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))
            
            total_loss += loss.item()

        final_wer = (total_wer / total_wer_len) * 100
        final_cer = (total_cer / total_cer_len) * 100
       
        val_loss = total_loss / total_batch_num

    return val_loss, final_cer, final_wer

def main():
    yaml_name = "/home/jhjeong/jiho_deep/las/label,csv/LAS.yaml"
    
    with open("./train.txt", "w") as f:
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
                      sos_id=53, 
                      eos_id=54, 
                      attention_head=config.model.las_dec.attention_head, 
                      num_layers=config.model.las_dec.n_layers,
                      projection_dim=config.model.las_dec.projection_dim, 
                      dropout_p=config.model.dropout, 
                      device=device)
    '''
        type: lstm
        max_length: 240
        attention_head: 4
        hidden_size: 2048
        n_layers: 2
        embedding_size: 96
        projection_dim: 640
    '''
    model = ListenAttendSpell(enc, las_dec)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.AdamW(model.module.parameters(), 
                            lr=config.optim.lr, 
                            weight_decay=config.optim.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones=config.optim.milestones, 
                                               gamma=config.optim.decay_rate)


    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)


    train_dataset = SpectrogramDataset(audio_conf, 
                                       config.data.train_path,
                                       feature_type=config.audio_data.type, 
                                       normalize=True, 
                                       spec_augment=True)

    train_loader = AudioDataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=config.data.batch_size,
                                    drop_last=True)

    val_dataset = SpectrogramDataset(audio_conf, 
                                     config.data.val_path, 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=config.data.batch_size,
                                    drop_last=True)

    #print(model)
    print("시작합니다.")

    pre_val_loss = 100000
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        print('{} 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()
        train_loss, train_cer, train_wer = train(model, train_loader, optimizer, criterion, device)
        train_total_time = time.time()-train_time
        print('{} Epoch {} (Training) Loss {:.4f} CER {:.2f} time: {:.4f}'.format(datetime.datetime.now(), epoch+1, train_loss, train_cer, train_total_time))
        
        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        val_loss, val_cer, val_wer = eval(model, val_loader, criterion, device)
        eval_total_time = time.time()-eval_time
        print('{} Epoch {} (val) Loss {:.4f}, time: {:.4f}'.format(datetime.datetime.now(), epoch+1, val_loss, val_cer, eval_total_time))
        
        with open("./train.txt", "a") as ff:
            ff.write('Epoch %d (Training) Loss %0.4f time %0.4f' % (epoch+1, train_loss, train_total_time))
            ff.write('\n')
            ff.write('Epoch %d (val) Loss %0.4f time %0.4f ' % (epoch+1, val_loss, eval_total_time))
            ff.write('\n')
            ff.write('\n')
        
        if pre_val_loss > val_loss:
            print("best model을 저장하였습니다.")
            torch.save(model.module.state_dict(), "./model_save/model_save.pth")
            pre_val_loss = val_loss
        
if __name__ == '__main__':
    main()
