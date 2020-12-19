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

def las_train(model, train_loader, optimizer, criterion, device, tf):
    model.train()

    eos_id = 53
    total_loss = 0
    total_num = 0

    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, targets, inputs_lengths, targets_lengths = data

        total_num += sum(targets_lengths)

        inputs_lengths = torch.IntTensor(inputs_lengths)
        targets_lengths = torch.IntTensor(targets_lengths)

        inputs = inputs.to(device) # (batch_size, time, freq)
        targets = targets.to(device)
        inputs_lengths = inputs_lengths.to(device)
        targets_lengths = targets_lengths.to(device)

        logits, _, _ = model(inputs, inputs_lengths, targets, tf, False)
        logits = torch.stack(logits, dim=1).to(device)

        hypothesis = logits.max(-1)[1]
        #hypothesis = hypothesis.squeeze()
                
        _, cer, _, cer_len = compute_cer(hypothesis.cpu().numpy(),targets.cpu().numpy()) 

        total_cer += cer
        total_cer_len += cer_len

        #loss = criterion(logits, targets, 0.05)
        loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('{} train_batch: {:4d}/{:4d}, train_loss: {:.4f}, train_cer: {:.2f} train_time: {:.2f}'
                  .format(datetime.datetime.now(), i, total_batch_num, loss.item(), (cer/cer_len)*100, time.time() - start_time))
            start_time = time.time()

    final_cer = (total_cer / total_cer_len) * 100

    train_loss = total_loss / total_batch_num

    return train_loss, final_cer

def las_eval(model, val_loader, criterion, device):
    model.eval()

    eos_id = 53
    total_loss = 0
    total_num = 0

    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

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

            total_cer += cer
            total_cer_len += cer_len
            
            loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))

            total_loss += loss.item()
    
        val_loss = total_loss / total_batch_num

    final_cer = (total_cer / total_cer_len) * 100

    return val_loss, final_cer

def main():
    
    yaml_name = "/home/jhjeong/jiho_deep/two_pass/label,csv/Two_Pass.yaml"

    with open("./second_step_train.txt", "w") as f:
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
    
    #학습된 enc 불러오기
    enc.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/plz_load/enc.pth"))
    
    #enc 고정 시키기
    for param in enc.parameters():
        param.requires_grad = False
  
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
    
    for param in las_dec.parameters():
        param.data.uniform_(-0.08, 0.08) 
    
    las_dec.load_state_dict(torch.load("./plz_load/second_las_dec_end.pth"))
    las_model = ListenAttendSpell(enc, las_dec, "False").to(device)
    las_model = nn.DataParallel(las_model).to(device)

    #-------------------------- Loss Initialize --------------------------

    las_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    #las_criterion = LabelSmoothingLoss(vocab_size=config.model.vocab_size, ignore_index=0, smoothing=0.1)
    
    #-------------------- Model Pararllel & Optimizer --------------------
    '''
    las_optimizer = optim.AdamW(las_model.module.parameters(), 
                                 lr=config.optim.lr, 
                                 weight_decay=config.optim.weight_decay)
    '''
    las_optimizer = optim.Adam(las_model.module.parameters(), 
                                lr=config.optim.lr)
    
    las_scheduler = optim.lr_scheduler.MultiStepLR(las_optimizer, 
                                                    milestones=config.optim.las_milestones, 
                                                    gamma=config.optim.decay_rate)
    
    #-------------------------- Data load --------------------------
    #train dataset
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
    
    #val dataset
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

    print(" ")
    print("second step las를 학습합니다.")
    print(" ")

    pre_test_cer = 100000
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        for param_group in las_optimizer.param_groups:
            print("lr = ", param_group['lr'])

        if epoch < 30:
               tf = 0.8
        elif epoch < 35:
            tf = 0.75
        elif epoch < 40:
            tf = 0.7
        elif epoch < 45:
            tf = 0.65
        else:
            tf = 0.6
        
        print("tf = ", tf)
        print('{} 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()
        train_loss, train_cer = las_train(las_model, train_loader, las_optimizer, las_criterion, device, tf)
        train_total_time = time.time() - train_time
        print('{} Epoch {} (Training) Loss {:.4f}, CER {:.2f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, train_loss, train_cer, train_total_time))
        
        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        val_loss, test_cer = las_eval(las_model, val_loader, las_criterion, device)
        eval_total_time = time.time() - eval_time
        print('{} Epoch {} (val) Loss {:.4f}, CER {:.2f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, val_loss, test_cer, eval_total_time))
        
        #las_scheduler.step()
        
        with open("./second_step_train.txt", "a") as ff:
            ff.write('tf = %0.4f' % (tf))
            ff.write('\n')
            ff.write('Epoch %d (Training) Loss %0.4f CER %0.4f time %0.4f' % (epoch+1, train_loss, train_cer, train_total_time))
            ff.write('\n')
            ff.write('Epoch %d (val) Loss %0.4f CER %0.4f time %0.4f ' % (epoch+1, val_loss, test_cer, eval_total_time))
            ff.write('\n')
            ff.write('\n')
        
        if pre_test_cer > test_cer:
            print("best model을 저장하였습니다.")
            #torch.save(las_model.module.state_dict(), "./model_save/second_las_train_model_save.pth")
            torch.save(las_dec.state_dict(), "./plz_load/second_las_dec_best.pth")
            pre_test_cer = test_cer

        if epoch+1 == 30:
            torch.save(las_dec.state_dict(), "./plz_load/second_las_dec_epoch_30.pth")
        
        elif epoch+1 == 40:
            torch.save(las_dec.state_dict(), "./plz_load/second_las_dec_epoch_40.pth")
        
        elif epoch+1 == 50:
            torch.save(las_dec.state_dict(), "./plz_load/second_las_dec_epoch_50.pth")
        
        elif epoch+1 == 60:
            torch.save(las_dec.state_dict(), "./plz_load/second_las_dec_epoch_60.pth")
        
        elif epoch+1 == 70:
            torch.save(las_dec.state_dict(), "./plz_load/second_las_dec_epoch_70.pth")
        
        else:
            pass

        torch.save(las_dec.state_dict(), "./plz_load/second_las_dec_end.pth")

if __name__ == '__main__':
    main()