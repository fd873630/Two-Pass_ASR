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

def rnnt_train(model, train_loader, optimizer, criterion, device):
    model.train()

    eos_id = 53
    total_loss = 0
    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        inputs, targets, inputs_lengths, targets_lengths = data

        targets = targets[targets != eos_id].view(targets.shape[0], -1)
        
        inputs_lengths = torch.IntTensor(inputs_lengths)
        targets_lengths = torch.IntTensor(targets_lengths)

        inputs = inputs.to(device) # (batch_size, time, freq)
        targets = targets.to(device)
        inputs_lengths = inputs_lengths.to(device)
        targets_lengths = targets_lengths.to(device)

        logits = model(inputs, inputs_lengths, targets, targets_lengths)
        loss = criterion(logits, targets.int(), inputs_lengths.int(), targets_lengths.int())

        max_inputs_length = inputs_lengths.max().item()
        max_targets_length = targets_lengths.max().item()
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('{} train_batch: {:4d}/{:4d}, train_loss: {:.2f}, train_time: {:.2f}'.format(datetime.datetime.now(), i, total_batch_num, loss.item(), time.time() - start_time))
            start_time = time.time()

    train_loss = total_loss / total_batch_num

    return train_loss

def rnnt_eval(model, val_loader, criterion, device):
    model.eval()

    eos_id = 53
    total_loss = 0
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets, inputs_lengths, targets_lengths = data

            targets = targets[targets != eos_id].view(targets.shape[0], -1)

            inputs_lengths = torch.IntTensor(inputs_lengths)
            targets_lengths = torch.IntTensor(targets_lengths)

            inputs = inputs.to(device) # (batch_size, time, freq)
            targets = targets.to(device)
            inputs_lengths = inputs_lengths.to(device)
            targets_lengths = targets_lengths.to(device)
                
            logits = model(inputs, inputs_lengths, targets, targets_lengths)
            loss = criterion(logits, targets.int(), inputs_lengths.int(), targets_lengths.int())
    
            total_loss += loss.item()
    
        val_loss = total_loss / total_batch_num

    return val_loss

def main():
    
    yaml_name = "/home/jhjeong/jiho_deep/two_pass/label,csv/Two_Pass.yaml"

    with open("./first_step_train.txt", "w") as f:
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
    
    for param in enc.parameters():
        param.data.uniform_(-0.08, 0.08)
    #enc.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/first_train_enc_save.pth"))

    #Transcription Network
    rnnt_dec = BaseDecoder(embedding_size=config.model.rnn_t_dec.embedding_size,
                           hidden_size=config.model.rnn_t_dec.hidden_size, 
                           vocab_size=config.model.vocab_size, 
                           output_size=config.model.rnn_t_dec.output_size, 
                           n_layers=config.model.rnn_t_dec.n_layers, 
                           dropout=config.model.dropout)
   
    for param in rnnt_dec.parameters():
        param.data.uniform_(-0.08, 0.08)

    #Joint Network
    joint = JointNet(input_size=config.model.enc.output_size, 
                     inner_dim=config.model.joint.inner_dim, 
                     vocab_size=config.model.vocab_size)
    
    for param in joint.parameters():
        param.data.uniform_(-0.08, 0.08)

    #Transducer
    rnnt_model = Transducer(encoder=enc, 
                            decoder=rnnt_dec, 
                            joint=joint,
                            enc_hidden=config.model.enc.hidden_size,
                            enc_projection=config.model.enc.output_size) 
    
    for param in rnnt_model.parameters():
        param.data.uniform_(-0.08, 0.08)

    #rnnt_model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/first_train_model_save.pth"))

    #-------------------------- Loss Initialize --------------------------
    rnnt_criterion = RNNTLoss().to(device)
    """
        acts: Tensor of [batch x seqLength x (labelLength + 1) x outputDim] containing output from network
        (+1 means first blank label prediction)
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
    """

    #-------------------- Model Pararllel & Optimizer --------------------
    rnnt_model = nn.DataParallel(rnnt_model).to(device)

    rnnt_optimizer = optim.AdamW(rnnt_model.module.parameters(), 
                                 lr=config.optim.lr, 
                                 weight_decay=config.optim.weight_decay)

    rnnt_scheduler = optim.lr_scheduler.MultiStepLR(rnnt_optimizer, 
                                                    milestones=config.optim.milestones, 
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
    print("first step RNN-T를 학습합니다.")
    print(" ")

    pre_val_loss = 100000
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        print('{} 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()
        train_loss = rnnt_train(rnnt_model, train_loader, rnnt_optimizer, rnnt_criterion, device)
        train_total_time = time.time()-train_time
        print('{} Epoch {} (Training) Loss {:.4f}, time: {:.4f}'.format(datetime.datetime.now(), epoch+1, train_loss, train_total_time))
        
        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        val_loss = rnnt_eval(rnnt_model, val_loader, rnnt_criterion, device)
        eval_total_time = time.time()-eval_time
        print('{} Epoch {} (val) Loss {:.4f}, time: {:.4f}'.format(datetime.datetime.now(), epoch+1, val_loss, eval_total_time))

        rnnt_scheduler.step()
        
        with open("./first_step_train.txt", "a") as ff:
            ff.write('Epoch %d (Training) Loss %0.4f time %0.4f' % (epoch+1, train_loss, train_total_time))
            ff.write('\n')
            ff.write('Epoch %d (val) Loss %0.4f time %0.4f ' % (epoch+1, val_loss, eval_total_time))
            ff.write('\n')
            ff.write('\n')
        
        if pre_val_loss > val_loss:
            print("best model을 저장하였습니다.")
            torch.save(rnnt_model.module.state_dict(), "./model_save/first_train_model_save_no_blank.pth")
            torch.save(enc.state_dict(), "./model_save/first_train_enc_save_no_blank.pth")
            pre_val_loss = val_loss

        torch.save(rnnt_model.module.state_dict(), "./model_save/first_train_model_save_no_blank_end.pth")
        torch.save(enc.state_dict(), "./model_save/first_train_enc_save_no_blank_end.pth")
if __name__ == '__main__':
    main()