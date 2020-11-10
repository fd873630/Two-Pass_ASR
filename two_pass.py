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

def main():
    
    yaml_name = "/home/jhjeong/jiho_deep/two_pass/label,csv/Two_Pass.yaml"

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

    #-------------------------- Model Initialize --------------------------
    #Prediction Network
    enc = BaseEncoder(input_size=config.model.enc.input_size,
                      hidden_size=config.model.enc.hidden_size, 
                      output_size=config.model.enc.output_size,
                      n_layers=config.model.enc.n_layers, 
                      dropout=config.model.dropout, 
                      bidirectional=config.model.enc.bidirectional)
    
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

    #Las decoder
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

    #Transducer
    rnnt_model = Transducer(encoder=enc, 
                            decoder=rnnt_dec, 
                            joint=joint,
                            enc_hidden=config.model.enc.hidden_size,
                            enc_projection=config.model.enc.output_size) 
    
    #LAS
    las_model = ListenAttendSpell(enc, las_dec)


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
    rnnt_model = nn.DataParallel(rnnt_model).to(device)
    las_model = nn.DataParallel(las_model).to(device)

    rnnt_optimizer = optim.AdamW(rnnt_model.module.parameters(), 
                                 lr=config.optim.lr, 
                                 weight_decay=config.optim.weight_decay)

    las_optimizer = optim.AdamW(las_model.module.parameters(), 
                                lr=config.optim.lr, 
                                weight_decay=config.optim.weight_decay)

    rnnt_scheduler = optim.lr_scheduler.MultiStepLR(rnnt_optimizer, 
                                                    milestones=config.optim.milestones, 
                                                    gamma=config.optim.decay_rate)
    
    las_scheduler = optim.lr_scheduler.MultiStepLR(las_optimizer, 
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

    print("시작합니다.")
    pre_val_loss = 100000
    
    if config.training.multi_step == "first":
        print('{} first step RNN-T 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()

        for epoch in range(config.training.begin_epoch, config.training.end_epoch):
            train_loss = train(model, train_loader, optimizer, criterion, device)
    
    
    
    
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        
        pass

    '''
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_total_time = time.time()-train_time
        print('{} Epoch {} (Training) Loss {:.4f}, time: {:.4f}'.format(datetime.datetime.now(), epoch+1, train_loss, train_total_time))

        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        val_loss = eval(model, val_loader, criterion, device)
        eval_total_time = time.time()-eval_time
        print('{} Epoch {} (val) Loss {:.4f}, time: {:.4f}'.format(datetime.datetime.now(), epoch+1, val_loss, eval_total_time))
        
        scheduler.step()

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
    '''






        
if __name__ == '__main__':
    main()