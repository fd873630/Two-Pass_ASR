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
from model_rnnt.rnn_t_topk_decoder import TopK_RNN_T_Decoder
from model_rnnt.hangul import moasseugi
from model_rnnt.data_loader_deepspeech import SpectrogramDataset, AudioDataLoader, AttrDict

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

        cer = eval_cer(pred, label)
        cer_len = len(label.replace(" ", ""))

        total_cer += cer
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

def inference(model, las_model, val_loader, device, beam_search):
    model.eval()
    las_model.eval()

    total_loss = 0
   
    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

    with open("./first_inference.txt", "w") as f:
        f.write('\n')
        f.write("inference 시작")
        f.write('\n')

    char2index, index2char = load_label('./label,csv/hangul.labels')

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            total_batch_num = len(val_loader)
        
            inputs, targets, inputs_lengths, targets_lengths = data

            inputs_lengths = torch.IntTensor(inputs_lengths)
            targets_lengths = torch.IntTensor(targets_lengths)

            inputs = inputs.to(device) # (batch_size, time, freq)
            targets = targets.to(device)
            inputs_lengths = inputs_lengths.to(device)
            targets_lengths = targets_lengths.to(device)

            transcripts = [targets.cpu().numpy()[i][:targets_lengths[i].item()]
                       for i in range(targets.size(0))]
            
            begin_time = time.time()

            beam_width = 5
            
            preds = model(inputs, inputs_lengths, W=beam_width) # torch.size([batch_size * W, max_len])
            preds = preds.to(device)
                      
            batch_size = inputs.size(0)

            for beam_index in range(batch_size):
                preds_targets = preds[beam_index * beam_width : (beam_index+1) * beam_width, :]
                expend_inputs = inputs[beam_index].repeat(beam_width,1,1)
                              
                logits, _, _ = las_model(expend_inputs, None, preds_targets, 1, False)
                
                logits = torch.stack(logits, dim=1).to(device)

                hypothesis = logits.max(-1)[1]

            predic_chars = []        
            for aa in range(hypothesis.size(0)):    
                for aaa in hypothesis[aa*5].cpu().numpy():
                    if aaa == 53: # eos
                        break
                    predic_chars.append(index2char[aaa])

                print(moasseugi(predic_chars))

            '''
            _, cer, _, cer_len = compute_cer(preds, transcripts)
     
            total_cer += cer
            total_cer_len += cer_len

            for a, b in zip(transcripts,preds):
                chars = []
                predic_chars = []
                
                for w in a:
                    chars.append(index2char[w])

                for y in b:
                    predic_chars.append(index2char[y])
                
                with open("./first_inference.txt", "a") as f:
                    f.write('\n')
                    f.write(moasseugi(chars))
                    f.write('\n')
                    f.write(moasseugi(predic_chars))
                    f.write('\n')
            '''
    final_cer = (total_cer / total_cer_len) * 100  

    return final_cer

def main():
    beam_mode = True
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
    
    enc.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/first_train_enc_save_no_blank.pth"))
    
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

    las_dec.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/second_last_las_dec_save_no_blank_end2.pth"))
    
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

    rnnt_model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/two_pass/model_save/first_train_model_save_no_blank.pth"))

    rnnt_model = TopK_RNN_T_Decoder(rnnt_model, 5, device) # 5 = beam size
    rnnt_model = rnnt_model.to(device)
    
    las_model = ListenAttendSpell(enc, las_dec, "false").to(device)


    #rnnt_model = nn.DataParallel(rnnt_model).to(device)
   
    val_dataset = SpectrogramDataset(audio_conf, 
                                     config.data.val_path, 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                 shuffle=False,
                                 num_workers=config.data.num_workers,
                                 batch_size=5,
                                 drop_last=True)
    
    print(" ")
    print('{} test inference 시작'.format(datetime.datetime.now()))
    print(" ")
    
    final_cer = inference(rnnt_model, las_model, val_loader, device, beam_mode)

    print("final_cer -> ")
    print(final_cer)
    
    print('{} inference 끝'.format(datetime.datetime.now()))
    
if __name__ == '__main__':
    main()
    
