import math
import os
import time
from matplotlib import pyplot as plt
import pandas as pd
import librosa.display, librosa
import numpy as np
import scipy.signal
import soundfile as sf
import sox
import torch
import csv
from .spec_augment import spec_augment
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
import matplotlib

windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann,
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

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

char2index, index2char = load_label('./label,csv/AI_hub_label.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value

class AudioParser(object):
    def parse_transcript(self, transcript_path):
        raise NotImplementedError

    def parse_audio(self, audio_path):
        raise NotImplementedError

class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, feature_type, normalize, spec_augment):
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.feature_type = feature_type
        self.spec_augment = spec_augment

    def parse_audio(self, audio_path):
        
        y,sr = librosa.load(audio_path, self.sample_rate)
        
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)

        #log mel feature
        melspec = librosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=80)
        logmelspec = librosa.power_to_db(melspec)

        spect = torch.FloatTensor(logmelspec)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        
        if self.spec_augment:
            spect = spec_augment(spect)

        if False:
            path = './test_img'
            os.makedirs(path, exist_ok=True)
            matplotlib.image.imsave('./test_img/'+ audio_path[50:-4] +'name.png', spect)
        

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, feature_type, normalize, spec_augment):
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        
        self.ids = ids
        self.size = len(ids)
               
        super(SpectrogramDataset, self).__init__(audio_conf, feature_type, normalize, spec_augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        
        transcript = self.parse_transcript(transcript_path)
        spect = self.parse_audio(audio_path)
        
        spect = torch.transpose(spect, 0, 1)
        
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as f:
            transcript = f.read()
            transcript = transcript.strip().split(' ')
            transcript = list(map(int, transcript))
            transcript.append(EOS_token)           

        return transcript

    def __len__(self):
        return self.size

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])
    
    def target_length_(p):
        return len(p[1])
  
    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1])-1 for s in batch] #eos lenth제거를 위해 -1 (rnn-t)

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD_token)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

#torch.Size([16, 1192, 240])
#torch.Size([16, 31])
#[919, 474, 649, 365, 544, 225, 407, 590, 627, 468, 473, 450, 304, 436, 406, 1192]
#[24, 16, 15, 12, 31, 10, 18, 16, 21, 17, 15, 20, 11, 14, 13, 17]

