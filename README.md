# 아직 제작중입니다!


# Two-Pass End-to-End Speech Recognition

|Strategy|Feature|Dataset|CER|  
|--------|-----|-------|------| 
|B0|80-dimensional log-Mel features|KsponSpeech_val(길이 조절 데이터)|17.32|
|B1|80-dimensional log-Mel features|KsponSpeech_val(길이 조절 데이터)|?|
|E0|80-dimensional log-Mel features|KsponSpeech_val(길이 조절 데이터)|27.98|


* B0 : RNN-T only(beam_mode)
* B1 : LAS only
* E0 : 2nd Beam Search
* E1 : Rescoring

## Intro
한국어를 위한 Two-Pass End-to-End Speech Recognition입니다. 실시간 인식에는 attention기반의 모델보다 RNN-Transducer가 사용된다고 합니다. 하지만 기존의 Listen attend and Spell 보다 성능이 좋지 못합니다. 이 논문에서는 RNN-T와 LAS를 합쳐 성능과 실시간 인식률을 두마리의 토끼를 잡자는 컨셉의 논문입니다. 현재 git hub에는 Pytorch Two pass 코드와 성능결과가 없어 한국어 Two-pass를 구현하고 성능을 확인하였습니다.

## Version
* torch version = 1.2.0
* Cuda compilation tools, release 9.1, V9.1.85
* nn.DataParallel를 통해 multi GPU 학습

## How to install RNN-T Loss
* https://github.com/HawkAaron/warp-transducer/tree/master/pytorch_binding

## Data
### Dataset information
AI hub에서 제공하는 '한국어 음성데이터'를 사용하였습니다. AI Hub 음성 데이터는 다음 링크에서 신청 후 다운로드 하실 수 있습니다.

AI Hub 한국어 음성 데이터 : http://www.aihub.or.kr/aidata/105 

### Data format
* 음성 데이터 : 16bit, mono 16k sampling WAV audio
* 정답 스크립트 : 제공된 스크립트를 자소로 변환된 정답 -> 공백(띄어쓰기) 제거
  ```js
  1. "b/ (70%)/(칠 십 퍼센트) 확률이라니 " => "칠십퍼센트확률이라니" 
  
  2. "칠십퍼센트확률이라니" => "ㅊㅣㄹㅅㅣㅂㅍㅓㅅㅔㄴㅌㅡㅎㅘㄱㄹㅠㄹㅇㅣㄹㅏㄴㅣ"

  3. "ㅊㅣㄹㅅㅣㅂㅍㅓㅅㅔㄴㅌㅡㅎㅘㄱㄹㅠㄹㅇㅣㄹㅏㄴㅣ" => "16 41 7 1 11 41 9 1 19 25 11 26 4 18 39 ..."
  
  최종:  "b/(70%)/(칠십퍼센트)확률이라니 " => "16 41 7 1 11 41 9 1 19 25 11 26 4 18 39 ..."
  ```

1. 위의 txt 전처리는 https://github.com/sooftware/KoSpeech/wiki/Preparation-before-Training 다음을 참고하였습니다.

2. ./model_rnnt/hangul.py 에 있는 pureosseugi 함수를 통해 자소 분리를 하였습니다.

3. ./label,csv/hangul.labels 를 기반으로 대응하는 숫자로 변환하였습니다.

### Dataset folder structure
* DATASET-ROOT-FOLDER
```
|--DATA
   |--train
      |--wav
         +--a.wav, b.wav, c.wav ...
      |--txt
         +--a.txt, b.txt, c.txt ...
   |--val
      |--wav
         +--a_val.wav, b_val.wav, c_val.wav ...
      |--txt
         +--a_val.txt, b_val.txt, c_val.txt ...
```
* data_list.csv
  ```
  <wav-path>,<script-path>
  KsponSpeech_000001.wav,KsponSpeech_000001.txt
  KsponSpeech_000002.wav,KsponSpeech_000002.txt
  KsponSpeech_000003.wav,KsponSpeech_000003.txt
  KsponSpeech_000004.wav,KsponSpeech_000004.txt
  KsponSpeech_000005.wav,KsponSpeech_000005.txt
  ...
  ```

데이터를 커스텀하여 사용하고 싶으신분들은 다음과 같은 형식으로 .csv 파일을 제작하면 됩니다.

* hangul.labels
  ```
  #id\char 
  0   _
  1   ㄱ
  ...
  51   ㅄ
  52   <s>
  53   </s>
  ```

## Model
### Feature
* 80-dimensional log-Mel features

  parameter | value
  ------|-----
  N_FFT | sample_rate * window_size
  window_size | 20ms
  window_stride | 10ms
  window function | hamming window

* code
  ```python
  def parse_audio(self, audio_path):
    y,sr = librosa.load(audio_path, self.sample_rate)
        
    n_fft = int(self.sample_rate * self.window_size)
    win_length = n_fft
    hop_length = int(self.sample_rate * self.window_stride)

    #log mel feature
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=80)
    logmelspec = librosa.power_to_db(melspec)

    ```
### Architecture
<img width = "400" src = "https://user-images.githubusercontent.com/43025347/98681590-3d9a5580-23a6-11eb-9e95-96c5d9b79081.png">

### Print Model
```python

```

## References
### Git hub References
* https://github.com/1ytic/warp-rnnt
* https://github.com/ZhengkunTian/rnn-transducer
* https://github.com/HawkAaron/E2E-ASR
* https://github.com/sooftware

### Paper References
* Two-Pass End-to-End Speech Recognition (https://arxiv.org/abs/1908.10992)
* Sequence Transduction With Recurrent Neural Networks (https://arxiv.org/abs/1211.3711)
* Speech Recognition With Deep Recurrent Neural Networks (https://arxiv.org/abs/1303.5778)
* State-of-the-art Speech Recognition With Sequence-to-Sequence Models (https://arxiv.org/abs/1712.01769)
* Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models (https://arxiv.org/abs/1712.01818)


### Blog References
* https://gigglehd.com/zbxe/14052329
* https://dos-tacos.github.io/paper%20review/sequence-transduction-with-rnn/

### Youtube References
* https://www.youtube.com/watch?v=W7b77hv3Rak&ab_channel=KrishnaDN

## computer power
* NVIDIA TITAN Xp * 4

## Q & A
Q1 : (Data set part) KsponSpeech_val(길이 조절 데이터)은 왜 따로 나눴는지?

A1 : RNN-T는 RNN-T Loss를 사용합니다. 그러므로 wav len과 script len에 따라서 시간과 메모리를 잡아 먹습니다. KsponSpeech_eval_clean의 데이터를 wav len과 script len은 특정 길이로 제한하게 되면 데이터의 양이 너무 적어 학습 데이터에서 5시간을 분리했습니다.

* train data 총 길이 - 약 254시간 
* val data 총 길이 - 약 5시간 
* KsponSpeech_eval_clean(AI_hub eval 데이터) - 약 2.6시간

Q2 : (labels part) 왜 음절 단위 말고 자소 단위로 나눴는지?

A2 : RNN-T Loss wav len과 script len뿐만 아니라 vocab size도 메모리를 잡아 먹습니다.즉 vocab size가 증가 할 수록 메모리를 많이 잡아 먹기 때문에 학습에서 gpu 메모리 이득을 보기 위해 다음과 같이 사용하였습니다. (gpu 메모리가 여유가 있으시면 음절 단위로 해보셔도 좋을것 같습니다.)


## Contacts
학부생의 귀여운 시도로 봐주시고 해당 작업에 대한 피드백, 문의사항 모두 환영합니다.

fd873630@naver.com로 메일주시면 최대한 빨리 답장드리겠습니다.

인하대학교 전자공학과 4학년 정지호




