import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, autograd
import math

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size*2, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)
        
    def forward(self, enc_state, dec_state):
        
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)
   
            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)
        
        return outputs


class Transducer(nn.Module):
    def __init__(self, encoder, decoder, joint, enc_hidden):
        super(Transducer, self).__init__()
        # define model
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        
    def forward(self, inputs, inputs_lengths, targets, targets_lengths):
        
        zero = torch.zeros((targets.shape[0], 1)).long()
        if targets.is_cuda: zero = zero.cuda()
        
        targets_add_blank = torch.cat((zero, targets), dim=1)

        enc_state, _ = self.encoder(inputs, inputs_lengths)
        
        dec_state, _ = self.decoder(targets_add_blank, targets_lengths+1)
        
        logits = self.joint(enc_state, dec_state)

        return logits

    #only one
    def recognize(self, inputs, inputs_length):
    
        batch_size = inputs.size(0)

        enc_states, _ = self.encoder(inputs, inputs_length)
        #enc_states = self.project_layer(enc_states)
        zero_token = torch.LongTensor([[0]])

        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decoder(zero_token)
            #print(len(hidden))
            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = []
        
        for i in range(batch_size):
            
            decoded_seq = decode(enc_states[i], inputs_length[i])
            
            results.append(decoded_seq)

        return results

    def beam_search(self, inputs, inputs_length, W): 
        use_gpu = inputs.is_cuda
        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        def forward_step(label, hidden):
            #if use_gpu: label = label.cuda()
            
            label = torch.LongTensor([[label]])
            if use_gpu: label = label.cuda()
            pred, hidden = self.decoder(inputs=label, hidden = hidden)

            return pred[0][0], hidden

        B = [Sequence(blank=0)]

        #---------------------- input 작업 ----------------------------
        batch_size = inputs.size(0)

        #inputs - torch.Size([1, 185, 80])
        #inputs_length - torch.Size([1])
        enc_states, _ = self.encoder(inputs, inputs_length)
        #enc_states = self.project_layer(enc_states)
        enc_states_for_beam = enc_states.squeeze()
        
        prefix = True
        
        for i, x in enumerate(enc_states_for_beam):
            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B
            B = []

            # len(A) 맨처음 1 그 뒤로 5
            if prefix: # <- 맨 처음은 실행 안됨
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        
                        pred, _ = forward_step(A[i].k[-1], A[i].h)

                        idx = len(A[i].k)
                        ytu = self.joint(x, pred)

                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])

                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(x, A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp) # y* = most probable in A
                #a.lop중 가장 큰 값이 y_hat
                               
                A.remove(y_hat) # remove y* from A
                # A에서 가장 큰 확률 제거 ex) 1->0 53->52

                # calculate P(k|y_hat, t)
                # get last label and hidden state
        
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h) # label, hidden state

                ytu = self.joint(x, pred) # x는 input

                logp = F.log_softmax(ytu, dim=0) # log probability for each k -> torch.Size[54]
                
                # 여기부터 !!
                for k in range(len(logp)):
                    yk = Sequence(y_hat)

                    yk.logp += float(logp[k])

                    if k == 0:
                        B.append(yk)                        
                        continue
                    
                    # store prediction distribution and last hidden state
                    # yk.h.append(hidden); yk.k.append(k)
                    yk.h = hidden; yk.k.append(k); 
                    
                    if prefix: 
                        yk.g.append(pred)
                    
                    A.append(yk)

                y_hat = max(A, key=lambda a: a.logp)   

                yb = max(B, key=lambda a: a.logp)

                if len(B) >= W and yb.logp >= y_hat.logp: 
                    break
                
            sorted(B, key=lambda a: a.logp, reverse=True)
            
        return B[0].k, -B[0].logp

    def MCER_beam_search(self, inputs, inputs_length, W): 
        use_gpu = inputs.is_cuda
        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        def forward_step(label, hidden):
            #if use_gpu: label = label.cuda()
            
            label = torch.LongTensor([[label]])
            if use_gpu: label = label.cuda()
            pred, hidden = self.decoder(inputs=label, hidden = hidden)

            return pred[0][0], hidden
        
        B = [Sequence(blank=0)]
        
        #---------------------- input 작업 ----------------------------
        batch_size = inputs.size(0)
        enc_states, _ = self.encoder(inputs, inputs_length)
        
        enc_states = self.project_layer(enc_states)
        enc_states_for_beam = enc_states.squeeze()
        
        prefix = True
        
        for i, x in enumerate(enc_states_for_beam):
            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B
            B = []

            if prefix:
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        
                        pred, _ = forward_step(A[i].k[-1], A[i].h)

                        idx = len(A[i].k)
                        ytu = self.joint(x, pred)

                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])

                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(x, A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp) # y* = most probable in A
                A.remove(y_hat) # remove y* from A

                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)               
                ytu = self.joint(x, pred) # x는 input

                logp = F.log_softmax(ytu, dim=0) # log probability for each k
                for k in range(len(logp)):
                    yk = Sequence(y_hat)

                    yk.logp += float(logp[k])
                    
                    if k == 0:
                        B.append(yk)                        
                        continue
                    
                    # store prediction distribution and last hidden state
                    # yk.h.append(hidden); yk.k.append(k)
                    yk.h = hidden; yk.k.append(k); 
                    
                    if prefix: yk.g.append(pred)
                    
                    A.append(yk)

                y_hat = max(A, key=lambda a: a.logp)   

                yb = max(B, key=lambda a: a.logp)

                if len(B) >= W and yb.logp >= y_hat.logp: break
                
            sorted(B, key=lambda a: a.logp, reverse=True)

        return B # B[0].k

    def new_beam_search(self, enc_states_for_beam, W, device): 
        use_gpu = enc_states_for_beam.is_cuda
        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        def forward_step(label, hidden):
            #if use_gpu: label = label.cuda()
            
            label = torch.LongTensor([[label]])
            if use_gpu: label = label.cuda()

            #label = label.to(device)
            pred, hidden = self.decoder(inputs=label, hidden=hidden)

            return pred[0][0], hidden

        B = [Sequence(blank=0)]

        #---------------------- input 작업 ----------------------------
        
        prefix = True
        
        for i, x in enumerate(enc_states_for_beam):
            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B
            B = []

            if prefix:
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        
                        pred, _ = forward_step(A[i].k[-1], A[i].h)

                        idx = len(A[i].k)
                        ytu = self.joint(x, pred)

                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])

                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(x, A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp) # y* = most probable in A
                A.remove(y_hat) # remove y* from A

                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)               
                ytu = self.joint(x, pred) # x는 input

                logp = F.log_softmax(ytu, dim=0) # log probability for each k
                
                for k in range(len(logp)):
                    yk = Sequence(y_hat)

                    yk.logp += float(logp[k])

                    if k == 0:
                        B.append(yk)                        
                        continue
                    
                    # store prediction distribution and last hidden state
                    # yk.h.append(hidden); yk.k.append(k)
                    yk.h = hidden; yk.k.append(k); 
                    
                    if prefix: yk.g.append(pred)
                    
                    A.append(yk)

                y_hat = max(A, key=lambda a: a.logp)   

                yb = max(B, key=lambda a: a.logp)

                if len(B) >= W and yb.logp >= y_hat.logp: break
                
            sorted(B, key=lambda a: a.logp, reverse=True)
            
        return B

def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))

class Sequence():
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [blank] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale

        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp