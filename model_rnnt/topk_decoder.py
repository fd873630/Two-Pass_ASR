import torch
import torch.nn as nn
from torch import Tensor

def _inflate(tensor, n_repeat, dim):
    repeat_dims = [1] * len(tensor.size())
    repeat_dims[dim] *= n_repeat

    return tensor.repeat(*repeat_dims)


class TopKDecoder(nn.Module):
    def __init__(self, decoder, batch_size):
        super(TopKDecoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = decoder.hidden_dim
        self.pad_id = decoder.pad_id
        self.eos_id = decoder.eos_id
        self.sos_id = decoder.sos_id
        self.device = decoder.device
        #self.num_layers = decoder.num_layers
        self.ongoing_beams = None
        self.cumulative_ps = None
        self.finished = [[] for _ in range(batch_size)]
        self.finished_ps = [[] for _ in range(batch_size)]
        self.validate_args = decoder.validate_args
        self.forward_step = decoder.forward_step

    def forward(self, input_var, encoder_outputs, k):
        batch_size, hidden, attn = encoder_outputs.size(0), None, None

        max_length = 55
        inputs_add_sos = torch.LongTensor([self.sos_id]*batch_size).view(batch_size, 1)
        
        if input_var.is_cuda: inputs_add_sos = inputs_add_sos.cuda()
        
        step_outputs, hidden, attn, _ = self.forward_step(inputs_add_sos, hidden, encoder_outputs, attn)
        
        self.cumulative_ps, self.ongoing_beams = step_outputs.topk(k) # cumulative_ps : torch.Size([1, 1, 5])

        self.ongoing_beams = self.ongoing_beams.view(batch_size * k, 1) # torch.Size([5, 1])
        self.cumulative_ps = self.cumulative_ps.view(batch_size * k, 1) # torch.Size([5, 1])

        input_var = self.ongoing_beams

        encoder_dim = encoder_outputs.size(2)
        encoder_outputs = _inflate(encoder_outputs, k, dim=0)
        
        encoder_outputs = encoder_outputs.view(k, batch_size, -1, encoder_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.reshape(batch_size * k, -1, encoder_dim)
        hidden_0 = _inflate(hidden[0], k, dim=1)
        hidden_1 = _inflate(hidden[1], k, dim=1)
        
        hidden = (hidden_0, hidden_1)

        for di in range(max_length - 1):
            if self.is_all_finished(k):
                break
            
            step_outputs, hidden, attn, _  = self.forward_step(input_var, hidden, encoder_outputs, attn)
            step_outputs = step_outputs.view(batch_size, k, -1) # torch.Size([5, 1, 54]) -> torch.Size([1, 5, 54])
            
            current_ps, current_vs = step_outputs.topk(k) # current_vs : torch.Size([1, 5, 54]) -> torch.Size([1, 5, 5])
            #current_ps : 확률 , current_vs : index
            
            self.cumulative_ps = self.cumulative_ps.view(batch_size, k)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, k, -1)

            current_ps = (current_ps.permute(0, 2, 1) + self.cumulative_ps.unsqueeze(1)).permute(0, 2, 1) # [batch, index, probability]
            
            current_ps = current_ps.view(batch_size, k ** 2) # torch.Size([1, 25])
            current_vs = current_vs.view(batch_size, k ** 2) # torch.Size([1, 25])
            
            self.cumulative_ps = self.cumulative_ps.view(batch_size, k)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, k, -1)
            
            topk_current_ps, topk_status_ids = current_ps.topk(k) # torch.Size([1, 5])
            prev_status_ids = (topk_status_ids // k)

            topk_current_vs = torch.zeros((batch_size, k), dtype=torch.long)
            prev_status = torch.zeros(self.ongoing_beams.size(), dtype=torch.long)

            for batch_idx, batch in enumerate(topk_status_ids):
                for idx, topk_status_idx in enumerate(batch):
                    topk_current_vs[batch_idx, idx] = current_vs[batch_idx, topk_status_idx]
                    prev_status[batch_idx, idx] = self.ongoing_beams[batch_idx, prev_status_ids[batch_idx, idx]]
            
            self.ongoing_beams = torch.cat([prev_status, topk_current_vs.unsqueeze(2)], dim=2).to(self.device)
            self.cumulative_ps = topk_current_ps.to(self.device)
            
            '''
            if torch.any(topk_current_vs == self.eos_id):
                finished_ids = torch.where(topk_current_vs == self.eos_id)
                num_successors = [1] * batch_size

                for (batch_idx, idx) in zip(*finished_ids):
                    self.finished[batch_idx].append(self.ongoing_beams[batch_idx, idx])
                    self.finished_ps[batch_idx].append(self.cumulative_ps[batch_idx, idx])

                    if k != 1:
                        eos_cnt = self.get_successor(
                            current_ps=current_ps,
                            current_vs=current_vs,
                            finished_ids=(batch_idx, idx),
                            num_successor=num_successors[batch_idx],
                            eos_cnt=1,
                            k=k
                        )
                        num_successors[batch_idx] += eos_cnt
            '''
            input_var = self.ongoing_beams[:, :, -1]
            input_var = input_var.view(batch_size * k, -1)

        return self.ongoing_beams

    def is_all_finished(self, k):
        for done in self.finished:
            if len(done) < k:
                return False

        return True

    def fill_sequence(self, hypothesis):
        batch_size = len(hypothesis)
        max_length = -1

        for y_hat in hypothesis:
            if len(y_hat) > max_length:
                max_length = len(y_hat)

        matched = torch.zeros((batch_size, max_length), dtype=torch.long).to(self.device)

        for batch_idx, y_hat in enumerate(hypothesis):
            matched[batch_idx, :len(y_hat)] = y_hat
            matched[batch_idx, len(y_hat):] = int(self.pad_id)

        return matched

    def get_hypothesis(self):
        hypothesis = list()

        for batch_idx, batch in enumerate(self.finished):
            for idx, beam in enumerate(batch):
                self.finished_ps[batch_idx][idx] /= self.get_length_penalty(len(beam))

        for batch_idx, batch in enumerate(self.finished):
            # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
            if len(batch) == 0:
                prob_batch = self.cumulative_ps[batch_idx].to(self.device)
                top_beam_idx = int(prob_batch.topk(1)[1])
                hypothesis.append(self.ongoing_beams[batch_idx, top_beam_idx])

            # bring highest probability sentence
            else:
                top_beam_idx = int(torch.FloatTensor(self.finished_ps[batch_idx]).topk(1)[1])
                hypothesis.append(self.finished[batch_idx][top_beam_idx])

        hypothesis = self.fill_sequence(hypothesis).to(self.device)

        return hypothesis
    
    def get_successor(self, current_ps, current_vs, finished_ids, num_successor, eos_cnt, k):
        finished_batch_idx, finished_idx = finished_ids

        successor_ids = current_ps.topk(k + num_successor)[1]
        successor_idx = successor_ids[finished_batch_idx, -1]

        successor_p = current_ps[finished_batch_idx, successor_idx].to(self.device)
        successor_v = current_vs[finished_batch_idx, successor_idx].to(self.device)

        prev_status_idx = (successor_idx // k)
        prev_status = self.ongoing_beams[finished_batch_idx, prev_status_idx]
        prev_status = prev_status.view(-1)[:-1].to(self.device)

        successor = torch.cat([prev_status, successor_v.view(1)])

        if int(successor_v) == self.eos_id:
            self.finished[finished_batch_idx].append(successor)
            self.finished_ps[finished_batch_idx].append(successor_p)
            eos_cnt = self.get_successor(current_ps, current_vs, finished_ids, num_successor + eos_cnt, eos_cnt + 1, k)

        else:
            self.ongoing_beams[finished_batch_idx, finished_idx] = successor
            self.cumulative_ps[finished_batch_idx, finished_idx] = successor_p

        return eos_cnt

    def get_length_penalty(self, length: int, alpha: float = 1.2, min_length: int = 5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((min_length + length) / (min_length + 1)) ** alpha