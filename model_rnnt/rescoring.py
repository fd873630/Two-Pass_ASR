import torch
import torch.nn as nn
from torch import Tensor

class RescoringDecoder(nn.Module):
    def __init__(self, decoder, rnn_t, batch_size):
        super(RescoringDecoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = decoder.hidden_dim
        self.pad_id = decoder.pad_id
        self.eos_id = decoder.eos_id
        self.sos_id = decoder.sos_id
        self.device = decoder.device
        

        self.rnn_t = rnn_t
        self.ongoing_beams = None
        self.cumulative_ps = None
        self.finished = [[] for _ in range(batch_size)]
        self.finished_ps = [[] for _ in range(batch_size)]
        self.validate_args = decoder.validate_args
        self.forward_step = decoder.forward_step

    def forward(self, inputs, input_lengths, input_var, encoder_outputs, k):
        batch_size, hidden, attn = encoder_outputs.size(0), None, None

        preds = self.rnn_t.MCER_beam_search(inputs, input_lengths, W=5)

        final_result = []    
        # 여기부터! 결과값 tf 으로 집어넣기
        for a in preds:
            result = []
            qqq = [self.sos_id]
            qqq.extend(a.k[1:])
            qqq.append(self.eos_id)
            
            target = torch.tensor(qqq)
            target = target.unsqueeze(0).to(self.device)
            
            step_outputs, hidden, attn, _ = self.forward_step(target, hidden, encoder_outputs, attn)

            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                result.append(step_output)
            
            logits = torch.stack(result, dim=1).to(self.device)
            
            hypothesis = logits.max(-1)[1].squeeze()
            hypothesis = hypothesis.tolist()
            final_result.append(hypothesis)

        return final_result
            