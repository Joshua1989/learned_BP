from learned_BP.models.BP_decoder import BP_Decoder
import numpy as np
import torch
from torch import nn


def mixing(msg_prev, msg_new, beta):
    # Convex combination of new value and value from last iteration
    return (1 - beta) * msg_prev + beta * msg_new


class RRD_BP_Decoder_Opt:
    def __init__(self, **kwargs):
        self.tie_rrd = kwargs.get('tie_rrd', True)
        self.T_rrd = kwargs.get('T_rrd', 30)
        self.mixing_init = np.clip(kwargs.get('mixing_init', 1.0), 1e-4, 1 - 1e-4)
        self.mixing_train = kwargs.get('mixing_train', True)
        self.use_cuda = kwargs.get('use_cuda', True)


class RRD_BP_Decoder(nn.Module):
    def __init__(self, code, **kwargs):
        super(RRD_BP_Decoder, self).__init__()
        opt = RRD_BP_Decoder_Opt(**kwargs)
        self.code, self.opt = code, opt

        self.beta_logit = nn.Parameter(torch.Tensor(1), requires_grad=opt.mixing_train)
        self.beta_logit.data.fill_(np.log(opt.mixing_init / (1 - opt.mixing_init)))

        if opt.tie_rrd:
            self.outer_iter = BP_Decoder(code, **kwargs)
            for t in range(1, opt.T_rrd + 1):
                setattr(self, f'outer_iter_{t}', self.outer_iter)
        else:
            for t in range(1, opt.T_rrd + 1):
                setattr(self, f'outer_iter_{t}', BP_Decoder(code, **kwargs))

        if opt.use_cuda:
            self.cuda()

    def name(self):
        opt = self.opt
        ans = [f'T_rrd={opt.T_rrd}', f'tie_rrd={opt.tie_rrd}']
        if opt.mixing_train:
            ans += [f'mixing_train={opt.mixing_init}']
        else:
            ans += [f'mixing_fixed={opt.mixing_init}']
        return ','.join(ans + [self.outer_iter_1.name()])

    def iteration_number(self):
        return sum(getattr(self, f'outer_iter_{t}').iteration_number() for t in range(1, self.opt.T_rrd + 1))

    def Outer_Iter(self, t, chn_llr, soft_outut):
        # mix channel LLR and soft output of last outer iteration
        beta = torch.sigmoid(self.beta_logit)
        soft_input = mixing(chn_llr, soft_outut, beta)
        # apply code automorphism permutation
        perm = self.code.random_automorphism()[0]
        soft_input = soft_input[perm]
        # apply BP inner iteration
        outputs = getattr(self, f'outer_iter_{t}')(soft_input)
        # invert the permutation to get output
        inv_perm = np.argsort(perm)
        outputs = [v[inv_perm] for v in outputs]
        return outputs

    def forward(self, chn_llr):
        soft_outut, outputs = chn_llr.clone(), []
        for t in range(1, self.opt.T_rrd + 1):
            output = self.Outer_Iter(t, chn_llr, soft_outut)
            soft_outut = output[-1]
            outputs.append(output)
        return outputs
