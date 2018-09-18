from learned_BP.models.BP_iteration import BP_Iteration
import numpy as np
import torch
from torch import nn


class BP_Decoder_Opt:
    def __init__(self, **kwargs):
        self.tie = kwargs.get('tie', False)
        self.T = kwargs.get('T', 30)
        self.damping_init = np.clip(kwargs.get('damping_init', 1.0), 1e-4, 1 - 1e-4)
        self.damping_train = kwargs.get('damping_train', True)
        self.use_cuda = kwargs.get('use_cuda', True)


class BP_Decoder(nn.Module):
    def __init__(self, code, **kwargs):
        super(BP_Decoder, self).__init__()
        opt = BP_Decoder_Opt(**kwargs)
        self.code, self.opt = code, opt

        self.gamma_logit = nn.Parameter(torch.Tensor(1), requires_grad=opt.damping_train)
        self.gamma_logit.data.fill_(np.log(opt.damping_init / (1 - opt.damping_init)))

        if opt.tie:
            self.iter = BP_Iteration(code.H, **kwargs)
            for t in range(1, opt.T + 1):
                setattr(self, f'iter_{t}', self.iter)
        else:
            for t in range(1, opt.T + 1):
                setattr(self, f'iter_{t}', BP_Iteration(code.H, **kwargs))

        if opt.use_cuda:
            self.cuda()

    def name(self):
        opt = self.opt
        ans = [f'T={opt.T}', f'tie={opt.tie}']
        if opt.damping_train:
            ans += [f'damping_train={opt.damping_init}']
        else:
            ans += [f'damping_fixed={opt.damping_init}']
        return ','.join(ans + [self.iter_1.name()])

    def iteration_number(self):
        return self.opt.T

    def BP_Iter(self, t, chn_llr, msg_C2V, msg_V2C):
        func = getattr(self, f'iter_{t}')
        gamma = torch.sigmoid(self.gamma_logit)
        return func(chn_llr, msg_C2V, msg_V2C, gamma)

    def forward(self, chn_llr):
        shape = (self.code.H.E, chn_llr.shape[1])
        msg_C2V, msg_V2C, outputs = torch.zeros(*shape), torch.zeros(*shape), []
        if self.opt.use_cuda:
            msg_C2V, msg_V2C = msg_C2V.cuda(), msg_V2C.cuda()
        for t in range(1, self.opt.T + 1):
            msg_C2V, msg_V2C, output = self.BP_Iter(t, chn_llr, msg_C2V, msg_V2C)
            outputs.append(output)
        return outputs
