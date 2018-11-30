import numpy as np
import torch
from torch import nn


def atanh(x, eps=1e-6):
    # The inverse hyperbolic tangent function, missing in pytorch.
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def damping(msg_prev, msg_new, gamma):
    # Convex combination of new value and value from last iteration
    return (1 - gamma) * msg_prev + gamma * msg_new


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        out = self.linear2(h_relu).sigmoid()
        return out


class AdaptiveNet(torch.nn.Module):
    def __init__(self, D_in, width):
        super(AdaptiveNet, self).__init__()
        self.gamma_nets = TwoLayerNet(D_in, width, 1)
        self.Wi_nets = TwoLayerNet(D_in, width, 1)
        self.We_nets = TwoLayerNet(D_in, width, 1)

    def forward(self, x):
        gamma = self.gamma_nets(x)
        Wi = self.Wi_nets(x)
        We = self.We_nets(x)
        return torch.stack([gamma, Wi, We], dim=1).squeeze()


class AdaBP_Decoder_Opt:
    def __init__(self, **kwargs):
        self.llr_clip = kwargs.get('llr_clip', 15.0)
        self.T = kwargs.get('T', 30)
        self.max_weight = kwargs.get('max_weight', 1.0)
        self.est_SNR = kwargs.get('est_SNR', True)
        self.use_cuda = kwargs.get('use_cuda', True)


class AdaBP_Decoder(nn.Module):
    def __init__(self, code, **kwargs):
        super(AdaBP_Decoder, self).__init__()
        self.code, self.H, self.opt = code, code.H, AdaBP_Decoder_Opt(**kwargs)
        if self.opt.est_SNR:
            self.adapter_nn = AdaptiveNet(1, 20)
        else:
            self.adapter_nn = AdaptiveNet(code.N, code.N)

        if self.opt.use_cuda:
            self.cuda()

    def name(self):
        opt = self.opt
        return f'T={opt.T},mw={opt.max_weight},est_SNR,adaBP' if opt.est_SNR else f'T={opt.T},mw={opt.max_weight},adaBP'

    def iteration_number(self):
        return self.opt.T

    def V_Step(self, ell, lam_hat, Wi, We):
        '''
        Vertical step of BP, compute V2C message using C2V message
        Arguments:
            ell {torch.tensor} -- Channel LLR (N x B)
            lam_hat {torch.tensor} -- V2C message (E x B)
        Returns:
            torch.tensor -- C2V message (E x B)
        '''
        ell, lam_hat = Wi * ell, We * lam_hat
        llr_int, llr_ext = self.H.col_gather(ell), self.H.col_sum_loo(lam_hat)
        return llr_int + llr_ext

    def H_Step(self, lam):
        '''
        Horizontal step of BP, compute C2V message using V2C message
        Arguments:
            lam {torch.tensor} -- C2V message (E x B)
        Returns:
            torch.tensor -- V2C message (E x B)
        '''
        # clamp message for numerical stability
        lam = lam.clamp(-self.opt.llr_clip, self.opt.llr_clip)
        # sign of output
        sgn = (-1.0) ** self.H.row_sum_loo((lam < 0).float())
        # amplitude of output
        abs_lam = abs_lam.abs().clamp(-np.log(np.tanh(self.opt.llr_clip / 2)), self.opt.llr_clip)
        amp = self.H.row_sum_loo(torch.log(torch.tanh(abs_lam / 2)))
        # print(abs_lam, sgn, amp)
        return sgn * 2 * atanh(amp.exp())

    def M_Step(self, ell, lam_hat, Wi, We):
        '''
        Marginalization step of BP, compute soft output using C2V message
        Arguments:
            ell {torch.tensor} -- Channel LLR (N x B)
            lam_hat {torch.tensor} -- V2C message (E x B)
        Returns:
            torch.tensor -- marginal output (N x B)
        '''
        ell, lam_hat = Wi * ell, We * lam_hat
        return ell + self.H.col_sum(lam_hat)

    def forward(self, chn_llr):
        # chn_llr N x B, use adapter NN to generate BP parameters
        if self.opt.est_SNR:
            E = (chn_llr ** 2).mean(dim=0)
            snr_hat = 10 * ((E / (1 + (1 + E).sqrt())) / (4 * self.code.rate)).log10()
            params = self.adapter_nn(snr_hat.reshape((-1, 1))).t()
        else:
            params = self.adapter_nn(chn_llr.t()).t()
        gamma, Wi, We = [tensor.reshape((1, -1)) for tensor in params]
        Wi, We = self.opt.max_weight * Wi, self.opt.max_weight * We
        # Initialize BP messages
        shape = (self.code.H.E, chn_llr.shape[1])
        msg_C2V, msg_V2C, outputs = torch.zeros(*shape), torch.zeros(*shape), [None] * self.opt.T
        if self.opt.use_cuda:
            msg_C2V, msg_V2C = msg_C2V.cuda(), msg_V2C.cuda()
        # Run BP for T iterations
        for t in range(self.opt.T):
            msg_V2C = damping(msg_V2C, self.V_Step(chn_llr, msg_C2V, Wi, We), gamma)
            msg_C2V = damping(msg_C2V, self.H_Step(msg_V2C), gamma)
            outputs[t] = self.M_Step(chn_llr, msg_C2V, Wi, We)
        return outputs
