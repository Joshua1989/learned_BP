import numpy as np
import torch
from torch import nn
from learned_BP.models.piecewise_linear import Piecewise_Linear


def atanh(x, eps=1e-6):
    # The inverse hyperbolic tangent function, missing in pytorch.
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def damping(msg_prev, msg_new, gamma):
    # Convex combination of new value and value from last iteration
    return (1 - gamma) * msg_prev + gamma * msg_new


def phi_func(x, llr_clip):
    phi_min = np.log((np.exp(llr_clip) + 1) / (np.exp(llr_clip) - 1))
    x = x.clamp(phi_min, llr_clip)
    return torch.log((torch.exp(x) + 1) / (torch.exp(x) - 1))


class BP_Iteration_Opt:
    def __init__(self, **kwargs):
        self.llr_clip = kwargs.get('llr_clip', 15.0)
        self.mode = kwargs.get('mode', 'simple')
        self.Wi_train = kwargs.get('Wi_train', True)
        self.Wi_mean_init = kwargs.get('Wi_mean_init', 1)
        self.Wi_std_init = kwargs.get('Wi_std_init', 1e-3) * self.Wi_train
        self.We_train = kwargs.get('We_train', True)
        self.We_mean_init = kwargs.get('We_mean_init', 1)
        self.We_std_init = kwargs.get('We_std_init', 1e-3) * self.We_train
        self.use_cuda = kwargs.get('use_cuda', True)


class BP_Iteration(nn.Module):
    def __init__(self, H, **kwargs):
        super(BP_Iteration, self).__init__()
        opt = BP_Iteration_Opt(**kwargs)
        self.pw_linear = Piecewise_Linear(**kwargs)
        if self.pw_linear.K > 0:
            opt.We_train, opt.We_mean_init = False, 1
        self.H, self.opt = H, opt

        if opt.mode == 'plain':
            self.Wi = self.We = None
        else:
            if opt.mode == 'simple':
                self.Wi = nn.Parameter(torch.Tensor(1), requires_grad=opt.Wi_train)
                self.We = nn.Parameter(torch.Tensor(1), requires_grad=opt.We_train)
            elif opt.mode == 'full':
                self.Wi = nn.Parameter(torch.Tensor(self.H.N, 1), requires_grad=opt.Wi_train)
                self.We = nn.Parameter(torch.Tensor(self.H.E, 1), requires_grad=opt.We_train)
            else:
                raise Exception(f'Unrecognized mode: {opt.mode}')
            if opt.Wi_train:
                self.Wi.data.normal_(mean=opt.Wi_mean_init, std=opt.Wi_std_init)
            else:
                self.Wi.data.fill_(opt.Wi_mean_init)
            if opt.We_train:
                self.We.data.normal_(mean=opt.We_mean_init, std=opt.We_std_init)
            else:
                self.We.data.fill_(opt.We_mean_init)

    def name(self):
        opt = self.opt
        ans = [f'mode={opt.mode}', f'llr_clip={opt.llr_clip}', f'pw_init={str(self.pw_linear.opt.pw_init).replace(" ", "")}']
        if self.opt.mode != 'plain':
            if opt.Wi_train:
                ans += [f'Wi_train={opt.Wi_mean_init}']
            else:
                ans += [f'Wi_fixed={opt.Wi_mean_init}']
            if opt.We_train:
                ans += [f'We_train={opt.We_mean_init}']
            else:
                ans += [f'We_fixed={opt.We_mean_init}']
        return ','.join(ans)

    def V_Step(self, ell, lam_hat):
        '''
        Vertical step of BP, compute V2C message using C2V message
        Arguments:
            ell {torch.tensor} -- Channel LLR (N x B)
            lam_hat {torch.tensor} -- V2C message (E x B)
        Returns:
            torch.tensor -- C2V message (E x B)
        '''
        # if not plain BP, multiply messages with weights
        if self.opt.mode != 'plain':
            ell, lam_hat = self.Wi * ell, self.We * lam_hat
        llr_int, llr_ext = self.H.col_gather(ell), self.H.col_sum_loo(lam_hat)
        # return llr_int + llr_ext.clamp(-self.opt.llr_clip, self.opt.llr_clip)
        return llr_int + (self.pw_linear(llr_ext) if self.pw_linear.K > 0 else llr_ext)

    def H_Step(self, lam):
        '''
        Horizontal step of BP, compute C2V message using V2C message
        Arguments:
            lam {torch.tensor} -- C2V message (E x B)
        Returns:
            torch.tensor -- V2C message (E x B)
        '''
        # prod_alfa = (-1.0) ** self.H.row_sum_loo((lam < 0).float())
        # phi = phi_func(torch.abs(lam), self.opt.llr_clip)
        # phi_tot = phi_func(self.H.row_sum_loo(phi), self.opt.llr_clip)
        # return (prod_alfa * phi_tot).clamp(-self.opt.llr_clip, self.opt.llr_clip)
        # clamp message for numerical stability
        lam = lam.clamp(-self.opt.llr_clip, self.opt.llr_clip)
        # sign of output
        sgn = (-1.0) ** self.H.row_sum_loo((lam < 0).float())
        # amplitude of output
        abs_lam = abs_lam.abs().clamp(-np.log(np.tanh(self.opt.llr_clip / 2)), self.opt.llr_clip)
        amp = self.H.row_sum_loo(torch.tanh(abs_lam / 2).log())
        # return (sgn * 2 * atanh(torch.exp(amp))).clamp(-self.opt.llr_clip, self.opt.llr_clip)
        return (sgn * 2 * atanh(amp.exp()))

    def M_Step(self, ell, lam_hat):
        '''
        Marginalization step of BP, compute soft output using C2V message
        Arguments:
            ell {torch.tensor} -- Channel LLR (N x B)
            lam_hat {torch.tensor} -- V2C message (E x B)
        Returns:
            torch.tensor -- marginal output (N x B)
        '''
        if self.opt.mode != 'plain':
            ell, lam_hat = self.Wi * ell, self.We * lam_hat
        # return ell + self.H.col_sum(lam_hat).clamp(-self.opt.llr_clip, self.opt.llr_clip)
        llr_ext = self.H.col_sum(lam_hat)
        return ell + (self.pw_linear(llr_ext) if self.pw_linear.K > 0 else llr_ext)

    def forward(self, chn_llr, msg_C2V, msg_V2C, gamma):
        msg_V2C = damping(msg_V2C, self.V_Step(chn_llr, msg_C2V), gamma)
        msg_C2V = damping(msg_C2V, self.H_Step(msg_V2C), gamma)
        output = self.M_Step(chn_llr, msg_C2V)
        return msg_C2V, msg_V2C, output
