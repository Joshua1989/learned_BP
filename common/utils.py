import numpy as np
from torch import abs, exp, log, tanh
import torch.nn.functional as F


#########################################################################################################
# Some utility functions
#########################################################################################################
def damping(msg_prev, msg_new, gamma):
    # Convex combination of new value and value from last iteration
    return (1 - gamma) * msg_prev + gamma * msg_new


def atanh(x, eps=1e-6):
    # The inverse hyperbolic tangent function, missing in pytorch.
    x = x * (1 - eps)
    return 0.5 * log((1.0 + x) / (1.0 - x))


def sigmoid(x):
    # Sigmoid function to convert log(p0/p1) into p0 and log(p1/p0) into p1
    return 1 / (1 + exp(-x))


def hard_decision(logit):
    # Make hard decision on LLR = log(p0/p1)
    return (logit < 0).float()


#########################################################################################################
# BP updating rules
#########################################################################################################
def vertical_step(H, ell, lam_hat, Wi=None, We=None):
    # Vertical step of BP, compute V2C message using C2V message
    if Wi is not None:
        ell *= Wi
    if We is not None:
        lam_hat *= We
    llr_int, llr_ext = H.col_gather(ell), H.col_sum_loo(lam_hat)
    return llr_int + llr_ext


def horizontal_step(H, lam, We=None, clip_thresh=15.0):
    # Horizontal step of BP, compute C2V message using V2C message
    lam = lam.clamp(-clip_thresh, clip_thresh)
    if We is not None:
        lam *= We
    sgn = (-1.0) ** H.row_sum_loo((lam < 0).float())

    abs_lam = abs(lam).clamp(-np.log(np.tanh(clip_thresh / 2)), clip_thresh)
    amp = H.row_sum_loo(log(tanh(abs_lam / 2)))
    # print(abs_lam, sgn, amp)
    return sgn * 2 * atanh(exp(amp))


def marginal_step(H, ell, lam_hat, Wi=None, We=None):
    # Marginalization step of BP, compute soft output using C2V message
    if Wi is not None:
        ell *= Wi
    if We is not None:
        lam_hat *= We
    return ell + H.col_sum(lam_hat)


#########################################################################################################
# Metrics
#########################################################################################################
def cross_entropy(x, llr):
    # x, llr are of size N-by-batch_size, LLR = log(p0/p1)
    return F.binary_cross_entropy_with_logits(-llr, x)


def bit_error_rate(x, x_hat):
    # x, x_hat are of size N-by-batch_size
    return abs(x - x_hat).mean()


def word_error_rate(x, x_hat):
    # x, x_hat are of size N-by-batch_size
    return (abs(x - x_hat).mean(dim=0) > 0).float().mean()
