import numpy as np
import torch
from torch import nn


class Piecewise_Linear_Opt:
    def __init__(self, **kwargs):
        self.pw_init = kwargs.get('pw_init', [])


class Piecewise_Linear(nn.Module):
    def __init__(self, **kwargs):
        super(Piecewise_Linear, self).__init__()
        opt = Piecewise_Linear_Opt(**kwargs)
        self.opt, self.K = opt, len(opt.pw_init)
        self.log_points = nn.Parameter(torch.tensor(opt.pw_init).log().float().reshape(-1, 1, 1), requires_grad=self.K > 0)
        self.log_slopes = nn.Parameter(torch.zeros(self.K + 1, 1, 1), requires_grad=self.K > 0)
        self.relu = nn.ReLU()

    def __repr__(self):
        return 'Piecewise_Linear with initial value: ' + str(self.opt.pw_init)

    def forward(self, x):
        slope, points = self.log_slopes.exp(), self.log_points.exp()
        slope0, slope_diff = slope[0], slope[1:] - slope[:-1]
        ans = slope0 * x
        if self.K > 0:
            ans += x.sign() * (slope_diff * self.relu(x.abs() - points)).sum(dim=0)
        return ans
