import numpy as np
import torch
from torch.distributions import Binomial, Normal, Uniform


class Channel:
    def __init__(self, param_range):
        self.check_valid(param_range)
        self.param_range = param_range
        self.name = self.generate_channel_name()
        print('Successfully created channel', self.name)

    def generate_channel_name(self):
        return f'Channel_{self.param_range[0]}_{self.param_range[1]}'

    def __repr__(self):
        return self.name

    def check_valid(self, param_range):
        return NotImplemented

    def generate_sample(self, code, batch_size, param_range=None, all_zero=True):
        return NotImplemented


class BSC_Channel(Channel):
    def generate_channel_name(self):
        return f'BSC_{self.param_range[0]}_{self.param_range[1]}'

    def check_valid(self, param_range):
        param_min, param_max = param_range[0], param_range[1]
        if not (0 <= param_min <= param_max <= 1):
            raise Exception(f'Invalid parameters [{param_min}, {param_max}]')

    def generate_sample(self, code, batch_size, param_range=None, all_zero=True):
        if param_range:
            self.check_valid(param_range)
        else:
            param_range = self.param_range
        # Generate transmitted codeword and channel parameter
        x = code.generate_codeword(batch_size, all_zero)

        if not code.use_cuda:
            param = Uniform(
                low=param_range[0] * torch.ones((1, batch_size)),
                high=param_range[1] * torch.ones((1, batch_size))
            ).sample()
            # Generate noise and channel output, compute the channel LLR
            e = Binomial(total_count=1, probs=param.repeat(code.N, 1)).sample()
        else:
            param = torch.cuda.FloatTensor(1, batch_size).uniform_(*param_range)
            e = param.repeat(code.N, 1).bernoulli_()
        y = (x + e) % 2
        llr = (-1)**y * torch.log((1 - param) / param)
        return x, y, param, llr


class AWGN_Channel(Channel):
    def generate_channel_name(self):
        return f'AWGN_{self.param_range[0]}_{self.param_range[1]}'

    def check_valid(self, param_range):
        param_min, param_max = param_range[0], param_range[1]
        if not (param_min <= param_max):
            raise Exception(f'Invalid parameters [{param_min}, {param_max}]')

    def generate_sample(self, code, batch_size, param_range=None, all_zero=True):
        if param_range:
            self.check_valid(param_range)
        else:
            param_range = self.param_range
        # Generate transmitted codeword and channel parameter
        x = code.generate_codeword(batch_size, all_zero)
        # Generate noise and channel output, compute the channel LLR
        if not code.use_cuda:
            if batch_size % 10 == 0:
                h = np.linspace(param_range[0], param_range[1], 11)
                param = Uniform(
                    low=torch.tensor(np.kron(h[1:], np.ones(batch_size // 10))),
                    high=torch.tensor(np.kron(h[1:], np.ones(batch_size // 10)))
                ).sample().float()
            else:
                param = Uniform(
                    low=param_range[0] * torch.ones((1, batch_size)),
                    high=param_range[1] * torch.ones((1, batch_size))
                ).sample()
            sigma = 1 / torch.sqrt(2 * code.rate * 10**(param / 10))
            e = Normal(loc=0, scale=sigma.repeat(code.N, 1)).sample().float()
        else:
            if batch_size % 10 == 0:
                h = np.repeat(np.linspace(param_range[0], param_range[1], 11), batch_size // 10)
                param = torch.cuda.FloatTensor(h[batch_size // 10:])
                # param.sub_(torch.cuda.FloatTensor(1, batch_size).uniform_(0, (param_range[1] - param_range[0]) / 10))
            else:
                param = torch.cuda.FloatTensor(1, batch_size).uniform_(param_range[0], param_range[1])
            sigma = 1 / torch.sqrt(2 * code.rate * 10**(param / 10))
            e = sigma * torch.cuda.FloatTensor(code.N, batch_size).normal_()
        y = (-1)**x + e
        llr = 2 / sigma**2 * y
        return x, y, param, llr
