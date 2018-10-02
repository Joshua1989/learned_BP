from collections import defaultdict, Counter
import glob
import io
from ipywidgets import VBox, Image
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pprint
import re
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR

# Use the correct progress bar for Jupyter and command line
try:
    __IPYTHON__
    from mytqdm import tqdm_notebook_EX as tqdm
    env = 'Jupyter'
except Exception:
    from tqdm import tqdm as tqdm
    env = 'shell'


def hard_decision(logit):
    # Make hard decision on LLR = log(p0/p1)
    return (logit < 0).float()


def cross_entropy(x, llr):
    # x, llr are of size N-by-batch_size, LLR = log(p0/p1)
    return F.binary_cross_entropy_with_logits(-llr, x)


def tanh_loss(x, llr, alpha=1):
    return (1 - (alpha * (1 - 2 * x) * llr).tanh()).mean() / 2 / alpha


def bit_error_rate(x, x_hat):
    # x, x_hat are of size N-by-batch_size
    return (x - x_hat).abs().mean()


def word_error_rate(x, x_hat):
    # x, x_hat are of size N-by-batch_size
    return ((x - x_hat).abs().mean(dim=0) > 0).float().mean()


def multi_loss_CE(x, outputs, discount=1):
    if isinstance(outputs[-1], list):
        outputs = [v for o in outputs for v in o]
    scale, num, den = 1, 0, 0
    for v in reversed(outputs):
        num += scale * cross_entropy(x, v)
        den += scale
        scale *= discount
        if scale == 0:
            break
    return num / den


def single_loss_CE(x, outputs):
    return multi_loss_CE(x, outputs, discount=0)


def multi_loss_tanh(x, outputs, alpha=1, discount=1):
    if isinstance(outputs[-1], list):
        outputs = [v for o in outputs for v in o]
    scale, num, den = 1, 0, 0
    for v in reversed(outputs):
        num += scale * tanh_loss(x, v, alpha)
        den += scale
        scale *= discount
        if scale == 0:
            break
    return num / den


def multi_loss_tanh_balanced(x, outputs, alpha=1, discount=1):
    if isinstance(outputs[-1], list):
        outputs = [v for o in outputs for v in o]
    scale, num, den = 1, 0, 0
    for v in reversed(outputs):
        num += scale * torch.sigmoid((2 * x - 1) * v).mean(dim=0)
        den += scale
        scale *= discount
        if scale == 0:
            break
    return 1 / (den / num).mean()


def single_loss_tanh(x, outputs):
    return multi_loss_tanh(x, outputs, discount=0)


class TrainHandler_Opt:
    def __init__(self, **kwargs):
        self.checkpoint_dir = kwargs.get('checkpoint_dir', '.')
        self.tensorboard_dir = kwargs.get('tensorboard_dir', '.')
        self.test_result_dir = kwargs.get('test_result_dir', '.')
        self.report_every = kwargs.get('report_every', 100)
        self.use_cuda = kwargs.get('use_cuda', True)

        self.lr_init = kwargs.get('lr_init', 1e-3)
        self.lr_floor = kwargs.get('lr_decay', 1.0)
        self.l2_lambda = kwargs.get('l2_lambda', 0.0)
        self.grad_clip = kwargs.get('grad_clip', 0.1)


class TrainHandler:
    def __init__(self, loader, model, loss=multi_loss_CE, **kwargs):
        opt = TrainHandler_Opt(**kwargs)
        self.loader, self.model, self.loss, self.opt = loader, model, loss, opt
        # use Adam optimizer with given initial learning rate and l2 regularization
        self.optimizer = RMSprop(model.parameters(), lr=opt.lr_init, weight_decay=opt.l2_lambda)
        # learning rate will half when there is no progress for three epochs
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda ep: 0.5 ** (ep // 5))

    def init_summary_writer(self):
        opt = self.opt
        save_dir = os.path.join(self.opt.checkpoint_dir, self.name())
        if not os.path.exists(save_dir):
            self.writer = SummaryWriter(os.path.join(opt.tensorboard_dir, self.name()))
            self.writer.add_text('Training Handler Parameters',
                                 pprint.pformat(self.opt.__dict__))
            self.writer.add_text('Model Parameters',
                                 pprint.pformat({k: v.opt.__dict__ for k, v in self.model.named_modules() if hasattr(v, 'opt')}))
        else:
            self.writer = SummaryWriter(os.path.join(opt.tensorboard_dir, self.name()))

    def name(self):
        return ','.join([self.loader.name(), self.model.name(), self.loss.__name__, f'lr_init={self.opt.lr_init}'])

    def __repr__(self):
        return pprint.pformat(self.opt.__dict__) + '\n' + \
            pprint.pformat({k: v.opt.__dict__ for k, v in self.model.named_modules()})

    def load_ckpt(self, save_dir=None):
        save_dir = save_dir or os.path.join(self.opt.checkpoint_dir, self.name())
        s = save_dir.replace('[', '@1').replace(']', '@2')
        s = s.replace('@1', '[[]').replace('@2', '[]]')
        ckpt_files = sorted(glob.glob(s + '/epoch=*.pth'))
        if os.path.exists(save_dir) and len(ckpt_files) > 0:
            ckpt_file = ckpt_files[-1]
            ckpt_dict, model_dict = torch.load(ckpt_file), self.model.state_dict()
            for k, v in ckpt_dict.items():
                if k in model_dict:
                    model_dict[k] = v
            self.model.load_state_dict(model_dict)
            self.scheduler.last_epoch = int(re.search('epoch=(\d+)', ckpt_file).group(1))
        else:
            self.scheduler.last_epoch = 0

    def save_ckpt(self, temp=False):
        save_dir = os.path.join(self.opt.checkpoint_dir, self.name())
        if not os.path.exists(save_dir):
            os.system(f'mkdir {save_dir}')
        if temp:
            ckpt_file = os.path.join(save_dir, 'temp.pth')
            if os.path.exists(ckpt_file):
                os.system(f'rm {ckpt_file}')
        else:
            ckpt_file = os.path.join(save_dir, f'epoch={self.scheduler.last_epoch:03d}.pth')
        torch.save(self.model.state_dict(), ckpt_file)

    def report_progress(self, mb_idx, x, param, outputs, cost):
        mb_count = self.scheduler.last_epoch * len(self.loader) + mb_idx
        # convert list into dict
        if isinstance(outputs[-1], list):
            x_hat_final = hard_decision(outputs[-1][-1])
            outputs = {f'outer_iter_{t_out:03d}.iter_{t:03d}': o
                       for t_out, block in enumerate(outputs, 1)
                       for t, o in enumerate(block, 1)}
        else:
            x_hat_final = hard_decision(outputs[-1])
            outputs = {f'iter_{t:03d}': o for t, o in enumerate(outputs, 1)}
        # compute metrics w.r.t. channel parameter for this mini-batch
        ber = abs(x - x_hat_final).mean(dim=0)
        wer = (ber > 0).float()
        for tup in zip(param, ber, wer):
            p, b, w = map(float, tup)
            self.stat[p].append((b, w))
        # compute hard decision decoded codeword, BER and WER
        x_hats = {k: hard_decision(o) for k, o in outputs.items()}
        BER = Counter({k: bit_error_rate(x, x_hat) for k, x_hat in x_hats.items()})
        WER = Counter({k: word_error_rate(x, x_hat) for k, x_hat in x_hats.items()})
        # cumulate metrics or write into summary
        if mb_idx % self.opt.report_every != 0:
            self.cum_cost += cost
            self.cum_BER, self.cum_WER = self.cum_BER + BER, self.cum_WER + WER
            return {}
        else:
            try:
                for var, tensor in dict(self.model.named_parameters()).items():
                    if 'Wi' in var or 'We' in var:
                        self.writer.add_scalar(f'weights/{var}', tensor.mean(), mb_count)
                    elif 'beta' in var or 'gamma' in var:
                        self.writer.add_scalar(
                            f"parameter/{var.replace('_logit', '')}",
                            torch.sigmoid(tensor), mb_count
                        )
                lr = [group['lr'] for group in self.optimizer.param_groups][0]
                self.writer.add_scalar('progress/learning_rate', lr, mb_count)

                k = self.opt.report_every
                ret = {
                    'loss': float(self.cum_cost / k),
                    'BER': float(self.cum_BER[max(self.cum_BER)] / k),
                    'WER': float(self.cum_WER[max(self.cum_WER)] / k)
                }
                self.writer.add_scalar('progress/loss', ret['loss'], mb_count)
                self.writer.add_scalar('progress/BER', ret['BER'], mb_count)
                self.writer.add_scalar('progress/WER', ret['WER'], mb_count)

                # for var, metric in self.cum_BER.items():
                #     self.writer.add_scalar(f'BER/{var}', np.log10(metric / k), mb_count)
                # for var, metric in self.cum_WER.items():
                #     self.writer.add_scalar(f'WER/{var}', np.log10(metric / k), mb_count)

                # reset cumulate metrics
                self.cum_cost, self.cum_BER, self.cum_WER = 0, Counter(), Counter()
                return ret
            except Exception:
                print(x_hats, BER, WER)
                return {}

    def write_plot(self):
        plt.switch_backend('agg')
        epoch = self.scheduler.last_epoch
        dic = {k: np.array(v).mean(axis=0) for k, v in self.stat.items()}
        param, ber, wer = np.array(sorted((k, *v) for k, v in dic.items())).T

        ber_fig = plt.figure()
        plt.semilogy(param, ber)
        plt.grid()
        plt.xlabel('channel parameter')
        plt.ylabel('BER')
        plt.xlim(self.loader.channel.param_range)
        plt.ylim((1e-6, 1))
        self.writer.add_figure('BER', ber_fig, epoch)

        wer_fig = plt.figure()
        plt.semilogy(param, wer)
        plt.grid()
        plt.xlabel('channel parameter')
        plt.ylabel('WER')
        plt.xlim(self.loader.channel.param_range)
        plt.ylim((1e-6, 1))
        self.writer.add_figure('WER', wer_fig, epoch)

        pw_fig = plt.figure()
        if hasattr(self.model, 'iter_1'):
            layer = self.model.iter_1
        elif hasattr(self.model, 'outer_iter_1'):
            layer = self.model.outer_iter_1.iter_1
        try:
            if len(layer.pw_linear.opt.pw_init) > 0:
                x = torch.linspace(-layer.opt.llr_clip, layer.opt.llr_clip, 101).reshape((1, -1))
                if self.opt.use_cuda:
                    x = x.cuda()
                y = layer.pw_linear(x)
                plt.plot(x[0].cpu().detach().numpy(), y[0].cpu().detach().numpy())
                points = layer.pw_linear.log_points.exp().cpu().detach().numpy().reshape(-1)
                points = np.round(np.concatenate(([0], points)), 2)
                slopes = layer.pw_linear.log_slopes.exp().cpu().detach().numpy().reshape(-1)
                slopes = np.round(slopes, 2)
                plt.title(f'points: {points}\nslopes: {slopes}')
                plt.xlim((-layer.opt.llr_clip, layer.opt.llr_clip))
                plt.ylim((-layer.opt.llr_clip, layer.opt.llr_clip))
                self.writer.add_figure('piecewise linear', pw_fig, epoch)
        except Exception:
            pass

    def train(self, max_epoch):
        trainable_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        trainable = len(trainable_params) > 0
        max_epoch = max_epoch if trainable else 1
        # load last checkpoint
        self.load_ckpt()
        self.init_summary_writer()
        if env == 'Jupyter':
            vbox = VBox()
            display(vbox)
            kwargs = {'vbox': vbox, 'leave': False}
        else:
            kwargs = {'leave': False}
        self.model.train()
        with tqdm(total=max_epoch, **kwargs) as outer_pbar:
            for epoch in range(max_epoch):
                if epoch >= self.scheduler.last_epoch:
                    # initialize cumulative results
                    self.cum_cost, self.epoch_cost = 0, 0
                    self.cum_BER, self.cum_WER = Counter(), Counter()
                    self.stat = defaultdict(list)
                    with tqdm(total=len(self.loader), **kwargs) as inner_pbar:
                        for mb_idx, (x, y, param, llr) in self.loader.generator():
                            # feed forward to compute cost
                            outputs = self.model(llr)
                            cost = self.loss(x, outputs)
                            self.epoch_cost += float(cost)
                            # back propagation with gradient clipping
                            if trainable:
                                self.optimizer.zero_grad()
                                cost.backward()
                                clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
                                self.optimizer.step()
                            # print result regularly
                            ret = self.report_progress(mb_idx, x, param, outputs, cost)
                            if ret:
                                inner_pbar.set_postfix(**ret)
                            inner_pbar.update(1)
                    # save recent epoch
                    self.scheduler.step()
                    self.save_ckpt()
                    self.write_plot()
                outer_pbar.update(1)
        print(f'finished training for {max_epoch} epochs')

    def test(self, chn_param_list, min_mb_num=10000, min_word_error=100, all_zero=False, rerun=False):
        # load previously saved result
        def default_value():
            return {
                'cum_loss_CE': np.zeros(self.model.iteration_number()),
                'cum_loss_tanh': np.zeros(self.model.iteration_number()),
                'bit_error': np.zeros(self.model.iteration_number()),
                'word_error': np.zeros(self.model.iteration_number()),
                'mb_count': 0
            }
        test_result_file = os.path.join(self.opt.test_result_dir, self.loader.code.name + '.pickle')
        try:
            with open(test_result_file, 'rb') as f:
                full_result = pickle.load(f)
        except Exception:
            full_result = {}
        if rerun or self.name() not in full_result:
            full_result[self.name()] = {}
        model_result = full_result[self.name()]

        # initialized image area
        image_area = Image(format='png')
        display(image_area)

        def iter_plot(chn_param, param_result):
            plt.switch_backend('agg')
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.semilogy(param_result['bit_error'] / param_result['mb_count'] / self.loader.code.N / self.loader.batch_size)
            plt.grid()
            plt.xlabel('BP Iteration')
            plt.ylabel('BER')
            plt.title(f'BER for channel parameter {chn_param}')
            plt.subplot(1, 2, 2)
            plt.semilogy(param_result['word_error'] / param_result['mb_count'] / self.loader.batch_size)
            plt.grid()
            plt.xlabel('BP Iteration')
            plt.ylabel('WER')
            plt.title(f'WER for channel parameter {chn_param}')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_area.value = buf.read()

        def param_plot(iter_idx, model_result):
            if not model_result:
                return
            param_list = sorted(model_result.keys())
            if iter_idx < 0:
                iter_idx += self.model.iteration_number()
            ber = [model_result[p]['bit_error'][iter_idx] / model_result[p]['mb_count'] / self.loader.code.N / self.loader.batch_size for p in param_list]
            wer = [model_result[p]['word_error'][iter_idx] / model_result[p]['mb_count'] / self.loader.batch_size for p in param_list]

            plt.switch_backend('agg')
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.semilogy(param_list, ber)
            plt.grid()
            plt.xlabel('Channel Parameter')
            plt.ylabel('BER')
            plt.title(f'BER after {iter_idx + 1} BP iteration')
            plt.subplot(1, 2, 2)
            plt.semilogy(param_list, wer)
            plt.grid()
            plt.xlabel('Channel Parameter')
            plt.ylabel('WER')
            plt.title(f'WER after {iter_idx + 1} BP iteration')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_area.value = buf.read()

        # set model to evaluation mode and test
        self.model.eval()
        with tqdm(total=len(chn_param_list), leave=False) as chn_pbar:
            for chn_param in chn_param_list:
                chn_pbar.set_postfix(channel_param=chn_param)
                if chn_param not in model_result:
                    model_result[chn_param] = default_value()
                param_result = model_result[chn_param]
                with tqdm(total=min_mb_num, leave=False) as pbar:
                    for i in range(min_mb_num):
                        # update log on every 100 mini-batch
                        if i % 1 == 0 or param_result['word_error'][-1] >= min_word_error:
                            with open(test_result_file, 'wb') as f:
                                pickle.dump(full_result, f)
                            pbar.set_postfix(
                                bit_error=param_result['bit_error'][-1],
                                word_error=param_result['word_error'][-1],
                                mb_count=param_result['mb_count']
                            )
                            pbar.update(i - pbar.n)
                            # if word error count already exceeds given threshold, break the loop
                            if param_result['word_error'][-1] >= min_word_error:
                                pbar.update(min_mb_num - pbar.n)
                                break
                        # test a new mini-batch
                        if i >= param_result['mb_count']:
                            x, y, param, llr = self.loader.next_batch([chn_param, chn_param], all_zero)
                            outputs = self.model(llr)
                            if isinstance(outputs[-1], list):
                                outputs = [o for block in outputs for o in block]
                            cum_loss_CE_curr = [cross_entropy(x, o).data.cpu() for o in outputs]
                            cum_loss_tanh_curr = [tanh_loss(x, o).data.cpu() for o in outputs]
                            bit_error_curr = [bit_error_rate(x, hard_decision(o)).data.cpu() for o in outputs]
                            word_error_curr = [word_error_rate(x, hard_decision(o)).data.cpu() for o in outputs]
                            param_result['cum_loss_tanh'] += np.array(cum_loss_tanh_curr)
                            param_result['cum_loss_CE'] += np.array(cum_loss_CE_curr)
                            param_result['bit_error'] += np.array(bit_error_curr) * self.loader.code.N * self.loader.batch_size
                            param_result['word_error'] += np.array(word_error_curr) * self.loader.batch_size
                            param_result['mb_count'] += 1
                            del x, y, param, llr

                chn_pbar.update(1)
                iter_plot(chn_param, param_result)
                with open(test_result_file, 'wb') as f:
                    pickle.dump(full_result, f)
            param_plot(-1, model_result)
