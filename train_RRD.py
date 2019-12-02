import sys
import os

from learned_BP import *  # noqa
from itertools import product  # noqa


def multi_loss_CE(x, outputs, discount):
    return multi_loss(cross_entropy, x, outputs, discount)


def multi_loss_sBER(x, outputs, discount):
    return multi_loss(soft_BER, x, outputs, discount)


# Give name and directory of a set of trainings
exp_name = 'train_RRD'
exp_dir = os.path.join('.', exp_name)

if not os.path.exists(exp_dir):
    os.system(f'mkdir {exp_dir}')
if not os.path.exists(f'{exp_dir}/ckpts'):
    os.system(f'mkdir {exp_dir}/ckpts')
if not os.path.exists(f'{exp_dir}/logs'):
    os.system(f'mkdir {exp_dir}/logs')
if not os.path.exists(f'{exp_dir}/test_results'):
    os.system(f'mkdir {exp_dir}/test_results')

CUDA = False
handler_kwargs = {
    'checkpoint_dir': f'{exp_dir}/ckpts',           # directory to store training checkpoints
    'tensorboard_dir': f'{exp_dir}/logs',           # directory to store tensorboard logs
    'test_result_dir': f'{exp_dir}/test_results',   # directory to store simulation results
    'report_every': 50,                             # write one tensorboard point every such many mini-batches
    'use_cuda': CUDA,                               # whether to use CUDA
    'lr_init': 1e-3,                                # initial learning rate
    'weight_decay': 0.0,                            # L2 regularization term
    'grad_clip': 0.1,                               # gradient clipping
    'optimizer': 'RMSprop',                         # optimizer
    'lr_lambda': 'lambda ep: 0.8 ** (ep // 5)',     # learning rate decay along epoch number
    # multiloss discount as a function of initial discount factor and epoch number
    'discount_lambda': 'lambda d_init, ep: d_init * 0.5 ** (min(ep, 40) // 5)',
    # additional information to append at the end of the model name
    'name_suffix': 'lr_decay_factor=0.8,discount_decay_factor=0.9'
}

# create the linear code, mode determines parity-check matrix
code = BCH_Code(127, 64, mode='cr', use_cuda=CUDA)
# create the channel
channel = AWGN_Channel([1, 8])
# create data loader for generating mini-batches
loader = DataLoader(code, channel, 100, use_cuda=CUDA)

# train and simulate a set of learned RRD decoders over a grid
# RNN-SS/RNN-FW, turn damping on/off, turn mixing on/off, cross-entropy multiloss
for args in product(['simple', 'full'], [(1, False), (0.5, True)], [(1, False), (0.5, True)], [multi_loss_CE]):
    mode, (damping_init, damping_train), (mixing_init, mixing_train), loss = args
    # construct RRD decoder with T_in = 2 and T_out = 30
    model = RRD_BP_Decoder(
        code, T_rrd=30, tie_rrd=True, T=2, tie=True,
        mode=mode,
        damping_init=damping_init,
        damping_train=damping_train,
        mixing_init=mixing_init,
        mixing_train=mixing_train,
        use_cuda=CUDA
    )
    # create train handler
    handler = TrainHandler(loader, model, loss=loss, **handler_kwargs)
    print(handler.name())
    # train for 50 epochs, in this case 50k mini-batches
    handler.train(50)
    # simulate the decoder after training
    handler.test(np.arange(1, 6.5, 0.5), min_word_error=1000, min_mb_num=20000)
