import pickle
import re
import glob
from collections import defaultdict
import plotly.offline as off
import plotly.graph_objs as go
from ipywidgets import interact
from pandas import DataFrame
off.init_notebook_mode(connected=True)


def export_results(exp_name, batch_size=100):
    B = batch_size
    result = defaultdict(lambda: defaultdict(list))
    pickle_files = glob.glob(f'{exp_name}/test_results/*.pickle')
    for file in pickle_files:
        match = re.search('_(\d+)_(\d+).pickle', file)
        N, K = int(match.group(1)), int(match.group(2))
        for k, v in sorted(pickle.load(open(file, 'rb')).items()):
            if k in result:
                print(f'model {k} already exists')
            for snr, vv in sorted(v.items()):
                sample_count = vv['mb_count'] * B
                result[k]['SNR'].append(snr)
                for t, bit_err in enumerate(vv['bit_error'], 1):
                    result[k]['BER' + str(t).zfill(2)].append(bit_err / sample_count / N)
                for t, word_err in enumerate(vv['word_error'], 1):
                    result[k]['WER' + str(t).zfill(2)].append(word_err / sample_count)
                result[k]['sample_count'].append(sample_count)
    return {k: DataFrame(v) for k, v in result.items()}


def filter_function(**d):
    def f(key):
        if d.get('channel', '') not in key:
            return False
        if d.get('T', '') not in key:
            return False
        if d.get('mode', '') not in key:
            return False
        if d.get('loss', '') not in key:
            return False
        if d.get('optimizer', '') not in key:
            return False
        if 'code' in d:
            return d['code'] + ',' in key
        if 'damping' in d:
            if not d['damping']:
                return 'damping_fixed=0.9999' in key or 'no-damping' in key
            else:
                return 'damping_fixed=0.9999' not in key or 'no-damping' not in key
        if 'mixing' in d:
            if not d['mixing']:
                return 'mixing_fixed=0.9999' in key or 'no-mixing' in key
            else:
                return 'mixing_fixed=0.9999' not in key or 'no-mixing' not in key
        return True
    return f


def rename_function(show_code=False, show_loss=True, show_opt=True, show_tm=False, show_channel=False):
    def f(key):
        segs = []
        if show_code:
            segs.append(key.split(',')[0])
        if 'mode' not in key:
            segs.append('adaBP')
        else:
            segs.append(re.search('mode=(\w+),', k).group(1))
        if 'rrd' in key:
            if 'mixing_fixed=0.9999' not in key and 'no-mixing' not in key:
                if 'mixing_fixed' in key:
                    segs.append('mix' + re.search('mixing_fixed=([-+]?[0-9]*\.?[0-9]+)', k).group(1))
                else:
                    segs.append('mix')
            else:
                segs.append('no_mix')
        if 'damping_fixed=0.9999' not in key and 'no-damping' not in key:
            if 'damping_fixed' in key:
                segs.append('damp' + re.search('damping_fixed=([-+]?[0-9]*\.?[0-9]+)', k).group(1))
            else:
                segs.append('damp')
        else:
            segs.append('no_damp')
        if show_loss:
            if 'CE' in key:
                segs.append('CE')
            if 'sBER' in key:
                segs.append('sBER')
        if show_opt:
            if 'Adam' in key:
                segs.append('Adam')
            if 'RMSprop' in key:
                segs.append('RMSprop')
        if show_tm:
            if 'r_decay' in k:
                segs.append('lr' + re.search('r_decay_factor=([-+]?[0-9]*\.?[0-9]+)', k).group(1))
            if 'discount_decay' in k:
                segs.append('disc' + re.search('discount_decay_factor=([-+]?[0-9]*\.?[0-9]+)', k).group(1))
        if show_channel:
            segs.append(key.split(',')[1])
        return '-'.join(segs)
    return f


def generate_dropdown(keys):
    dropdowns = {}
    for items in zip(*[k.split('-') for k in keys]):
        if len(set(items)) == 1:
            continue
        if 'BCH' in items[0] or 'RM' in items[0]:
            dropdowns['code'] = ['all'] + list(set(items))
        if items[0] in ['CE', 'BER']:
            dropdowns['loss'] = ['all'] + list(set(items))
        if items[0] in ['Adam', 'RMSprop']:
            dropdowns['optimizer'] = ['all'] + list(set(items))
        if items[0] in ['adaBP', 'plain', 'simple', 'full']:
            dropdowns['mode'] = ['all'] + list(set(items))
        if items[0] == 'mix':
            dropdowns['mixing'] = ['all'] + list(set(items))
        if items[0] == 'damp':
            dropdowns['damping'] = ['all'] + list(set(items))
        if 'lr' in items[0]:
            dropdowns['learning_rate'] = ['all'] + list(set(items))
        if 'disc' in items[0]:
            dropdowns['discount'] = ['all'] + list(set(items))
        if 'AWGN' in items[0] or 'BSC' in items[0]:
            dropdowns['channel'] = ['all'] + list(set(items))
    return dropdowns


def create_plot(result, filter_func, rename_func, metric, filename='default'):
    # filter and rename models
    result = {rename_func(k): v[['SNR', metric]]
              for k, v in result.items()
              if filter_func(k) and metric in v.columns}
    dropdowns = generate_dropdown(result.keys())
    all_data = [go.Scatter(x=list(v['SNR']), y=list(v[metric]), name=k)
                for k, v in result.items()]
    layout = dict(
        title=metric,
        showlegend=True,
        yaxis=dict(
            type='log',
            autorange=True
        )
    )

    def draw_plot(**kwargs):
        args = [x for x in kwargs.values() if x != 'all']
        data = [d for d in all_data if all(x in d.name for x in args)]
        fig = dict(data=data, layout=layout)
        off.iplot(fig, filename=filename)
    return interact(draw_plot, **dropdowns)
