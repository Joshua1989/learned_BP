import pickle
import re
import glob
from collections import defaultdict
import plotly.offline as off
import plotly.graph_objs as go
from ipywidgets import interact
from pandas import DataFrame
off.init_notebook_mode(connected=True)


def export_results(exp_name, batch_size=100, overwrite=True):
    B = batch_size
    result = defaultdict(lambda: defaultdict(list))
    pickle_files = glob.glob(f'{exp_name}/test_results/*.pickle')
    for file in pickle_files:
        match = re.search('_(\d+)_(\d+).pickle', file)
        N = int(match.group(1))
        for k, v in sorted(pickle.load(open(file, 'rb')).items()):
            if k in result:
                if overwrite:
                    result[k] = defaultdict(list)
                else:
                    continue
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
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = [v]

    def f(key):
        if 'channel' in d and all(x not in key for x in d['channel']):
            return False
        if 'T' in d and all(x not in key for x in d['T']):
            return False
        if 'mode' in d and all(x.replace('rrd_', '') not in key or (('rrd' in key) ^ ('rrd' in x)) for x in d['mode']):
            return False
        if 'loss' in d and all(x not in key for x in d['loss']):
            return False
        if 'optimizer' in d and all(x not in key for x in d['optimizer']):
            return False
        if 'code' in d and all(x + ',' not in key for x in d['code']):
            return False
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
            mode = re.search('mode=(\w+),', key).group(1)
            if 'rrd' in key:
                mode = 'rrd_' + mode
            segs.append(mode)
        if 'rrd' in key:
            if 'mixing_fixed=0.9999' not in key and 'no-mixing' not in key:
                if 'mixing_fixed' in key:
                    segs.append('mix' + re.search('mixing_fixed=([-+]?[0-9]*\.?[0-9]+)', key).group(1))
                else:
                    segs.append('mix')
            else:
                segs.append('no_mix')
        if 'damping_fixed=0.9999' not in key and 'no-damping' not in key:
            if 'damping_fixed' in key:
                segs.append('damp' + re.search('damping_fixed=([-+]?[0-9]*\.?[0-9]+)', key).group(1))
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
            if 'r_decay' in key:
                segs.append('lr' + re.search('r_decay_factor=([-+]?[0-9]*\.?[0-9]+)', key).group(1))
            if 'discount_decay' in key:
                segs.append('disc' + re.search('discount_decay_factor=([-+]?[0-9]*\.?[0-9]+)', key).group(1))
        if show_channel:
            segs.append(key.split(',')[1])
        return '-'.join(segs)
    return f


def generate_dropdown(keys):
    dropdowns = {}
    for items in zip(*[k.split('-') for k in keys]):
        if len(set(items)) == 1:
            continue
        if 'BCH_' in items[0] or 'RM_' in items[0]:
            dropdowns['code'] = ['all'] + list(set(items))
        if items[0] in ['CE', 'BER']:
            dropdowns['loss'] = ['all'] + list(set(items))
        if items[0] in ['Adam', 'RMSprop']:
            dropdowns['optimizer'] = ['all'] + list(set(items))
        if items[0] in ['adaBP', 'plain', 'simple', 'full', 'rrd_adaBP', 'rrd_plain', 'rrd_simple', 'rrd_full']:
            dropdowns['mode'] = ['all'] + list(set(items))
        if 'mix' in items[0]:
            dropdowns['mixing'] = ['all'] + list(set(items))
        if 'damp' in items[0]:
            dropdowns['damping'] = ['all'] + list(set(items))
        if 'lr' in items[0]:
            dropdowns['learning_rate'] = ['all'] + list(set(items))
        if 'disc' in items[0]:
            dropdowns['discount'] = ['all'] + list(set(items))
        if 'AWGN' in items[0] or 'BSC' in items[0]:
            dropdowns['channel'] = ['all'] + list(set(items))
    return dropdowns


def create_plot(result, filter_func, rename_func, metric, filename='default'):
    tickvals, ticktext = [1], ['10^0']
    for L in range(1, 9):
        tickvals += reversed([k / 10**L for k in range(1, 10)])
        ticktext += [''] * 9 + ['$10^{#}$'.replace('#', str(-L))]

    # filter and rename models
    result = {rename_func(k): v[['SNR', metric]]
              for k, v in result.items()
              if filter_func(k) and metric in v.columns}
    dropdowns = generate_dropdown(result.keys())
    all_data = [go.Scatter(x=list(v['SNR']), y=list(v[metric]), name=k)
                for k, v in result.items()]
    layout = dict(
        title=metric if filename == 'default' else filename,
        showlegend=True,
        xaxis=dict(showline=True, title='Eb/No (dB)'),
        yaxis=dict(
            showline=True,
            title='metric',
            type='log',
            autorange=True,
            tickvals=tickvals,
            ticktext=ticktext,
            ticks='inside',
            hoverformat='.4e',
        )
    )

    def draw_plot(**kwargs):
        args = [x for x in kwargs.values() if x != 'all']
        data = [d for d in all_data if all(x in d.name.split('-') for x in args)]
        fig = dict(data=data, layout=layout)
        off.iplot(fig, filename=filename)
    return interact(draw_plot, **dropdowns)


def create_convergence_plot(result, filter_func, rename_func, SNR, metric='BER', filename='default'):
    tickvals, ticktext = [1], ['10^0']
    for L in range(1, 9):
        tickvals += reversed([k / 10**L for k in range(1, 10)])
        ticktext += [''] * 9 + ['$10^{#}$'.replace('#', str(-L))]

    # filter and rename models
    result = {rename_func(k): v[v.SNR == SNR][[col for col in v.columns if metric in col]].values[0]
              for k, v in result.items() if filter_func(k)}
    dropdowns = generate_dropdown(result.keys())
    all_data = [go.Scatter(x=list(range(1, v.size + 1)), y=list(v), name=k)
                for k, v in result.items()]
    layout = dict(
        title=f'{metric} under {SNR}dB' if filename == 'default' else filename,
        showlegend=True,
        xaxis=dict(showline=True, title='# BP Iteration'),
        yaxis=dict(
            showline=True,
            title='metric',
            type='log',
            autorange=True,
            tickvals=tickvals,
            ticktext=ticktext,
            ticks='inside',
            hoverformat='.4e',
        )
    )

    def draw_plot(**kwargs):
        args = [x for x in kwargs.values() if x != 'all']
        data = [d for d in all_data if all(x in d.name.split('-') for x in args)]
        fig = dict(data=data, layout=layout)
        off.iplot(fig, filename=filename)
    return interact(draw_plot, **dropdowns)
