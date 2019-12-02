import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

#########################################################################################################
# Data gathering and plotting
#########################################################################################################
def gather_simul_data(file_name):
    simul_data = {}
    with open(file_name) as fp:
        res = json.loads(fp.read())
        for name, data in res.items():
            simul_data[name] = { 'BER': [], 'WER': [] }
            sd, chn_param, layer_index = simul_data[name], [], None
            for key, detail in sorted(data.items()):
                df = pd.DataFrame(detail)
                if key != 'layer_index':
                    chn_param.append(float(key))
                    sd['BER'].append(df['BER'])
                    sd['WER'].append(df['WER'])
                else:
                    layer_index = detail
            sd['BER'] = pd.concat(sd['BER'], axis=1)
            sd['BER'].columns, sd['BER'].index = chn_param, layer_index
            sd['WER'] = pd.concat(sd['WER'], axis=1)
            sd['WER'].columns, sd['WER'].index = chn_param, layer_index
    return simul_data

def plot_simul_data(result, figsize=(14,14), metric='WER', chn_param=None, layer_index=None, 
                    style_func=lambda x: {'label': x.replace('_','-')}, exclude_func=lambda x: False, tikz_only=False):
    tikz_output = {}
    if chn_param is not None and layer_index is None:
        # Plot the metric vs. iteration under specific channel parameter
        plt.figure(figsize=figsize)
        for curve, data in sorted(result.items()):
            if exclude_func(curve):
                continue
            try:
                x = np.arange(len(data[metric])) + 1
                y = data[metric][chn_param]
                tikz_output[curve] = ' '.join([str(p) for p in zip(x,y)])
                if not tikz_only:
                    plt.semilogy(x, y, **style_func(curve))
                    plt.xticks(x, data[metric].index, rotation='vertical')
                    plt.margins(0.2)
                    plt.subplots_adjust(bottom=0.15)
                    plt.grid(True, which="both")
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
                    plt.xlabel('iteration')
                    plt.ylabel(metric)
            except:
                print('No result of channel parameter {0} for curve {1}'.format(chn_param, curve))
    elif chn_param is None and layer_index is not None:
        # Plot the metric vs. channel parameter after specific BP iteration
        plt.figure(figsize=figsize)
        for curve, data in sorted(result.items()):
            if exclude_func(curve):
                continue
            try:
                x = data[metric].columns
                y = data[metric].iloc[layer_index-1]
                tikz_output[curve] = ' '.join([str(p) for p in zip(x,y)])
                if not tikz_only:
                    plt.semilogy(x, y, **style_func(curve))
                    plt.grid(True, which="both")
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
                    plt.xlabel('channel parameter')
                    plt.ylabel(metric)
            except:
                print('No result of layer index {0} for curve {1}'.format(chn_param, curve))
    else:
        raise Exception('Exact one of channel parameter and layer index should exist!')
    return tikz_output