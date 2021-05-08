from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import os
from pylab import *
from math import ceil, floor
from tqdm import tqdm
from experiment import *
matplotlib.use("Agg")

COLOR_MAP = plt.cm.jet
colormap = {'DT': 'FireBrick',
            'EDT': 'ForestGreen',
            'TDT': 'RoyalBlue',
            'PO': 'SlateGray',
            'CS': 'DarkGoldenRod',
            'ST': 'Indigo'}
hatchmap = {'DT': None,
            'EDT': '//',
            'TDT': 'xx',
            'PO': '.',
            'CS': '\\',
            'ST': '--'}
markermap = {'DT': '^',
             'EDT': 'v',
             'TDT': 'o'}
linestylemap = {'DT': '-.',
                'EDT': '--',
                'TDT': '-'}

dashmap = { 'DT': [6,2],
                'EDT': [8,1,1,1],
                'TDT': [1,0],
                'ST':[1,1],
                'CS':[2,1]}


def timePlot(ax, time, data, label, linestyle='-'):

    markermap = ['P', 'o', 'X']
    # markevery = [(2500, 10000), (7500, 10000), (0, 10, 1)]
    markevery = [(1000, 4000), (2000, 4000), (2200, 4000)]
    ax.plot(time, data, label=label, linewidth=4, linestyle=linestyle)
    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    colors = ['firebrick', 'limegreen', 'royalblue']
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])
        j.set_marker(markermap[i])
        j.set_markevery(markevery[i])
        j.set_markersize(10)
    ax.legend()


def bufferPlot(ax, data, ports):
    bufferdata = data[np.where(data[:, 0] == 0)]
    bufferdata = bufferdata[:, 1:]

    # num = int((endTime - startTime) / interval)
    portmap = {0: 1, 1: 2, 15: 3}
    for port in ports:
        portdata = bufferdata[np.where(bufferdata[:, 0] == port)]
        timePlot(ax, portdata[:, 1]-1, portdata[:, 2],
                 'port ' + str(portmap[port]))

    #     data = np.zeros((2, num), dtype=float)
    #     for i in range(num):
    #         data[0, i] = startTime + i * interval
    #         index = (portdata[:, 1] >= startTime + i*interval) & (
    #             portdata[:, 1] < startTime + (i+1)*interval)
    #         temp = portdata[index]
    #         data[1, i] = temp[:, 2].mean()
    #     print(data)


def burst_absorbing():
    data_dt = np.loadtxt('data/results/determine/plot/microburst_1000us_DT')
    data_edt = np.loadtxt('data/results/determine/plot/microburst_1000us_EDT')
    data_tdt = np.loadtxt('data/results/determine/plot/microburst_1000us_TDT')

    data_dt = np.loadtxt('data/results/burst_length/plot/microburst_1000us_DT')
    data_edt = np.loadtxt(
        'data/results/burst_length/plot/microburst_1000us_EDT')
    data_tdt = np.loadtxt(
        'data/results/burst_length/plot/microburst_1000us_TDT')

    # fig = plt.figure(figsize=(20, 4))
    # plt.subplots_adjust(wspace=0.4, hspace=0.2)
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)

    # bufferPlot(ax1, data_dt, [0, 1, 15])
    # bufferPlot(ax2, data_edt, [0, 1, 15])
    # bufferPlot(ax3, data_tdt, [0, 1, 15])
    # plt.sca(ax1)
    # plt.legend(fontsize=18)
    # plt.ylim(-50, 600)
    # plt.sca(ax2)
    # plt.legend(fontsize=18)
    # plt.ylim(-50, 600)
    # plt.sca(ax3)
    # plt.legend(fontsize=18)
    # plt.ylim(-50, 600)
    # plt.savefig('data/plot_result/DT_vs_EDT_vs_TDT.png', dpi=300)

    fig = plt.figure(figsize=(6, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, left=0.2)

    ax1 = fig.add_subplot(111, frameon=False)
    bufferPlot(ax1, data_dt, [0, 1, 15])
    plt.ylim(-50, 667)
    plt.tick_params(labelsize=24)
    plt.yticks(np.linspace(0, 667, 3), [0, 'B/2', 'B'])
    plt.xticks(np.linspace(0, 0.2, 5))
    plt.grid(linestyle='-.')
    plt.legend(fontsize=24, loc='center',
               frameon=False, bbox_to_anchor=(0.4, 0.7))
    plt.ylabel('Queue Length', fontsize=24)
    plt.xlabel('Time(sec)', fontsize=24)
    plt.savefig('data/plot_result/burst_absorb_DT.pdf', dpi=300)

    plt.clf()
    ax1 = fig.add_subplot(111, frameon=False)
    bufferPlot(ax1, data_tdt, [0, 1, 15])
    ax1.annotate('evacuation', xy=(0.015, 222), xytext=(0.05, 150), size=20,
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.0))
    ax1.annotate('absorption', xy=(0.15, 600), xytext=(0.16, 700), size=20,
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.0))
    plt.ylim(-50, 667)
    plt.yticks(np.linspace(0, 667, 3), [0, 'B/2', 'B'])
    plt.xticks(np.linspace(0, 0.2, 5))
    plt.grid(linestyle='-.')
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=24, loc='center',
               frameon=False, bbox_to_anchor=(0.4, 0.7))
    plt.ylabel('Queue Length', fontsize=24)
    plt.xlabel('Time(sec)', fontsize=24)
    plt.savefig('data/plot_result/burst_absorb_TDT.pdf', dpi=300)

    plt.clf()
    ax1 = fig.add_subplot(111, frameon=False)
    bufferPlot(ax1, data_edt, [0, 1, 15])
    plt.ylim(-50, 667)
    plt.tick_params(labelsize=24)
    plt.yticks(np.linspace(0, 667, 3), [0, 'B/2', 'B'])
    plt.xticks(np.linspace(0, 0.2, 5))
    plt.grid(linestyle='-.')
    plt.legend(fontsize=24, loc='center',
               frameon=False, bbox_to_anchor=(0.4, 0.7))
    plt.ylabel('Queue Length', fontsize=24)
    plt.xlabel('Time(sec)', fontsize=24)
    plt.savefig('data/plot_result/burst_absorb_EDT.pdf', dpi=300)


def data_postprocessing(data):
    result = []
    data = data[data[:, 0] == 1]
    data = data[:, 1:]
    data = data[data[:, 0] >= 8]
    for i in range(data.shape[0]):
        if data[i, 3]/(data[i, 2]-data[i, 1]) > 200000:
            # result.append([data[i, 2]-data[i, 1], data[i, 3]-data[i, 4]])
            duration = data[i, 3] * 2.7939677238464355e-06 / 2
            result.append([duration, data[i, 3]-data[i, 4]])

    return np.array(result)


def loss_ratio_vs_burst_length():
    algorithms = ['DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/stochastic/30%_250us_2/loss/'
    all_burst = {'DT': np.array([]), 'EDT': np.array([]), 'TDT': np.array([])}
    lossless_burst = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([])}
    for i in range(50):
        for algorithm in algorithms:
            data = np.loadtxt(lossSaveDir + 'trace' +
                              str(i) + '_' + algorithm)
            result = data_postprocessing(data)
            all_burst[algorithm] = np.hstack(
                (all_burst[algorithm], result[:, 0]))
            lossless = result[result[:, 1] == 0]
            lossless_burst[algorithm] = np.hstack(
                (lossless_burst[algorithm], lossless[:, 0]))

    burst, bin_edges = np.histogram(
        all_burst['DT'], bins=4, range=(0, 0.001))
    burst[-1] += np.sum(all_burst['DT'] > 0.001)
    print(np.sum(burst))

    lossless = {}
    for algorithm in algorithms:
        lossless[algorithm], burst_absorbing = np.histogram(
            lossless_burst[algorithm], bins=4, range=(0, 0.001))
        print(np.sum(lossless[algorithm]))
        print(np.sum(lossless[algorithm])/np.sum(burst))
        lossless[algorithm] = lossless[algorithm] / burst

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, left=0.15, right=0.95)
    xlabels = ['[0,0.25)', '[0.25,0.5)', '[0.5,0.75)',
               '[0.75,+$\infty$)', r'[2,+$\infty$)', ]
    x = np.arange(4)
    total_width, n = 0.75, 3
    width = total_width / n
    for i, algorithm in enumerate(algorithms):
        ax.bar(x+width*i, lossless[algorithm], width=0.75*width,
               label=algorithm, color='w', edgecolor=colormap[algorithm],
               lw=3, hatch=hatchmap[algorithm])

    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='upper right',
               bbox_to_anchor=(1.1, 1.1), frameon=False)
    plt.tick_params(labelsize=28)
    plt.xticks(x+total_width/2-width/2, xlabels)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('Lossless Ratio', fontsize=32)
    plt.xlabel('Burst Duration(ms)', fontsize=32)
    plt.savefig('data/plot_result/loss-sensitive2.pdf')


def loss_vs_burst_length():

    def to_percent(temp, position):
        return '%.0f' % (100 * temp) + '%'

    algorithms = ['DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/determine/250-2000/loss/'

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, left=0.2)

    for algorithm in algorithms:
        result = []
        for i in range(250, 2250, 250):
            data = np.loadtxt(lossSaveDir + 'microburst_' +
                              str(i) + 'us_'+algorithm)
            result.append([i, data[1, -1]/data[:, -1].sum()])
        result = np.array(result)
        ax.plot(result[:, 0]/1000, result[:, 1], label=algorithm, color=colormap[algorithm],
                linewidth=4, marker=markermap[algorithm], markersize=15)

    plt.ylim(-0.1, 1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.grid(linestyle='-.')
    plt.legend(fontsize=28, loc='best', ncol=3, frameon=False)
    plt.tick_params(labelsize=28)
    plt.ylabel('Packet Loss Rate(%)', fontsize=28)
    plt.xticks(np.linspace(0.25, 2.0, 8))
    plt.xlabel('Micro-burst Duration(ms)', fontsize=28)
    plt.savefig('data/plot_result/loss_vs_burst_length.png')


def stochastic_fairness_plot():
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, left=0.25, top=0.9, right=0.95)
    algorithms = ['DT', 'EDT', 'TDT', 'ST', 'CS']
    algorithms = ['CS', 'ST', 'DT', 'EDT', 'TDT']

    result = stochastic_fairness()
    df = pd.DataFrame(result.T, columns=algorithms)
    color = dict(boxes='DarkBlue', whiskers='Gray',
                 medians='Red', caps='DarkBlue')

    for i, algorithm in enumerate(algorithms):
            ax.bar(i + 1, df[algorithm].mean(), width=0.75,
                   label=algorithm, color='w', edgecolor=colormap[algorithm],
                   lw=3, hatch=hatchmap[algorithm])
    xlabels =  algorithms = ['CS', 'ES', 'DT', 'EDT', 'TDT']

    plt.xticks(np.linspace(1, 5, 5), xlabels)

    plt.ylim(0.6, 1.01)
    plt.yticks(np.linspace(0.6, 1.0, 5))
    plt.grid(linestyle='-.')
    plt.ylabel('Fairness Index', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.savefig('data/plot_result/stochastic_fairness.pdf')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, left=0.3, right=0.95)
    algorithms = ['DT', 'EDT', 'TDT']
    plt.boxplot([(1 - df[algo]) * 1e4 for algo in algorithms],
                labels=algorithms,
                sym='o',
                boxprops={'color': 'RoyalBlue', 'linewidth': 3},
                medianprops={'color': 'FireBrick', 'linewidth': 3},
                capprops={'color': 'RoyalBlue', 'linewidth': 3},
                whiskerprops={'color': 'Gray', 'linewidth': 3},
                flierprops={'markersize': 12})


    ax.invert_yaxis()
    plt.grid(linestyle='-.')
    plt.ylabel('Fairness \nAttenuation ($10^{-4}$)', fontsize=32,) 
    plt.yticks([0,5,10])
    # plt.yticks([0.9994,0.9996,0.9998,1])
    plt.tick_params(labelsize=32)
    plt.savefig('data/plot_result/stochastic_fairness2.pdf')


def cdfPlot(data, ax, label, linestyle='-'):

    num_bins = 100
    data_mean = np.mean(data)
    label = label + ": " + str(format(data_mean, '.4f'))
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[:-1], cdf/len(data), label=label,
            linewidth=4, linestyle=linestyle)
    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])
    ax.legend(fontsize=16)


def tcp_fct_cdf():

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.spines['bottom'].set_color('gray')
    ax1.spines['left'].set_color('gray')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.spines['bottom'].set_color('gray')
    ax2.spines['left'].set_color('gray')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2)
    algorithms = ['EDT', 'TDT', 'DT']
    fct, p99, fct_mice = get_fct()
    for algorithm in algorithms:
        fct[algorithm] = np.array(fct[algorithm]) / np.array(fct['DT'])
        fct_mice[algorithm] = np.array(
            fct_mice[algorithm]) / np.array(fct_mice['DT'])
        p99[algorithm] = np.array(
            p99[algorithm]) / np.array(p99['DT'])
    for algorithm in ['DT', 'EDT', 'TDT']:
        ax1.plot(fct_mice[algorithm], color=colormap[algorithm],
                 linewidth=4, marker=markermap[algorithm], label=algorithm)
        ax2.plot(fct[algorithm], color=colormap[algorithm],
                 linewidth=4, marker=markermap[algorithm], label=algorithm)

    plt.sca(ax1)
    plt.legend(fontsize=20, loc='best', frameon=False)
    plt.ylim(0., 1.2)
    plt.ylabel('Normalized FCT', fontsize=20)
    plt.xlabel('Active Port(s)', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.yticks(np.linspace(0, 1.2, 5))
    plt.xticks(np.linspace(0, 4, 5), np.linspace(1, 5, 5, dtype=int))
    plt.grid(linestyle='-.')
    plt.sca(ax2)
    plt.legend(fontsize=20, loc='best', frameon=False)
    plt.ylim(0.4, 1.2)
    plt.ylabel('Normalized FCT', fontsize=20)
    plt.xlabel('Active Port(s)', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.xticks(np.linspace(0, 4, 5), np.linspace(1, 5, 5, dtype=int))
    plt.yticks(np.linspace(0.4, 1.2, 5))
    plt.grid(linestyle='-.')
    plt.savefig('data/plot_result/tcp_fct.png', dpi=300)


def DT_example():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, left=0.2)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    thres = [2, 2/3, 2/3, 2/3]
    queue = [0, 2/3, 2/3, 2/3]
    plt.plot(thres, color='firebrick', linewidth=4, label='Threshold')
    plt.plot(queue, color='royalblue', linewidth=4, label='Queue Length')
    plt.legend(fontsize=20, loc='best', frameon=False)
    plt.tick_params(labelsize=20)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Queue Length', fontsize=20)
    plt.xticks(np.linspace(0, 3, 4), [0, r'$t_1$', r'$t_2$', r'$t_2$'])
    plt.grid(linestyle='-.')
    plt.yticks([0, 1/3, 2/3, 1, 2],
               [0, r'$B/3$', r'$2B/3$', r'$B$', r'$2B$'])
    plt.savefig('data/plot_result/DT_example.png', dpi=300)


def queue_length_evolution():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, top=0.95, left=0.15, right=0.98)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    thres = [1, 5/6, 2/3, 2/3, 1, 1]
    queue1 = [1, 5/6, 2/3, 2/3, 1, 1]
    queue2 = [0, 1/3, 2/3, 2/3, 0, 0]
    plt.plot(thres, color='limegreen', linewidth=4,
             label=r'Threshold', linestyle='-', marker='o', markersize=10)
    plt.plot(queue1, color='firebrick', linewidth=4,
             label=r'$Q_{N}$', linestyle='--', marker='o', markersize=10)
    plt.plot(queue2, color='royalblue', linewidth=4,
             label=r'$Q_{M}$', linestyle='--', marker='o', markersize=10)
    plt.legend(fontsize=20, loc='upper center', ncol=2, frameon=False)
    plt.tick_params(labelsize=25)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel('Queue Length', fontsize=25)
    plt.xticks([0, 2, 3, 4, 5], [r'$t_0$', r'$t_1$', r'$t_1 + d_i$'])
    plt.grid(linestyle='-.')
    plt.yticks([0, 2/3, 1, 1.5],
               [0, r'$q_1$', r'$q_0$'])
    plt.savefig('data/plot_result/queue_length_evolution_1.pdf', dpi=300)


def queue_length_evolution_2():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, top=0.95, left=0.15, right=0.98)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    thres = [1, 1/3, 1/2, 2/3, 2/3, 1, 1]
    queue1 = [1, 8/9, 7/9, 2/3, 2/3, 1, 1]
    queue2 = [0, 1/3, 1/2, 2/3, 2/3, 0, 0]
    plt.plot(thres, color='limegreen', linewidth=4,
             label=r'Threshold', linestyle='-', marker='o')
    plt.plot(queue1, color='firebrick', linewidth=4,
             label=r'$Q_{N}$', linestyle='--', marker='o')
    plt.plot(queue2, color='royalblue', linewidth=4,
             label=r'$Q_{M}$', linestyle='--', marker='o')
    plt.legend(fontsize=20, loc='upper center', ncol=2, frameon=False)
    plt.tick_params(labelsize=25)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel('Queue Length', fontsize=25)
    plt.xticks([0, 1, 3, 4, 5, 6], [r'$t_0$', r'$t_2$', r'$t_3$', r'$t_3 + d_i$'])
    plt.grid(linestyle='-.')
    plt.yticks([0, 1/3, 2/3, 1, 1.5],
               [0, r'$q_2$', r'$q_1$', r'$q_0$'])
    plt.savefig('data/plot_result/queue_length_evolution_2.pdf', dpi=300)


def deterministic_fairness():
    algorithms = ['DT', 'EDT', 'TDT', 'OP']
    lossSaveDir = 'data/results/determine/loss/'

    result = {'DT': [], 'EDT': [], 'TDT': [], 'OP': []}
    for algorithm in algorithms:
        data = np.loadtxt(lossSaveDir + 'microburst_1000us_'+algorithm)
        data = data[data[:,0]==0]
        for port in [0, 1, 15]:
            result[algorithm].append(data[port, 3])

    for algorithm in algorithms:
        result[algorithm] = np.array(
            result[algorithm]) / np.array(result['OP'])

    algorithms = ['DT', 'EDT', 'TDT']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.15, left=0.2)
    xlabels = ['port 1', 'port 2', 'port 3']
    x = np.arange(3)
    total_width, n = 0.8, 4
    width = total_width / n
    for i, algorithm in enumerate(algorithms):
        ax.bar(x+width*i, result[algorithm], width=0.75*width,
               label=algorithm, color='w', edgecolor=colormap[algorithm],
               lw=3, hatch=hatchmap[algorithm])

    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.15), columnspacing=0.5)
    plt.tick_params(labelsize=32)
    plt.xticks(x+total_width/2-width/2, xlabels)
    plt.ylim(0,1.2)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.ylabel('Normalized Throughput', fontsize=32)
    plt.savefig('data/plot_result/deterministic_fairness.pdf')


def plot_delay_sensitive():
    lossSaveDir = 'data/results/determine/loss/'
    # algorithms = ['DT', 'EDT', 'TDT']
    algorithms = ['EDT', 'TDT', 'CS']
    result = {}
    for algorithm in algorithms:
        data = np.loadtxt(lossSaveDir + 'delay-sensitive_'+algorithm)
        data = data[data[:, 0] == 2]
        data = data[data[:, 1] == 1]
        data = data[data[:, 4] >= 0.001]
        result[algorithm] = [data[:, 4], data[:, 5]]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, left=0.3)

    for algorithm in algorithms:
        temp = np.cumsum(result[algorithm][1]) / \
            np.cumsum(result[algorithm][1])[-1]
        if algorithm == 'EDT':
            label = 'DT&EDT'
        elif algorithm == 'TDT':
            label = 'TDT&ES'
        else:
            label = algorithm
        ax.plot(result[algorithm][0] * 1000, temp, linewidth=5,
                label=label, color=colormap[algorithm])

    plt.legend(fontsize=32, loc='lower right',
               frameon=False, bbox_to_anchor=(1.15, 0.0))
    plt.grid(linestyle='-.')
    plt.tick_params(labelsize=32)
    plt.xlim(0., 10)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.xticks(np.linspace(0, 10, 6))

    plt.xlabel('Delay(ms)', fontsize=32)
    plt.ylabel('CDF', fontsize=32)

    bbox_props = dict(boxstyle="larrow", fc="none", ec="red", lw=4)
    t = ax.text(5, 0.8, "Better", ha="center", va="center", rotation=0,
                size=32,
                bbox=bbox_props)

    plt.savefig('data/plot_result/delay-sensitive.pdf', dpi=300)

    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, left=0.3)
    plotSaveDir = 'data/results/determine/plot/'

    algorithms = ['DT', 'EDT', 'TDT', 'CS', 'ST']

    queuelen = {'DT': np.zeros(668), 'EDT': np.zeros(668), 'TDT': np.zeros(
        668), 'ST': np.zeros(668), 'CS': np.zeros(668)}
    for algorithm in algorithms:
        data = np.loadtxt(plotSaveDir + 'delay-sensitive_'+algorithm)
        data = data[data[:, 0] == 0]
        data = data[data[:, 1] == 0]
        for i in data[:, 3]:
            queuelen[algorithm][int(i)] += 1

    for algorithm in algorithms:
        queuelen[algorithm] = np.cumsum(queuelen[algorithm])
        queuelen[algorithm] = queuelen[algorithm] / queuelen[algorithm][-1]

        label = algorithm
        if algorithm == 'ST':
            label = 'ES'
            ax.plot(np.linspace(0, 42, 43), queuelen[algorithm][:43], linewidth=5,
                    label=label, color=colormap[algorithm], linestyle='--')
        elif algorithm == 'DT':
            pass
        else:
            if algorithm == 'EDT':
                label = 'DT&EDT'
            ax.plot(np.linspace(0, 667, 668), queuelen[algorithm], linewidth=5,
                    label=label, color=colormap[algorithm])

    plt.xlabel('Queue length', fontsize=32)
    plt.ylabel('CDF', fontsize=32)
    plt.legend(fontsize=32, loc='right', frameon=False,
               bbox_to_anchor=(1.15, 0.5))
    plt.grid(linestyle='-.')
    plt.tick_params(labelsize=32)
    plt.xlim(0, 700)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    # plt.xticks(np.linspace(0, 667, 5), [
    #            '0', r'$0.25B$', r'$0.5B$', r'$0.75B$', r'$B$'])
    plt.xticks(np.linspace(0, 667, 3), [
               '0', r'$0.5B$', r'$B$'])

    plt.savefig('data/plot_result/delay-queue-cdf.pdf', dpi=300)


def buffer_utilization_vs_thropughput():

    def to_percent(temp, position):
        return '%.0f' % (100 * temp) + '%'

    fig = plt.figure(figsize=(8, 6))

    for uti in [100, 10]:
        ax1 = fig.add_subplot(111)
        ax1.spines['top'].set_visible(False)
        ax2 = ax1.twinx()
        ax2.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        plt.subplots_adjust(wspace=0.4, hspace=0.2,
                            bottom=0.2, left=0.2, right=0.8)
        data = np.loadtxt(
            'data/results/determine/plot/buffer_utilization_' + str(uti))
        data = data[data[:, 0] == 4]
        data = data[data[:, 3] == 1]
        data = data[data[:, 2] < 1.01]
        temp = data[:, 2]
        throughput, bins = np.histogram(temp, bins=101)
        throughput = throughput * 1500 * 8 * 10000 / 1024**3
        smooth_window = 10
        throughput = pd.Series(throughput).rolling(smooth_window,
                                                   min_periods=1).mean()
        ax1.plot((bins[1:] - 1)*1000, throughput,
                 linewidth=4,  color='RoyalBlue', label='Throughput')

        data = np.loadtxt(
            'data/results/determine/plot/buffer_utilization_' + str(uti))
        data = data[data[:, 0] == 0]
        data = data[data[:, 1] == 16]
        data = data[data[:, 2] < 1.01]
        time = (data[:, 2] - 1) * 1000
        queue_length = data[:, 3]/100
        queue_length = pd.Series(queue_length).rolling(smooth_window,
                                                       min_periods=1).mean()
        ax2.plot(time, queue_length, linewidth=4,
                 color='Crimson', label='Buffer utilization')

        plt.sca(ax1)
        plt.grid(linestyle='-.')
        plt.ylabel('Normalized Throughput', fontsize=28)
        plt.xlabel('Time(ms)', fontsize=28)
        plt.ylim(0, 2)
        plt.yticks(np.linspace(0, 2, 5))
        plt.xticks(np.linspace(0, 10, 6))
        plt.tick_params(labelsize=28)
        if uti == 10:
            plt.legend(fontsize=28, loc='center',
                       frameon=False, bbox_to_anchor=(0.5, 0.6))
        else:
            plt.legend(fontsize=28, loc='center',
                       frameon=False, bbox_to_anchor=(0.5, 0.3))

        plt.sca(ax2)
        plt.ylabel('Utilization(%)', fontsize=28, rotation=270)
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.yticks(np.linspace(0, 1, 5))
        plt.tick_params(labelsize=28)
        if uti == 10:
            plt.legend(fontsize=28, loc='center',
                       frameon=False, bbox_to_anchor=(0.5, 0.2))
        else:
            plt.legend(fontsize=28, loc='center',
                       frameon=False, bbox_to_anchor=(0.5, 0.8))

        plt.savefig('data/plot_result/utilization_' +
                    str(uti) + '.pdf', dpi=300)
        plt.clf()

    # data = np.loadtxt('data/results/determine/plot/buffer_utilization_100')
    # data = data[data[:,0]==4]
    # data = data[data[:,3]==1]
    # temp = data[:,2]
    # throughput, bins = np.histogram(temp,bins=101)
    # ax.plot(bins[1:], throughput)


def fair_vs_equal():

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, left=0.2, right=0.9)
    for name in ['fair', 'equal']:
        data = np.loadtxt(
            'data/results/determine/plot/fairness_' + name)
        data = data[data[:, 0] == 0]
        port1 = data[data[:, 1] == 0]
        port2 = data[data[:, 1] == 1]

        temp = port1[:, 3]
        smooth_window = 3
        temp = pd.Series(temp).rolling(smooth_window,
                                       min_periods=1).mean()
        if name == 'fair':
            ax1.plot(port1[:, 2]*1000, temp/100, linewidth=4,
                     color='FireBrick', label='Port A')

        temp = port2[:, 3]
        smooth_window = 3
        temp = pd.Series(temp).rolling(smooth_window,
                                       min_periods=1).mean()

        if name == 'fair':
            ax1.plot(port2[:, 2]*1000, temp/100, linewidth=4, linestyle='--',
                     color='DarkGreen', label='Port B (ideal)')
        else:
            ax1.plot(port2[:, 2]*1000, temp/100, linewidth=4,
                     color='DarkGreen', label='Port B')

        plt.ylim(0, 1.2)
        plt.grid(linestyle='-.')
        plt.ylabel('Queue Length', fontsize=28)
        plt.xlabel('Time(ms)', fontsize=28)
        plt.yticks([0, 0.5, 1.0], ['0', r'$0.5B$', r'$B$'])
        plt.xticks(np.linspace(0, 15, 4))
        plt.tick_params(labelsize=28)

        plt.legend(fontsize=28, loc='center', ncol=2, frameon=False,
                   bbox_to_anchor=(0.6, 1.0), columnspacing=-2.0)

    plt.savefig('data/plot_result/fairness_example' + '.pdf', dpi=300)
    plt.clf()

    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, left=0.2, right=0.9)
    data = np.loadtxt(
        'data/results/determine/loss/fairness_equal')
    data = data[data[:, 0] == 0]
    port1 = data[data[:, 1] == 0]
    port2 = data[data[:, 1] == 1]
    port1 = port1[0, 3] / 1145
    port2 = port2[0, 3] / 1145

    ax.bar(1.15, port1, width=0.2,
           label='Actual', color='w', edgecolor='LightSteelBlue',
           lw=3, hatch='--')
    ax.bar(1.85, port2, width=0.2, color='w', edgecolor='LightSteelBlue',
           lw=3, hatch='--')

    data = np.loadtxt(
        'data/results/determine/loss/fairness_fair')
    data = data[data[:, 0] == 0]
    port1 = data[data[:, 1] == 0]
    port2 = data[data[:, 1] == 1]
    port1 = port1[0, 3] / 1145
    port2 = port2[0, 3] / 1145

    ax.bar(0.85, port1, width=0.2,
           label='Ideal', color='w', edgecolor='LightGreen',
           lw=3, hatch='||')
    ax.bar(1.55, port2, width=0.2, color='w', edgecolor='LightGreen',
           lw=3, hatch='||')

    plt.grid(linestyle='-.')
    plt.legend(fontsize=28, loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    plt.ylim(0, 0.7)
    plt.ylabel('Throughput', fontsize=28)
    plt.ylabel('Throughput', fontsize=28)
    # plt.yticks([0, 0.5, 1.0], ['0', r'$0.5B$', r'$B$'])
    plt.xticks([1, 1.7], ['Port A', 'Port B'])
    plt.tick_params(labelsize=28)

    plt.savefig('data/plot_result/fairness_example2.pdf', dpi=300)


def stochastic_delay():
    algorithms = ['DT', 'EDT', 'TDT', 'CS', 'ST']
    lossSaveDir = 'data/results/stochastic/30%_250us_1/loss/'
    delay = {'DT': np.zeros((100, ), dtype=int),
             'EDT': np.zeros((100, ), dtype=int),
             'TDT': np.zeros((100, ), dtype=int),
             'CS': np.zeros((100, ), dtype=int),
             'ST': np.zeros((100, ), dtype=int), }

    for i in range(50):
        for algorithm in algorithms:
            data = np.loadtxt(lossSaveDir + 'trace' +
                              str(i) + '_' + algorithm)

            data = data[data[:, 0] == 2]
            data = data[data[:, 1] == 169]

            for j in range(data.shape[0]):
                index = int(data[j, 2])
                delay[algorithm][index] += int(data[j, 5])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, top=0.9, left=0.3, right=0.9)

    for algorithm in algorithms:
        delay[algorithm] = np.cumsum(delay[algorithm])
        delay[algorithm] = delay[algorithm]/delay[algorithm][-1]

        if algorithm == 'ST':
            x = np.linspace(1.1, 1.6, 6)
            ax.plot(x[1:], delay[algorithm][11:16], label='ES',
                    color=colormap[algorithm], linewidth=4, linestyle='--')
        elif algorithm == 'DT':
            x = np.linspace(1.1, 5.1, 41)
            ax.plot(x[1:], delay[algorithm][11:51], label=algorithm,
                    color=colormap[algorithm], linewidth=4)
        else:
            x = np.linspace(1.1, 9.1, 81)
            ax.plot(x[1:], delay[algorithm][11:91], label=algorithm,
                    color=colormap[algorithm], linewidth=4)

    bbox_props = dict(boxstyle="larrow", fc="none", ec="red", lw=4)
    t = ax.text(3.2, 0.75, "Better", ha="center", va="center", rotation=0,
                size=24,
                bbox=bbox_props)

    plt.ylabel('CDF', fontsize=32)
    plt.xlabel('Delay(ms)', fontsize=32)

    plt.yticks(np.linspace(0, 1, 5))
    plt.xticks(np.linspace(0, 10, 6))
    plt.tick_params(labelsize=32)
    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='center', ncol=1,
               frameon=False, bbox_to_anchor=(0.85, 0.5))
    fig.savefig('data/plot_result/stochastic_delay.pdf')


def stochastic_throughput():
    algorithms = ['DT', 'EDT', 'TDT', ]
    lossSaveDir = 'data/results/stochastic/30%_250us_1/loss/'
    throughput = {'DT': np.zeros((50, ), dtype=float),
                  'EDT': np.zeros((50, ), dtype=float),
                  'CS': np.zeros((50, ), dtype=float),
                  'ST': np.zeros((50, ), dtype=float),
                  'TDT': np.zeros((50, ), dtype=float),
                  'PO': np.zeros((50, ), dtype=float),
                  }

    for i in range(50):
        for algorithm in algorithms:
            data = np.loadtxt(lossSaveDir + 'trace' +
                              str(i) + '_' + algorithm)

            data = data[data[:, 0] == 3]
            data = data[data[:, 1] == 169]

            throughput[algorithm][i] = data[0, -1] / 1000 ** 3

    df = pd.DataFrame(throughput, columns=algorithms)
    color = dict(boxes='DarkBlue', whiskers='Gray',
                 medians='Red', caps='DarkBlue')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, left=0.35)
    # df.plot.box(color=color, linewidth=4)
    plt.boxplot([df[algo] for algo in algorithms],
                labels=algorithms,
                sym='o',
                boxprops={'color': 'RoyalBlue', 'linewidth': 3},
                medianprops={'color': 'FireBrick', 'linewidth': 3},
                capprops={'color': 'RoyalBlue', 'linewidth': 3},
                whiskerprops={'color': 'Gray', 'linewidth': 3},
                flierprops={'markersize': 12})
    plt.grid(linestyle='-.')
    plt.ylabel('Throughput(Gbps)', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.savefig('data/plot_result/throughput.pdf')


def line_speed_demonstration():

    def to_percent(temp, position):
        return '%.0f' % (100 * temp) + '%'

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.25, top=0.7, left=0.25, right=0.9)

    offered_load = []
    with open('data/trace/determine/purpose_of_buffer') as f:
        rows = f.readlines()
        for i, row in enumerate(rows):
            if i >= 2:
                data = row.split()
                start = float(data[3]) * 1000
                end = float(data[4]) * 1000
                offered_load.append([start, 0])
                offered_load.append([start, 2])
                offered_load.append([end, 2])
                offered_load.append([end, 0])

    offered_load = np.array(offered_load)
    ax.plot(offered_load[:, 0], offered_load[:, 1],
            linewidth=5, color='RoyalBlue', label='Input', linestyle='-')

    data = np.loadtxt(
        'data/results/determine/plot/purpose_of_buffer0')
    data = data[data[:, 0] == 4]
    data = data[data[:, 3] == 1]
    temp = data[:, 2]
    throughput, bins = np.histogram(temp, bins=np.linspace(0, 0.0138, 139))

    throughput = throughput * 1500 * 8 * 10000 / 1000**3
    smooth_window = 1
    throughput = pd.Series(throughput).rolling(smooth_window,
                                               min_periods=1).mean()
    ax.plot((bins[1:])*1000, throughput,
            linewidth=4,  color='Coral', label='Output(w/o buffer)', linestyle='-')

    data = np.loadtxt(
        'data/results/determine/plot/purpose_of_buffer1')
    data = data[data[:, 0] == 4]
    data = data[data[:, 3] == 1]
    temp = data[:, 2]
    throughput, bins = np.histogram(temp, bins=np.linspace(0, 0.0138, 139))

    throughput = throughput * 1500 * 8 * 10000 / 1000**3
    smooth_window = 20
    throughput = pd.Series(throughput).rolling(smooth_window,
                                               min_periods=1).mean()
    ax.plot((bins[1:])*1000, throughput,
            linewidth=4,  color='DarkCyan', label='Output(w/ buffer)', linestyle='-')

    data = np.loadtxt(
        'data/results/determine/plot/purpose_of_buffer2')
    data = data[data[:, 0] == 4]
    data = data[data[:, 3] == 1]
    temp = data[:, 2]
    throughput, bins = np.histogram(temp, bins=np.linspace(0, 0.0138, 139))

    throughput = throughput * 1500 * 8 * 10000 / 1000**3
    smooth_window = 2
    throughput = pd.Series(throughput).rolling(smooth_window,
                                               min_periods=1).mean()
    ax.plot((bins[1:])*1000, throughput,
            linewidth=4,  color='MediumPurple', label='Output(small buffer)', linestyle='-')

    plt.grid(linestyle='-.')
    plt.ylabel('Transmission rate', fontsize=28)
    plt.xlabel('Time(ms)', fontsize=28)
    plt.ylim(0, 2.1)
    plt.xlim(-0.2, 12)
    plt.yticks(np.linspace(0, 2, 3))
    plt.xticks(np.linspace(0, 12, 7))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.tick_params(labelsize=28)
    plt.legend(fontsize=24, loc='center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.3))

    plt.savefig('data/plot_result/purpose_of_buffer.png', dpi=300)

    # for uti in [100, 10]:
    #     ax1 = fig.add_subplot(111)
    #     ax1.spines['top'].set_visible(False)
    #     ax2 = ax1.twinx()
    #     ax2.spines['top'].set_visible(False)
    #     # ax.spines['right'].set_visible(False)
    #     plt.subplots_adjust(wspace=0.4, hspace=0.2,
    #                         bottom=0.2, left=0.2, right=0.8)
    #     data = np.loadtxt(
    #         'data/results/determine/plot/buffer_utilization_' + str(uti))
    #     data = data[data[:, 0] == 4]
    #     data = data[data[:, 3] == 1]
    #     data = data[data[:, 2] < 1.01]
    #     temp = data[:, 2]
    #     throughput, bins = np.histogram(temp, bins=101)
    #     throughput = throughput * 1500 * 8 * 10000 / 1024**3
    #     smooth_window = 10
    #     throughput = pd.Series(throughput).rolling(smooth_window,
    #                                                min_periods=1).mean()
    #     ax1.plot((bins[1:] - 1)*1000, throughput,
    #              linewidth=4,  color='RoyalBlue', label='Throughput')

    #     data = np.loadtxt(
    #         'data/results/determine/plot/buffer_utilization_' + str(uti))
    #     data = data[data[:, 0] == 0]
    #     data = data[data[:, 1] == 16]
    #     data = data[data[:, 2] < 1.01]
    #     time = (data[:, 2] - 1) * 1000
    #     queue_length = data[:, 3]/100
    #     queue_length = pd.Series(queue_length).rolling(smooth_window,
    #                                                    min_periods=1).mean()
    #     ax2.plot(time, queue_length, linewidth=4,
    #              color='Crimson', label='Buffer utilization')

    #     plt.sca(ax1)
    #     plt.grid(linestyle='-.')
    #     plt.ylabel('Normalized throughput', fontsize=28)
    #     plt.xlabel('Time(ms)', fontsize=28)
    #     plt.ylim(0, 2)
    #     plt.yticks(np.linspace(0, 2, 5))
    #     plt.xticks(np.linspace(0, 10, 6))
    #     plt.tick_params(labelsize=28)
    #     if uti == 10:
    #         plt.legend(fontsize=28, loc='center',
    #                    frameon=False, bbox_to_anchor=(0.5, 0.6))
    #     else:
    #         plt.legend(fontsize=28, loc='center',
    #                    frameon=False, bbox_to_anchor=(0.5, 0.3))

    #     plt.sca(ax2)
    #     plt.ylabel('Utilization(%)', fontsize=28, rotation=270)
    #     plt.ylim(0, 1)
    #     plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    #     plt.yticks(np.linspace(0, 1, 5))
    #     plt.tick_params(labelsize=28)
    #     if uti == 10:
    #         plt.legend(fontsize=28, loc='center',
    #                    frameon=False, bbox_to_anchor=(0.5, 0.2))
    #     else:
    #         plt.legend(fontsize=28, loc='center',
    #                    frameon=False, bbox_to_anchor=(0.5, 0.8))

    #     plt.savefig('data/plot_result/utilization_' +
    #                 str(uti) + '.pdf', dpi=300)
    #     plt.clf()


def queue_threshold():

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, top=0.8,
                        bottom=0.2, left=0.2, right=0.9)
    data = np.loadtxt(
        'data/results/determine/plot/queue+threshold')
    data = data[data[:, 0] == 0]
    queue = data[data[:, 1] == 0]
    queue[:, 3] = queue[:, 3] / 667
    queue = np.append(queue, [[0, 0, 0.04, 0]], axis=0)

    data = np.loadtxt(
        'data/results/determine/plot/queue+threshold')
    data = data[data[:, 0] == 4]
    threshold = data[data[:, 1] == 0]
    threshold[:, 3] = threshold[:, 3] / 667
    threshold = np.append(threshold, [[0, 0, 0.04, 1]], axis=0)

    ax.plot(queue[:, 2]*1000, queue[:, 3], color='CornFlowerBlue',
            linewidth=4, label='Queue length', marker='P', markevery=(0.1, 0.2), markersize=10)
    ax.plot(threshold[:, 2]*1000, threshold[:, 3],
            color='IndianRed', linewidth=4, label='Threshold', marker='X', markevery=0.5, markersize=10)

    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='upper center', ncol=1,
               frameon=False, bbox_to_anchor=(0.75, 0.9))

    plt.axvspan(0, 1.1, facecolor='LightGreen', alpha=0.3)
    plt.axvspan(1.1, 5, facecolor='LightSteelBlue', alpha=0.3)
    plt.axvspan(5, 22.5, facecolor='LightGreen', alpha=0.3)
    plt.axvspan(22.5, 30, facecolor='LightCoral', alpha=0.15)
    plt.axvspan(30, 40, facecolor='LightGreen', alpha=0.3)

    plt.ylim(-0.05, 1.05)
    plt.ylabel('Queue Length', fontsize=32)
    plt.xlabel('Time(ms)', fontsize=32)
    plt.yticks([0, 0.5, 1.0, ], ['0', r'$0.5B$', r'$B$', ])
    plt.xticks([0, 10, 20, 30, 40])
    plt.tick_params(labelsize=32)

    bbox_props = dict(boxstyle="round", fc="LightGreen",
                      ec="none", lw=4, alpha=0.3)
    t = ax.text(0, 1.25, "Normal", ha="center", va="center", rotation=0,
                size=32, bbox=bbox_props)

    bbox_props = dict(boxstyle="round", fc="LightSteelBlue",
                      ec="none", lw=4, alpha=0.3)
    t = ax.text(15.5, 1.25, "Absorption", ha="center", va="center", rotation=0,
                size=32, bbox=bbox_props)

    bbox_props = dict(boxstyle="round", fc="LightCoral",
                      ec="none", lw=4, alpha=0.3)
    t = ax.text(33.5, 1.25, "Evacuation", ha="center", va="center", rotation=0,
                size=32, bbox=bbox_props)

    plt.savefig('data/plot_result/queue_threshold.pdf', dpi=300)


def plot_DPDK():

    # inFile = 'data/results/DPDK/0618_3/trace'

    # fct = {'DT': np.array(
    #     []), 'EDT': np.array([]), 'TDT': np.array([])}
    # mice = {'DT': np.array(
    #     []), 'EDT': np.array([]), 'TDT': np.array([])}

    # for algorithm in ['DT', 'EDT', 'TDT']:
    #     for i in range(20):
    #         data = np.loadtxt(
    #             inFile + '_' + algorithm + '_' + str(i) + '.txt')
    #         for j in range(data.shape[0]):
    #             fct[algorithm] = np.append(fct[algorithm], data[j, 1])
    #             if(data[j, 2] < 100000):
    #                 mice[algorithm] = np.append(mice[algorithm], data[j, 1])

    #     print(algorithm, fct[algorithm].mean(), mice[algorithm].mean())
    #     print(algorithm, np.percentile(
    #         fct[algorithm], 99), np.percentile(mice[algorithm], 99))

    # for i in range(20):
    #     dt = np.loadtxt(
    #         'data/results/DPDK/0618_2/trace_DT_' + str(i) + '.txt')
    #     edt = np.loadtxt(
    #         'data/results/DPDK/0618_2/trace_EDT_' + str(i) + '.txt')
    #     tdt = np.loadtxt(
    #         'data/results/DPDK/0618_2/trace_TDT_' + str(i) + '.txt')
    #     print(i, dt[:, 1].mean(), edt[:, 1].mean(), tdt[:, 1].mean())

    fctDir = 'data/results/DPDK/0624_3'

    fct = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }
    mice = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }
    p99 = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }

    algorithms = ['ST', 'CS', 'DT', 'EDT', 'TDT']
    algorithms = ['TDT','EDT','DT']
    # algorithms = ['DT','CS','TDT', 'EDT']

    # for algorithm in algorithms:
    #     for i in range(1):
    #         data = np.loadtxt(
    #             fctDir + '/trace_' + algorithm + '_' + str(i) + '.txt')

    #         data = data[data[:, 2] < 50000000]
    #         print(data[:, 1].mean(), end=" ")
    #         # fct[algorithm] = np.append(fct[algorithm], data[:, 1].mean())
    #         # p99[algorithm] = np.append(
    #         #     p99[algorithm], np.percentile(data[:, 1], 99))
    #         # data = data[data[:, 2] < 100000]
    #         # mice[algorithm] = np.append(mice[algorithm], data[:, 1].mean())
    #         for j in range(data.shape[0]):
    #             fct[algorithm] = np.append(fct[algorithm], data[j, 1])
    #             if(data[j, 2] < 100000):
    #                 mice[algorithm] = np.append(mice[algorithm], data[j, 1])
    #     print('\n')

    #     print(algorithm, fct[algorithm].mean(), mice[algorithm].mean())
    #     print(algorithm, np.percentile(
    #         fct[algorithm], 99), np.percentile(mice[algorithm], 99))

    fct, mice, p99 = read_DPDK()
    for algorithm in algorithms:
        print(algorithm, fct[algorithm].mean(), mice[algorithm].mean(), p99[algorithm].mean())

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, top=0.9, left=0.2, right=0.95)
    for algorithm in algorithms:
        num_bins = 1000
        label = algorithm
        if algorithm == 'ST':
            label = 'ES'
        counts, bin_edges = np.histogram(fct[algorithm], bins=num_bins)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf/len(fct[algorithm]), label=label,
                linewidth=6, color=colormap[algorithm], dashes=dashmap[algorithm])
        ax.legend(fontsize=16)
    
    bbox_props = dict(boxstyle="larrow", fc="none", ec="red", lw=4)
    t = ax.text(0.5, 0.95, "Better", ha="center", va="center", rotation=0,
                size=32,
                bbox=bbox_props)

    plt.tick_params(labelsize=32)
    plt.ylabel('CDF', fontsize=32)
    plt.yticks(np.linspace(0,1.0,5))
    plt.xticks(np.linspace(0,0.3,6))
    plt.xlabel('FCT(s)', fontsize=32)
    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='upper right',
               bbox_to_anchor=(1.0, 0.8), frameon=False)
    plt.savefig('data/plot_result/tcp_fct.pdf', dpi=300)

    plt.clf()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, top=0.9, left=0.2, right=0.95)
    for algorithm in algorithms:
        num_bins = 1000
        label = algorithm
        counts, bin_edges = np.histogram(mice[algorithm], bins=num_bins)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf/len(mice[algorithm]), label=label,
                linewidth=4, color=colormap[algorithm], dashes=dashmap[algorithm])
        ax.legend(fontsize=16)
    
    bbox_props = dict(boxstyle="larrow", fc="none", ec="red", lw=4)
    t = ax.text(0.5, 0.95, "Better", ha="center", va="center", rotation=0,
                size=32,
                bbox=bbox_props)

    plt.tick_params(labelsize=32)
    plt.ylabel('CDF', fontsize=32)
    plt.yticks(np.linspace(0,1.0,5))
    plt.xticks(np.linspace(0,2.5,6))
    plt.xlabel('FCT(s)', fontsize=32)
    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='upper right',
               bbox_to_anchor=(1.0, 0.8), frameon=False)
    plt.savefig('data/plot_result/tcp_mice.png', dpi=300)

    plt.clf()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, top=0.9, left=0.2, right=0.95)
    for algorithm in algorithms:
        num_bins = 1000
        label = algorithm
        counts, bin_edges = np.histogram(p99[algorithm], bins=num_bins)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf/len(p99[algorithm]), label=label,
                linewidth=6, color=colormap[algorithm], dashes=dashmap[algorithm])
        ax.legend(fontsize=16)
    
    bbox_props = dict(boxstyle="larrow", fc="none", ec="red", lw=4)
    t = ax.text(0.6, 0.3, "Better", ha="center", va="center", rotation=0,
                size=32,
                bbox=bbox_props)

    plt.tick_params(labelsize=32)
    plt.ylabel('CDF', fontsize=32)
    plt.yticks(np.linspace(0,1.0,5))
    plt.xticks(np.linspace(0,1.5,6))
    plt.xlabel('FCT(s)', fontsize=32)
    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='upper right',
               bbox_to_anchor=(1.0, 0.8), frameon=False)
    plt.savefig('data/plot_result/tcp_p99.pdf', dpi=300)


def plot_tcp():
    fctDir = 'data/results/tcp/2_ports/loss'

    fct = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }
    mice = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }
    p99 = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }

    algorithms = ['ST', 'CS', 'DT', 'EDT', 'TDT']
    for algorithm in algorithms:
        for i in range(100):
            data = np.loadtxt(
                fctDir + '/trace' + str(i) + '_' + algorithm)

            fct[algorithm] = np.append(fct[algorithm], data[:, 1].mean())
            p99[algorithm] = np.append(
                p99[algorithm], np.percentile(data[:, 1], 99))
            data = data[data[:, 2] < 100000]
            mice[algorithm] = np.append(mice[algorithm], data[:, 1].mean())
            # for j in range(data.shape[0]):
            #     fct[algorithm] = np.append(fct[algorithm], data[j, 1])
            #     if(data[j, 2] < 100000):
            #         mice[algorithm] = np.append(mice[algorithm], data[j, 1])

        # print(algorithm, fct[algorithm].mean(), mice[algorithm].mean())
        # print(algorithm, np.percentile(
        #     fct[algorithm], 99), np.percentile(mice[algorithm], 99))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.25, top=0.7, left=0.25, right=0.9)

    for algorithm in algorithms:
        num_bins = 1000
        label = algorithm
        counts, bin_edges = np.histogram(fct[algorithm], bins=num_bins)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf/len(fct[algorithm]), label=label,
                linewidth=4, color=colormap[algorithm])
        ax.legend(fontsize=16)

    plt.savefig('data/plot_result/tcp_fct.png', dpi=300)

    plt.clf()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.25, top=0.7, left=0.25, right=0.9)
    for algorithm in algorithms:
        num_bins = 100
        label = algorithm
        counts, bin_edges = np.histogram(mice[algorithm], bins=num_bins)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf/len(mice[algorithm]), label=label,
                linewidth=4, color=colormap[algorithm])
        ax.legend(fontsize=16)

    plt.savefig('data/plot_result/tcp_mice.png', dpi=300)

    plt.clf()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.25, top=0.7, left=0.25, right=0.9)
    for algorithm in algorithms:
        num_bins = 100
        label = algorithm
        counts, bin_edges = np.histogram(p99[algorithm], bins=num_bins)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[:-1], cdf/len(p99[algorithm]), label=label,
                linewidth=4, color=colormap[algorithm])
        ax.legend(fontsize=16)

    plt.savefig('data/plot_result/tcp_p99.png', dpi=300)


def tcp_long_lived():

    # for i in range(3,5):
    #     data = np.loadtxt('data/results/tcp-test/plot_0.1_' + str(i))
    #     data = data[data[:,0] == 0]
    #     data = data[data[:,1] == 0]

    #     plt.plot(data[:,2],data[:,3])
    #     # plt.xlim(0,2)
    #     plt.savefig('data/results/tcp-test/result_0.1_' + str(i) + '.png')
    #     plt.clf()

    # os.system('./waf --run \"scratch/tcp --inFile=data/trace/determine/tcp-test --outFile=data/results/determine/loss/tcp-fct-TDT --plotFile=data/results/determine/plot/tcp-plot-TDT --lineRate=1Gbps --delay=1.5ms --algorithm=TDT --buffer=667\" > data/TDT.out ')
    data = np.loadtxt('data/results/determine/plot/tcp-plot-DT')
    data = data[data[:, 0] == 0]
    data = data[data[:, 1] == 0]

    plt.plot(data[:, 2], data[:, 3])
    plt.xlim(0, 2)
    plt.savefig('data/plot_result/tcp-test2.png')
    plt.clf()


def read_DPDK():

    fct = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }
    mice = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }
    p99 = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([]), 'ST': np.array([]), 'CS': np.array([]), }

    algorithms = ['DT', 'EDT', 'TDT', 'ST', 'CS']
    algorithms = ['DT', 'TDT', 'EDT']
    # algorithms = ['DT','CS','TDT', 'EDT']


    multi = 1000000
    fctDir = 'data/results/DPDK/0629_3/'
    # fctDir = 'data/results/DPDK/result_0629/'
    fctDir = 'data/results/DPDK/final_result/'
    for algorithm in algorithms:
        for trace in range(2,61):
            temp = []
            mice_temp = []
            inFile = fctDir + algorithm + '_' + str(trace) + '_2_flows.out'
            with open(inFile, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    fct_item = line.replace("Size:","")
                    fct_item = fct_item.replace(", Duration(usec)","")
                    fsize = fct_item.split(':')[0]
                    fct_item = fct_item.split(":")[1]
                    fct_item = float(fct_item) / multi
                    temp.append(fct_item)
                    if int(fsize) < 1000000:
                        mice_temp.append(fct_item)

            # if np.mean(temp) < 4:
            fct[algorithm] = np.append(fct[algorithm], np.mean(temp))
            # fct[algorithm] = np.append(fct[algorithm], temp)
            mice[algorithm] = np.append(mice[algorithm], np.mean(mice_temp))
            # mice[algorithm] = np.append(mice[algorithm], mice_temp)
            p99[algorithm] = np.append(p99[algorithm], np.percentile(temp,99))

    # fctDir = 'data/results/DPDK/0626_1/'
    # for algorithm in algorithms:
    #     for trace in range(10):
    #         temp = []
    #         mice_temp = []
    #         inFile = fctDir + algorithm + '_' + str(trace) + '_2_flows.out'
    #         with open(inFile, 'r', encoding='utf8') as f:
    #             for line in f.readlines():
    #                 fct_item = line.replace("Size:","")
    #                 fct_item = fct_item.replace(", Duration(usec)","")
    #                 fsize = fct_item.split(':')[0]
    #                 fct_item = fct_item.split(":")[1]
    #                 fct_item = float(fct_item) / multi
    #                 temp.append(fct_item)
    #                 if int(fsize) < 1000000:
    #                     mice_temp.append(fct_item)
    #         fct[algorithm] = np.append(fct[algorithm], np.mean(temp))
    #         # fct[algorithm] = np.append(fct[algorithm], temp)
    #         mice[algorithm] = np.append(mice[algorithm], np.mean(mice_temp))
    #         # mice[algorithm] = np.append(mice[algorithm], mice_temp)
    #         p99[algorithm] = np.append(p99[algorithm], np.percentile(temp,99))

    
        print(algorithm,fct[algorithm])
        print(algorithm,fct[algorithm].mean())
    print(fct["TDT"].mean() / fct['DT'].mean())
    print(fct["TDT"].mean() / fct['EDT'].mean())
        # print(algorithm,mice[algorithm])
        # print(algorithm, fct[algorithm].mean(), mice[algorithm].mean())
        # print(algorithm, np.percentile(
        #     fct[algorithm], 99), np.percentile(mice[algorithm], 99))
    return fct, mice, p99


def relative_gain():

    def to_percent(temp, position):
        return '%.0f' % (100 * temp) + '%'

    fct, mice, p99 = read_DPDK()
    TDT = (fct['DT'] - fct['TDT']) / fct['DT']
    EDT = (fct['DT'] - fct['EDT']) / fct['DT']
    

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, top=0.95, left=0.2, right=0.9)

    num_bins = 100
    counts, bin_edges = np.histogram(TDT, bins=num_bins)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[:-1], cdf/len(TDT), label='TDT',
            linewidth=4, color=colormap['TDT'], dashes=dashmap['TDT'])
    
    num_bins = 1000
    counts, bin_edges = np.histogram(EDT, bins=num_bins)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[:-1], cdf/len(EDT), label='EDT',
            linewidth=4, color=colormap['EDT'], dashes=dashmap['EDT'])
    
    plt.axvline(x=0,ls="-.",linewidth=2,c="black")

    bbox_props = dict(boxstyle="rarrow", fc="none", ec="red", lw=4)
    t = ax.text(0.3, 0.3, "Better", ha="center", va="center", rotation=0,
                size=32,
                bbox=bbox_props)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.tick_params(labelsize=28)
    plt.ylabel('CDF', fontsize=32)
    plt.yticks(np.linspace(0,1.0,5))
    plt.xticks([-0.25, 0, 0.25, 0.5])
    # plt.xticks(np.linspace(-0.3,0.5,9))
    plt.xlim(-0.4,0.6)
    plt.xlabel('Relative Gain (Over DT)', fontsize=32)
    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='upper right',
               bbox_to_anchor=(1.1, 0.8), frameon=False)
    
    plt.savefig('data/plot_result/relative_gain.pdf', dpi=300)


def DPDK_bar():

    algorithms = ['DT','EDT','TDT']
    fct, mice, p99 = read_DPDK()
    df = pd.DataFrame(fct, columns=algorithms)
    color = dict(boxes='DarkBlue', whiskers='Gray',
                 medians='Red', caps='DarkBlue')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2,
                        bottom=0.2, top=0.95, left=0.3, right=0.9)
    # df.plot.box(color=color, linewidth=4)
    plt.boxplot([df[algo] for algo in algorithms],
                labels=algorithms,
                sym='o',
                boxprops={'color': 'RoyalBlue', 'linewidth': 3},
                medianprops={'color': 'FireBrick', 'linewidth': 3},
                capprops={'color': 'RoyalBlue', 'linewidth': 3},
                whiskerprops={'color': 'Gray', 'linewidth': 3},
                flierprops={'markersize': 12},
                showfliers=False)
    plt.grid(linestyle='-.')
    plt.ylabel('FCT(s)', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.savefig('data/plot_result/fct_bar2.pdf')

if __name__ == "__main__":
    os.chdir(sys.path[0])
    os.chdir('../')
    # burst_absorbing()
    # loss_ratio_vs_burst_length()
    # loss_vs_burst_length()
    # stochastic_fairness_plot()
    # tcp_fct_cdf()
    # DT_example()
    queue_length_evolution()
    queue_length_evolution_2()
    # deterministic_fairness()
    # plot_delay_sensitive()
    # stochastic_delay()
    # stochastic_throughput()
    # line_speed_demonstration()

    # buffer_utilization_vs_thropughput()
    # fair_vs_equal()
    # queue_threshold()
    # plot_DPDK()
    # plot_tcp()
    # tcp_long_lived()

    # read_DPDK()
    # DPDK_bar()
    # relative_gain()