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
            'TDT': 'RoyalBlue'}
hatchmap = {'DT': None,
            'EDT': '//',
            'TDT': 'xx'}
markermap = {'DT': '^',
             'EDT': 'v',
             'TDT': 'o'}


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
    for port in ports:
        portdata = bufferdata[np.where(bufferdata[:, 0] == port)]
        timePlot(ax, portdata[:, 1]-1, portdata[:, 2],
                 'port[' + str(port) + ']')

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

    fig = plt.figure(figsize=(8, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2)
    ax1 = fig.add_subplot(111, frameon=False)
    bufferPlot(ax1, data_edt, [0, 1, 15])
    plt.ylim(-50, 667)
    plt.tick_params(labelsize=20)
    plt.grid(linestyle='-.')
    plt.legend(fontsize=20, loc='best', frameon=False)
    plt.ylabel('Queue Length(pkt)', fontsize=24)
    plt.xlabel('Time(sec)', fontsize=24)
    plt.savefig('data/plot_result/burst_absorb_EDT.png', dpi=300)


def data_postprocessing(data):
    result = []
    data = data[data[:, 0] >= 8]
    for i in range(data.shape[0]):
        result.append([data[i, 2]-data[i, 1], data[i, 3]-data[i, 4]])
    return np.array(result)


def loss_ratio_vs_burst_length():
    algorithms = ['DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/stochastic/4Gbps_50%_500us/loss/'
    all_burst = {'DT': np.array([]), 'EDT': np.array([]), 'TDT': np.array([])}
    lossless_burst = {'DT': np.array(
        []), 'EDT': np.array([]), 'TDT': np.array([])}
    for i in range(100):
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
        all_burst['DT'], bins=5, range=(0, 0.002))
    burst[-1] += np.sum(all_burst['DT'] > 0.002)

    lossless = {}
    for algorithm in algorithms:
        lossless[algorithm], burst_absorbing = np.histogram(
            lossless_burst[algorithm], bins=5, range=(0, 0.002))
        lossless[algorithm] = lossless[algorithm] / burst

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2)
    xlabels = ['[0,0.4)', '[0.4,0.8)', '[0.8,1.2)',
               '[1.2,1.6)', r'[1.6,+$\infty$)', ]
    x = np.arange(5)
    total_width, n = 0.75, 3
    width = total_width / n
    for i, algorithm in enumerate(algorithms):
        ax.bar(x+width*i, lossless[algorithm], width=0.75*width,
               label=algorithm, color='w', edgecolor=colormap[algorithm],
               lw=3, hatch=hatchmap[algorithm])

    plt.grid(linestyle='-.')
    plt.legend(fontsize=32, loc='best', frameon=False)
    plt.tick_params(labelsize=28)
    plt.xticks(x+total_width/2-width/2, xlabels)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('Lossless Ratio', fontsize=28)
    plt.xlabel('Micro-burst Duration(ms)', fontsize=28)
    plt.savefig('data/plot_result/burst_length.png')


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
        ax.plot(result[:, 0]/1000, result[:, 1], label=algorithm,
                linewidth=4, marker=markermap[algorithm], markersize=15)

    plt.ylim(-0.1, 1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.grid(linestyle='-.')
    plt.legend(fontsize=22, loc='best', ncol=3, frameon=False)
    plt.tick_params(labelsize=20)
    plt.ylabel('Packet Loss Rate(%)', fontsize=24)
    plt.xticks(np.linspace(0.25, 2.0, 8))
    plt.xlabel('Micro-burst Duration(ms)', fontsize=24)
    plt.savefig('data/plot_result/loss_vs_burst_length.png')


def stochastic_fairness_plot():
    result = stochastic_fairness()
    print(result)


if __name__ == "__main__":
    os.chdir(sys.path[0])
    os.chdir('../')
    # burst_absorbing()
    # loss_ratio_vs_burst_length()
    # loss_vs_burst_length()
    # stochastic_fairness_plot()
