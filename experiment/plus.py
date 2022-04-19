from multiprocessing.dummy import Pool as ThreadPool
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

colormap = {
    'DT': 'FireBrick',
    'EDT': 'ForestGreen',
    'TDT': 'RoyalBlue',
    'PO': 'SlateGray',
    'CS': 'DarkGoldenRod',
    'ST': 'Indigo',
    'TDT_2state':'BlueViolet'
}
hatchmap = {
    'DT': None,
    'EDT': '//',
    'TDT': 'xx',
    'PO': '.',
    'CS': '\\',
    'ST': '--',
    'TDT_2state':'x'

}
markermap = {'DT': '^', 'EDT': 'v', 'TDT': 'o'}
linestylemap = {'DT': '-.', 'EDT': '--', 'TDT': '-'}


def run_baseline_parallel(inFile, outFile, buffer, lineRate, delay, plotFile,
                          seed, alpha, dcThres):

    filename = 'TDT_param'

    cmdLine = './waf --run-no-build \"scratch/' + filename + \
        ' --dropCounter=' + str(dcThres) + \
        ' --buffer=' + str(buffer) + \
        ' --lineRate=' + lineRate + \
        ' --plotFile=' + plotFile + \
        ' --delay=' + delay + \
        " --inFile=" + inFile + \
        " --outFile=" + outFile + \
        " --alpha=" + str(alpha) + \
        " --simSeed=" + str(seed) + '\"'
    print(cmdLine)
    os.system(cmdLine)


def run_baseline_wrap(args):
    return run_baseline_parallel(*args)


def run_baseline(traceDir,
                 lossSaveDir,
                 plotDir,
                 algorithms,
                 inFiles,
                 buffer,
                 lineRate,
                 delay,
                 dcThres,
                 seed=1,
                 alpha=1.0):

    os.system('./waf')
    args = []
    for algorithm in algorithms:
        for dt in dcThres:
            for f in inFiles:
                inFile = traceDir + f
                outFile = lossSaveDir + f + '_' + algorithm + '_' + str(dt)
                plotFile = plotDir + f + '_' + algorithm

                args.append((inFile, outFile, buffer, lineRate, delay,
                             plotFile, seed, alpha, dt))

    # pool = ThreadPool(8)
    # pool.map(run_baseline_wrap, args)
    # pool.close()
    # pool.join()

    with ThreadPool(10) as pool:
        res = list(
            tqdm(pool.imap(run_baseline_wrap, args),
                 total=len(args),
                 desc='current progress: '))
    pool.close()
    pool.join()


def run_baselines():
    config = '40%_10G_0620'
    traceDir = 'data/trace/stochastic/' + config + '/'
    config = 'TDT_0620'
    saveDir = 'data/results/stochastic/' + config
    lossSaveDir = 'data/results/stochastic/' + config + '/loss/'
    plotDir = 'data/results/stochastic/' + config + '/plot/'

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    if not os.path.exists(lossSaveDir):
        os.makedirs(lossSaveDir)
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    inFiles = []
    for i in range(50):
        inFiles.append('trace' + str(i))
    algorithms = ['TDT']
    buffer = 1333
    buffer = 2000
    lineRate = '10Gbps'
    delay = '1ms'
    seed = 1
    alpha = 1.0
    dcThres = np.arange(100, 2100, 100)
    # dcThres = [10,20,500,100]

    run_baseline(traceDir, lossSaveDir, plotDir, algorithms, inFiles, buffer,
                 lineRate, delay, dcThres, seed, alpha)


def data_postprocessing(data):
    burst, long = [], []
    real_sum = 0
    ideal_sum = 0
    data = data[data[:, 0] == 1]
    data = data[:, 1:]
    data = data[data[:, 0] >= 8]
    for i in range(data.shape[0]):
        if data[i, 3] / (data[i, 2] - data[i, 1]) > 2000000:
            burst.append([data[i, 2] - data[i, 1], data[i, 3] - data[i, 4]])
        else:
            duration = data[i, 2] - data[i, 1]
            ideal = duration * 10E9 / (1500 * 8)
            long.append(data[i, 4] / ideal)
            # long.append(data[i, 4] * 1500 * 8 / duration / 1E9)
            real_sum += data[i,4]
            ideal_sum += ideal
    
    # print(real_sum, ideal_sum, real_sum/ideal_sum)
    # return np.array(burst), np.array(long)
    return np.array(burst), real_sum/ideal_sum


def read_results(dir):

    algorithms = []
    all_burst = {}
    lossless_burst = {}
    throughput = {}
    dcThres = np.arange(100, 2100, 100)
    # dcThres = [10,20,100,500]

    for i in dcThres:
        algorithms.append('TDT_' + str(i))
        all_burst['TDT_' + str(i)] = np.array([])
        lossless_burst['TDT_' + str(i)] = np.array([])
        throughput['TDT_' + str(i)] = []

    for i in range(50):
        for algorithm in algorithms:
            data = np.loadtxt(dir + 'trace' + str(i) + '_' + algorithm)
            result, long = data_postprocessing(data)
            all_burst[algorithm] = np.hstack((all_burst[algorithm], result[:,
                                                                           0]))
            lossless = result[result[:, 1] == 0]
            lossless_burst[algorithm] = np.hstack(
                (lossless_burst[algorithm], lossless[:, 0]))
            throughput[algorithm].append(np.mean(long))

    burst, bin_edges = np.histogram(all_burst['TDT_100'],
                                    bins=5,
                                    range=(0, 2e-4))
    # print(burst)
    # # print(bin_edges)
    burst[-1] += np.sum(all_burst['TDT_100'] > 2e-4)
    # print(np.sum(burst))

    lossless = {}
    for algorithm in algorithms:
        lossless[algorithm], burst_absorbing = np.histogram(
            lossless_burst[algorithm], bins=5, range=(0, 2e-4))
        # print(lossless[algorithm])
        # print(np.sum(lossless[algorithm]))
        print(algorithm, np.sum(lossless[algorithm]) / np.sum(burst))
        print(algorithm, np.mean(throughput[algorithm]))
        lossless[algorithm] = np.sum(lossless[algorithm]) / np.sum(burst)
        # lossless[algorithm] = lossless[algorithm] / burst
    return lossless, throughput


def plot_throughput(throughput):
    d = []
    label = []
    for i in range(10, 310, 10):
        d.append('TDT_' + str(i))
        if (i in [10, 50, 100, 150, 200, 250, 300]):
            label.append(str(i))
        else:
            label.append('')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(0.65, 0.98)
    # plt.yticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    plt.subplots_adjust(wspace=0.4, hspace=0.2, left=0.2)
    # plt.errorbar(x=d, y=tp_mean, yerr=tp_std)
    plt.boxplot([throughput[t] for t in d], labels=label, showfliers=False)
    plt.grid(linestyle='-.')
    plt.ylabel('Throughput(norm)', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.savefig('data/plot_result/throughput.png')


def plot_lossless(lossless):
    algorithms = []
    y = []
    label = []
    for i in range(10, 310, 10):
        algorithms.append('TDT_' + str(i))
        y.append(lossless['TDT_' + str(i)])
        if (i in [10, 50, 100, 150, 200, 250, 300]):
            label.append(str(i))
        else:
            label.append('')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, left=0.2)
    x = np.arange(30)
    width = 1
    # plt.ylim(0.8,1)
    colors = [
        'lightsteelblue', 'cornflowerblue', 'royalblue', 'mediumblue',
        'darkblue'
    ]
    # for i, algorithm in enumerate(algorithms):
    #     for j in range(3):
    #         ax.bar(x, lossless[algorithm][j], width=0.9 * width,
    #             color=colors[j])
    plt.plot(x, y, linewidth=5, linestyle='--')
    plt.xticks(x, label)

    plt.grid(linestyle='-.')
    plt.ylabel('Lossless Ratio', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.savefig('data/plot_result/lossless.png')


def plot_both(throughput, lossless):
    algorithms = []
    y = []
    label = []
    dcThres = np.arange(100, 2100, 100)
    for i in dcThres:
        algorithms.append('TDT_' + str(i))
        y.append(lossless['TDT_' + str(i)])
        if (i in [100, 500, 1000]):
            label.append('B/%d'%(2000/i))
        elif i == 1500:
            label.append('3B/4')
        elif i == 2000:
            label.append('B')
        else:
            label.append('')
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.2,
                        left=0.15,
                        right=0.85,
                        bottom=0.2)
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(0.928, 0.952)
    plt.yticks([0.93, 0.94, 0.95])

    bp = ax.boxplot([throughput[t] for t in algorithms],
                labels=label,
                showfliers=False,
                boxprops={
                    'color': 'RoyalBlue',
                    'facecolor':'RoyalBlue',
                    'linewidth': 2
                },
                medianprops={
                    'color': 'Navy',
                    'linewidth': 2
                },
                capprops={
                    'color': 'RoyalBlue',
                    'linewidth': 2
                },
                whiskerprops={
                    'color': 'Gray',
                    'linewidth': 2
                },
                flierprops={'markersize': 12},
                patch_artist=True)

    ax.legend([bp['boxes'][0]], ['throughput'], loc=[0, 0.75], fontsize=32, frameon=False, handletextpad=0.2)

    plt.grid(linestyle='-.')
    plt.ylabel('Throughput(norm)', fontsize=32)
    plt.xlabel('Drop Threshold', fontsize=32)
    plt.tick_params(labelsize=32)

    plt.grid(linestyle='-.')
    plt.ylabel('Throughput(norm)', fontsize=32)
    plt.xlabel('Drop Threshold', fontsize=32)
    plt.tick_params(labelsize=32)

    ax2 = ax.twinx()
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['left'].set_color('gray')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    x = np.arange(1, 21, 1)
    plt.ylim(0.598, 0.622)
    plt.yticks([0.60, 0.61, 0.62])
    ax2.plot(x, y, linewidth=5, linestyle='-', marker='o', color='FireBrick', label='Lossless Ratio')

    plt.grid(linestyle='-.')
    plt.ylabel('Lossless Ratio', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.legend(fontsize=32, loc=[0.47, 0.75], frameon=False, handletextpad=0.2)
    plt.savefig('data/plot_result/sensitivity.pdf')


def loss_ratio_vs_burst_length():
    algorithms = ['DT', 'EDT', 'TDT', 'OP', 'CS', 'ST']
    algorithms = ['DT', 'EDT', 'TDT', 'TDT_2state']
    lossSaveDir = 'data/results/stochastic/2021-12-08/loss/'

    all_burst = {}
    lossless_burst = {}
    throughput = {}

    for algo in algorithms:
        all_burst[algo] = np.array([])
        lossless_burst[algo] = np.array([])
        throughput[algo] = []

    for i in range(10):
        for algorithm in algorithms:
            data = np.loadtxt(lossSaveDir + 'trace' + str(i) + '_' + algorithm)
            result, long = data_postprocessing(data)
            all_burst[algorithm] = np.hstack((all_burst[algorithm], result[:, 0]))
            lossless = result[result[:, 1] == 0]
            lossless_burst[algorithm] = np.hstack(
                (lossless_burst[algorithm], lossless[:, 0]))
            throughput[algorithm].append(np.mean(long))

    burst, bin_edges = np.histogram(all_burst['DT'], bins=6, range=(0, 3e-4))
    burst[-1] += np.sum(all_burst['DT'] > 3e-4)
    print(bin_edges)
    print(burst)

    lossless = {}
    for algorithm in algorithms:
        lossless[algorithm], burst_absorbing = np.histogram(
            lossless_burst[algorithm], bins=6, range=(0,3e-4))
        print(algorithm, lossless[algorithm])
        # print(np.sum(lossless[algorithm])/np.sum(burst))
        lossless[algorithm] = lossless[algorithm] / burst
        print(algorithm, len(throughput[algorithm]))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.2,
                        bottom=0.2,
                        left=0.1,
                        right=1.0)
    xlabels = [
        '[0,50)',
        '[50,100)',
        '[100,150)',
        '[150,200)',
        '[200,250)',
        '[250,+$\infty$)',
        # r'[2,+$\infty$)',
    ]

    x = np.arange(6)
    total_width, n = 0.75, 3
    width = total_width / n
    algorithms = ['DT', 'EDT', 'TDT', 'TDT_2state']
    labels = ['DT', 'EDT', 'TDT', 'TDT(2 state)']
    for i, algorithm in enumerate(algorithms):
        ax.bar(x + width * i,
               lossless[algorithm],
               width=0.75 * width,
               label=labels[i],
               color='w',
               edgecolor=colormap[algorithm],
               lw=3,
               hatch=hatchmap[algorithm])

    plt.grid(linestyle='-.')
    plt.legend(fontsize=32,
               loc='upper right',
               bbox_to_anchor=(1.03, 1.15),
               frameon=False,
               ncol=3)
    plt.tick_params(labelsize=24)
    plt.xticks(x + total_width / 2 - width / 2, xlabels)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('Lossless Ratio', fontsize=32)
    plt.xlabel('Burst Duration(us)', fontsize=32)
    plt.savefig('data/plot_result/lossless-10G.png')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.2,
                        bottom=0.2,
                        left=0.28,
                        right=1.0)

    algorithms = ['DT', 'EDT', 'TDT', 'TDT_2state']
    labels = ['DT', 'EDT', 'TDT', 'TDT(2 state)']
    plt.ylim(0.93, 0.95)
    plt.yticks([0.93,0.94,0.95])
    plt.boxplot([throughput[t] for t in algorithms],
                labels=algorithms,
                showfliers=False,
                boxprops={
                    'color': 'RoyalBlue',
                    'linewidth': 3
                },
                medianprops={
                    'color': 'FireBrick',
                    'linewidth': 3
                },
                capprops={
                    'color': 'RoyalBlue',
                    'linewidth': 3
                },
                whiskerprops={
                    'color': 'Gray',
                    'linewidth': 3
                },
                flierprops={'markersize': 12})
    plt.grid(linestyle='-.')
    plt.ylabel('Throughput(norm)', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.savefig('data/plot_result/throughput-10G.png')


def delay():
    algorithms = ['DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/stochastic/2021-06-15/loss/'

    delay = {}
    for algo in algorithms:
        delay[algo] = np.zeros((30, ), dtype=int)

    for i in range(10):
        for algorithm in algorithms:
            data = np.loadtxt(lossSaveDir + 'trace' + str(i) + '_' + algorithm)

            data = data[data[:, 0] == 2]
            data = data[data[:, 1] <= 8]

            for j in range(data.shape[0]):
                index = int(data[j, 2])
                delay[algorithm][index] += int(data[j, 5])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.4, hspace=0.2, bottom=0.2, left=0.3)

    for algorithm in algorithms:
        delay[algorithm] = np.cumsum(delay[algorithm])
        delay[algorithm] = delay[algorithm] / delay[algorithm][-1]

    print(delay)
    x = np.linspace(0, 30, 30)
    for algo in algorithms:
        ax.plot(x,
                delay[algo],
                label=algo,
                color=colormap[algo],
                linewidth=4,
                linestyle=linestylemap[algo])

    plt.xlim(9, 30)
    plt.ylabel('CDF', fontsize=32)
    plt.xlabel('Delay(ms)', fontsize=32)
    plt.yticks(fontsize=32)
    plt.xticks([10, 20, 30], fontsize=32)
    plt.legend()

    fig.savefig('data/plot_result/delay-0617.png')


if __name__ == "__main__":
    os.chdir(sys.path[0])
    os.chdir('../')
    dir = 'data/results/stochastic/TDT_0620/loss/'
    # lossless, throughput = read_results(dir)
    # plot_both(throughput, lossless)
    loss_ratio_vs_burst_length()
    # delay()
    # plot_throughput(throughput)
    # plot_lossless(lossless)
    # run_baselines()
