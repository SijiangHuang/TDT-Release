import numpy as np
import os
import sys
from math import floor
from matplotlib import pyplot as plt
from tqdm import tqdm

FLOW_SIZE_MICE = 100000


def compute_jains_index(n, throughput, fair_throughput):
    throughput = np.array(throughput)
    fair_throughput = np.array(fair_throughput)
    x = np.zeros((len(throughput), ), dtype=float)
    for i in range(len(fair_throughput)):
        if fair_throughput[i]:
            x[i] = throughput[i] / fair_throughput[i]
    return (np.sum(x) ** 2) / (n * np.sum(x ** 2))


def determine_fairness():
    algorithms = ['DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/determine/250-2000/loss/'
    po_data = np.loadtxt(lossSaveDir+'microburst_1000us_PO')
    # po_data = np.loadtxt(lossSaveDir+'burst_fairness_PO')
    fair_throughput = po_data[0, :]
    print(fair_throughput)
    for algorithm in algorithms:
        data = np.loadtxt(lossSaveDir+'microburst_1000us_' + algorithm)
        throughput = data[0, :]
        print(throughput)
        print(algorithm, compute_jains_index(3, throughput, fair_throughput))


def stochastic_fairness():
    algorithms = ['CS', 'ST', 'DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/stochastic/30%_250us_1/loss/'
    result = np.zeros((5, 50), dtype=float)
    for i in range(50):
        po_data = np.loadtxt(lossSaveDir + 'trace' + str(i) + '_OP')
        po_data = po_data[po_data[:, 0] == 0]
        fair_throughput = po_data[:, 3]
        for j, algorithm in enumerate(algorithms):
            data = np.loadtxt(lossSaveDir + 'trace' + str(i) + '_' + algorithm)
            data = data[data[:, 0] == 0]
            throughput = data[:, 3]
            fairness_index = compute_jains_index(
                9, throughput, fair_throughput)
            # print(fairness_index)
            result[j, i] = fairness_index

    return result


def bytes_to_packets():
    traceDir = 'data/trace/tcp/Config_incast_16/'
    newtraceDir = 'data/trace/tcp/16ports/'

    for i in range(105):
        inFile = traceDir + 'trace' + str(i)
        outFile = newtraceDir + 'trace' + str(i)
        fi = open(inFile, 'r')
        fo = open(outFile, 'w')
        for i, line in enumerate(fi.readlines()):
            if i < 2:
                fo.write(line)
            else:
                temp = line.split()
                temp[3] = (int(temp[3])//1448 + 1) * 1448
                fo.write("%s %s %s %s %s\n" %
                         (temp[0], temp[1], temp[2], temp[3], temp[4]))


def mice_elephant(fct_list, fsize_list):
    fsize_list = np.array(fsize_list)
    fct_list = np.array(fct_list)
    mice_flow_index = fsize_list <= FLOW_SIZE_MICE
    elephant_flow_index = fsize_list > FLOW_SIZE_MICE
    mice_fct = fct_list[mice_flow_index]
    mice_fsize = fsize_list[mice_flow_index]
    elephant_fct = fct_list[elephant_flow_index]
    elephant_fsize = fsize_list[elephant_flow_index]
    if mice_fct.size == 0:
        mice_fct = np.array([0])
    if elephant_fct.size == 0:
        elephant_fct = np.array([0])
    return mice_fct.tolist(), mice_fsize.tolist(), elephant_fct.tolist(), elephant_fsize.tolist()


def get_fct():
    algorithms = ['DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/tcp/Config_incast_16/loss/'
    fct = {'DT': [], 'EDT': [], 'TDT': []}
    p99 = {'DT': [], 'EDT': [], 'TDT': []}
    fct_mice = {'DT': [], 'EDT': [], 'TDT': []}
    # for algorithm in algorithms:
    #     for i in range(105):
    #         data = np.loadtxt(lossSaveDir+'trace'+str(i)+'_'+algorithm)
    #         mice_fct, _, _, _ = mice_elephant(data[:, 1], data[:, 2])
    #         fct[algorithm].append(data[:, 1].mean())
    #         p99[algorithm].append(np.percentile(data[:, 1], 99))
    #         fct_mice[algorithm].append(np.mean(mice_fct))

    for algorithm in algorithms:
        # active = [0, 5, 15, 25, 45, 65, 85, 105]
        active = [0, 5, 15, 25, 45, 65]
        for ap in range(len(active) - 1):
            fct_temp = []
            p99_temp = []
            mice_temp = []
            for i in range(active[ap], active[ap+1]):
                data = np.loadtxt(lossSaveDir+'trace'+str(i)+'_'+algorithm)
                mice_fct, _, _, _ = mice_elephant(data[:, 1], data[:, 2])
                fct_temp.append(data[:, 1].mean())
                p99_temp.append(np.percentile(data[:, 1], 99))
                mice_temp.append(np.mean(mice_fct))
            fct[algorithm].append(np.mean(fct_temp))
            p99[algorithm].append(np.mean(p99_temp))
            fct_mice[algorithm].append(np.mean(mice_temp))

    result = np.zeros((len(fct['DT']), 3), dtype=float)
    result[:, 0], result[:, 1], result[:,
                                       2] = fct['DT'], fct['EDT'], fct['TDT']
    np.savetxt('data/mean.txt', result, fmt="%.3f")

    result[:, 0], result[:, 1], result[:,
                                       2] = p99['DT'], p99['EDT'], p99['TDT']
    np.savetxt('data/p99.txt', result, fmt="%.3f")

    return fct, p99, fct_mice


def tcp_fct():
    algorithms = ['DT', 'EDT', 'TDT']
    lossSaveDir = 'data/results/tcp/Config_incast_16/loss/'

    for i in range(50):
        for algorithm in algorithms:
            data = np.loadtxt(lossSaveDir+'trace'+str(i)+'_'+algorithm)
            print(data[:, 1].mean(), end=" ")
        print('\n')


def full_buffer():

    for i in tqdm(range(5, 15)):
        interval = np.random.exponential(scale=0.1, size=i)
        with open('data/trace/tcp-test/trace' + str(i), 'w') as f:
            f.write('10\n')
            f.write(str(i) + '\n')
            time = 0.1
            for row in range(i):
                time += interval[row]
                f.write(str(row % 3 + 10) + ' 1 ' +
                        str(time) + ' 20000000 0\n')

        filename = 'tcp'
        algorithm = 'DT'
        buffer = 667
        lineRate = '1Gbps'
        delay = '1.5ms'
        alpha = 1
        seed = 1
        inFile = 'data/trace/tcp-test/trace' + str(i)
        outFile = 'data/results/tcp-test/fct_0.1_' + str(i)
        plotFile = 'data/results/tcp-test/plot_0.1_' + str(i)
        cmdLine = './waf --run-no-build \"scratch/' + filename + \
            ' --algorithm=' + algorithm + \
            ' --buffer=' + str(buffer) + \
            ' --lineRate=' + lineRate + \
            ' --plotFile=' + plotFile + \
            ' --delay=' + delay + \
            " --inFile=" + inFile + \
            " --outFile=" + outFile + \
            " --alpha=" + str(alpha) + \
            " --simSeed=" + str(seed) + '\" > data/temp'
        # print(cmdLine)
        os.system(cmdLine)

    # for i,t in enumerate(np.linspace(0.05,0.3,6)):
    #     interval = np.random.exponential(scale=t, size=10)
    #     with open('data/trace/tcp-test/trace_' + str(i),'w') as f:
    #         f.write('10\n')
    #         f.write(str(10) + '\n')
    #         time = 0.1
    #         for row in range(10):
    #             time += interval[row]
    #             f.write(str(row % 3 + 10) + ' 1 ' + str(time) + ' 20000000 0\n')

    #     filename = 'tcp'
    #     algorithm = 'DT'
    #     buffer = 667
    #     lineRate = '1Gbps'
    #     delay = '1.5ms'
    #     alpha = 1
    #     seed = 1
    #     inFile = 'data/trace/tcp-test/trace_' + str(i)
    #     outFile = 'data/results/tcp-test/fct_' + str(i)
    #     plotFile = 'data/results/tcp-test/plot_' + str(i)
    #     cmdLine = './waf --run-no-build \"scratch/' + filename + \
    #     ' --algorithm=' + algorithm + \
    #     ' --buffer=' + str(buffer) + \
    #     ' --lineRate=' + lineRate + \
    #     ' --plotFile=' + plotFile + \
    #     ' --delay=' + delay + \
    #     " --inFile=" + inFile + \
    #     " --outFile=" + outFile + \
    #     " --alpha=" + str(alpha) + \
    #     " --simSeed=" + str(seed) + '\" > data/temp'
    #     # print(cmdLine)
    #     os.system(cmdLine)


def move_results():

    base = 0
    nFile = 6
    fromDir = 'data/results/DPDK/0629_2/'
    fromDir = 'data/results/DPDK/0629_3/'
    toDir = 'data/results/DPDK/result_0629/'
    toDir = 'data/results/DPDK/final_result/'
    algorithms = ['DT', 'EDT', 'TDT']

    for i in range(nFile):
        for algorithm in algorithms:
            cmd = 'sudo cp ' + fromDir + algorithm + '_' + \
                str(i) + '_2_flows.out ' + toDir + algorithm + \
                '_' + str(base + i) + '_2_flows.out'
            os.system(cmd)
            # cmd = 'sudo rm ' + toDir + algorithm + \
            #     '_' + str(base + i) + '_2_flows.out'
            # os.system(cmd)

if __name__ == "__main__":
    # determine_fairness()
    # stochastic_fairness()
    # bytes_to_packets()

    # tcp_fct()
    # fct, p99, fct_mice = get_fct()
    # full_buffer()
    move_results()
