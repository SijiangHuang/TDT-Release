import numpy as np
import os
import sys
from multiprocessing import Pool  # for process pool
from multiprocessing.dummy import Pool as ThreadPool  # for thread pool
import datetime
from tqdm import tqdm

transport = 'udp'
# transport = 'tcp'


def run_baseline_parallel(inFile, outFile, buffer, lineRate, delay,
                          algorithm, plotFile, seed, alpha):

    if algorithm == 'TDT':
        filename = 'TDT_v3'
    elif algorithm == 'PO':
        filename = 'pushout'
    else:
        filename = 'EDT'
    if transport == 'tcp':
        filename = 'tcp'
        
    if algorithm == 'OP':
        algorithm = 'ST'
        buffer = 16 * buffer

    cmdLine = './waf --run-no-build \"scratch/' + filename + \
        ' --algorithm=' + algorithm + \
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


def run_baseline(traceDir, lossSaveDir, plotDir, algorithms, inFiles,
                 buffer, lineRate, delay, seed=1, alpha=1.0):

    os.system('./waf')
    args = []
    for algorithm in algorithms:
        for f in inFiles:
            inFile = traceDir + f
            outFile = lossSaveDir + f + '_' + algorithm
            plotFile = plotDir + f + '_' + algorithm

            args.append((inFile, outFile, buffer, lineRate,
                         delay, algorithm, plotFile, seed, alpha))

    # pool = ThreadPool(8)
    # pool.map(run_baseline_wrap, args)
    # pool.close()
    # pool.join()

    with ThreadPool(8) as pool:
        res=list(tqdm(pool.imap(run_baseline_wrap,args),total=len(args),desc='current progress: '))
    pool.close()
    pool.join()

def loss_vs_burstlength():

    # traceDir = 'data/trace/determine/'
    traceDir = 'data/trace/burst_length/'
    # lossSaveDir = 'data/results/determine/250-2000/loss/'
    # plotDir = 'data/results/determine/250-2000/plot/'
    lossSaveDir = 'data/results/burst_length/loss/'
    plotDir = 'data/results/burst_length/plot/'
    # inFiles = ['burst_fairness']
    inFiles = []
    for i in range(100, 1300, 100):
        inFiles.append('microburst_' + str(i) + 'us')

    buffer = 667
    lineRate = '1Gbps'
    delay = '1ms'
    seed = 1
    alpha = 1.0
    algorithms = ['DT', 'TDT', 'EDT', 'PO']
    algorithms = ['OP']

    run_baseline(traceDir, lossSaveDir, plotDir, algorithms,
                 inFiles, buffer, lineRate, delay, seed, alpha)


def determine_baseline():

    traceDir = 'data/trace/determine/'
    lossSaveDir = 'data/results/determine/loss/'
    plotDir = 'data/results/determine/plot/'
    inFiles = ['delay-sensitive']
    inFiles = ['microburst_1000us']

    buffer = 667
    lineRate = '1Gbps'
    delay = '1ms'
    seed = 1
    alpha = 1.0
    algorithms = ['DT', 'TDT', 'EDT', 'PO', 'ST', 'CS', 'OP']

    run_baseline(traceDir, lossSaveDir, plotDir, algorithms,
                 inFiles, buffer, lineRate, delay, seed, alpha)


def stochastic_flow():

    traceDir = 'data/trace/stochastic/8Gbps_50%_250us/'
    lossSaveDir = 'data/results/stochastic/8Gbps_50%_250us/loss/'
    plotDir = 'data/results/stochastic/8Gbps_50%_250us/plot/'

    config = '30%_500us_100ms'
    config = '30%_250us_1'
    # config = '8Gbps_30%_0'
    traceDir = 'data/trace/stochastic/' + config + '/'
    saveDir = 'data/results/stochastic/' + config
    lossSaveDir = 'data/results/stochastic/' + config + '/loss/'
    plotDir = 'data/results/stochastic/'+ config +'/plot/'

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    if not os.path.exists(lossSaveDir):
        os.makedirs(lossSaveDir)
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)   


    inFiles = []
    # for time in range(500, 1600, 100):
    #     for seed in range(5):
    #         inFiles.append(str(time) + 'us_' + str(seed))
    # inFiles = ['P+LN_20%', 'P+LN_50%']
    # inFiles = ['burst_test']
    for i in range(50):
        inFiles.append('trace' + str(i))

    buffer = 667
    lineRate = '1Gbps'
    delay = '1ms'
    seed = 1
    alpha = 1.0
    algorithms = ['DT', 'TDT', 'EDT', 'PO', 'ST', 'CS']
    # algorithms = ['DT', 'TDT', 'EDT']
    algorithms = ['OP']
    run_baseline(traceDir, lossSaveDir, plotDir, algorithms,
                 inFiles, buffer, lineRate, delay, seed, alpha)


def data_postprocessing(data):
    result = []
    data = data[data[:, 0] >= 8]
    for i in range(data.shape[0]):
        result.append([data[i, 2]-data[i, 1], data[i, 3]-data[i, 4]])
    return np.array(result)


def tcp_baseline():

    traceDir = 'data/trace/tcp/2_ports/'
    saveDir = 'data/results/tcp/2_ports/'
    lossSaveDir = 'data/results/tcp/2_ports/loss/'
    plotDir = 'data/results/tcp/2_ports/plot/'
    inFiles = []

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    if not os.path.exists(lossSaveDir):
        os.makedirs(lossSaveDir)
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)   
    # for time in range(500, 1600, 100):
    #     for seed in range(5):
    #         inFiles.append(str(time) + 'us_' + str(seed))
    # inFiles = ['P+LN_20%', 'P+LN_50%']
    # inFiles = ['burst_test']
    for i in range(100):
        inFiles.append('trace' + str(i))

    buffer = 667
    lineRate = '1Gbps'
    delay = '1.5ms'
    seed = 1
    alpha = 1.0
    algorithms = ['DT', 'TDT', 'EDT']
    algorithms = ['TDT', 'EDT','CS','ST']
    run_baseline(traceDir, lossSaveDir, plotDir, algorithms,
                 inFiles, buffer, lineRate, delay, seed, alpha)


if __name__ == "__main__":
    os.chdir(sys.path[0])
    os.chdir('../')

    # determine_baseline()
    # loss_vs_burstlength()
    stochastic_flow()
    # tcp_baseline()

    # dt_data = np.loadtxt('data/results/stochastic/loss/burst_test_DT')
    # edt_data = np.loadtxt('data/results/stochastic/loss/burst_test_EDT')
    # tdt_data = np.loadtxt('data/results/stochastic/loss/burst_test_TDT')

    # f = open('data/results/stochastic/4Gbps_50%/result.txt', 'w')
    # lossSaveDir = 'data/results/stochastic/4Gbps_50%/loss/'
    # for time in range(500, 1600, 100):
    #     for seed in range(5):
    #         line = '%d %d' % (time, seed)
    #         for algorithm in ['DT', 'EDT', 'TDT']:
    #             data = np.loadtxt(lossSaveDir + str(time) +
    #                               'us_' + str(seed) + '_' + algorithm)
    #             result = data_postprocessing(data)
    #             temp = np.sum(result[:, 1] == 0)
    #             line += ' %d' % temp
    #         f.write(line + '\n')
    # f.close()

    # edt_result = data_postprocessing(edt_data)
    # tdt_result = data_postprocessing(tdt_data)

    # print(np.sum(dt_result[:, 1] == 0))
    # print(np.sum(edt_result[:, 1] == 0))
    # print(np.sum(tdt_result[:, 1] == 0))
