import sys
import os
import numpy as np
import math


def get_lognormal_parameters(mean, variance):
    mu = math.log(mean) - 0.5 * math.log(1 + variance / mean ** 2)
    sigma = math.sqrt(math.log(1 + variance / mean ** 2))
    return mu, sigma


def trace_generation():

    port_number = 8
    # burst_flow_number = 36
    # longlived_flow_number = 4
    # total_flow_number = port_number + burst_flow_number * \
    #     (port_number >> 0) + longlived_flow_number * (port_number >> 0)
    simulation_time = 1

    # background
    background = 0.2
    background_interval = 1 / (background * 10e9 / 1500 / 8)

    # loss-sensitive burst
    burst_on = 2e-4
    burst_rate = "80Gbps"

    burst_off = 0.0078

    # throughput-sensitive long-lived flow
    longlived_on = 0.02
    longlived_rate = "15Gbps"
    longlived_off = 0.13

    # simulation_time = (burst_on + burst_off) * burst_flow_number + \
    #     (longlived_on + longlived_off) * longlived_flow_number

    burst_on_mu, burst_on_sigma = get_lognormal_parameters(burst_on, burst_on ** 2)
    longlived_on_mu, longlived_on_sigma = get_lognormal_parameters(
        longlived_on, longlived_on ** 2
    )
    burst_off_mu, burst_off_sigma = get_lognormal_parameters(burst_off, burst_off ** 2)
    longlived_off_mu, longlived_off_sigma = get_lognormal_parameters(
        longlived_off, longlived_off ** 2
    )

    outDir = "data/trace/stochastic/40%_10G_0620/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    outDir += "trace"

    for seed in range(100):
        np.random.seed(seed)

        flows = []
        outFile = outDir + str(seed)
        for port in range(port_number):
            t = np.random.lognormal(burst_off_mu, burst_off_sigma, 1)
            while t < simulation_time:
                if np.random.rand() < 0.9:
                    ontime = np.random.lognormal(burst_on_mu, burst_on_sigma, 1)
                    offtime = np.random.lognormal(burst_off_mu, burst_off_sigma, 1)
                    flows.append(
                        "D 0 %d %.6f %.6f %s\n" % (port + 1, t, t + ontime, burst_rate)
                    )
                    t += ontime + offtime
                else:
                    ontime = np.random.lognormal(longlived_on_mu, longlived_on_sigma, 1)
                    offtime = np.random.lognormal(longlived_off_mu, longlived_off_sigma, 1)
                    flows.append(
                        "D 0 %d %.6f %.6f %s\n" % (port + 1, t, t + ontime, longlived_rate)
                    )
                    t += ontime + offtime
        with open(outFile, 'w') as f:
            f.write('%.5f\n' % simulation_time)
            f.write(str(len(flows) + port_number) + '\n')
            for i in range(port_number):
                f.write('P 0 %d %f %f %e\n' %
                        (i+1, 0, simulation_time, background_interval))
            for flow in flows:
                f.write(flow)
    # for port in range(port_number):
    #     time = 0
    #     on_time = np.random.lognormal(
    #         burst_on_mu, burst_on_sigma, burst_flow_number)
    #     off_time = np.random.lognormal(
    #         burst_off_mu, burst_off_sigma, burst_flow_number)
    #     for i in range(burst_flow_number):
    #         time +


if __name__ == "__main__":
    os.chdir(sys.path[0])
    os.chdir("../")
    trace_generation()
