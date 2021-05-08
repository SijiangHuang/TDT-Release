/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Universita' degli Studi di Napoli "Federico II"
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Baseline comparision of ST(static threshold), CS(Complete Sharing) and DT(Dynamic Threshold) 
 * Author: Sijiang Huang <huangsj@bupt.edu.cn>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/packet-sink.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/timer.h"
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <assert.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TDT");

// topology variables
uint32_t nPorts = 16;
const uint32_t maxPorts = 16;

// statistical variables
std::vector<uint32_t> PortPass(nPorts, 0);
std::vector<uint32_t> PortLoss(nPorts, 0);
std::vector<uint32_t> PacketsInPort(nPorts, 0);

uint32_t sampleInterval = 5;
uint32_t sampleCounter = 0;

// buffer variables
std::string algo = "TDT";
uint32_t segmentSize = 1448;
uint32_t buffer = 175; // 0.25MB

// TDT variables
std::vector<uint32_t> dequeueCounter(nPorts, 0);
std::vector<uint32_t> netEnqueueCounter(nPorts, 0);
std::vector<uint32_t> lineRateCounter1(nPorts, 0);
std::vector<uint32_t> lineRateCounter2(nPorts, 0);
std::vector<uint32_t> dropCounter(nPorts, 0);
std::vector<bool> edtFlag(nPorts, false);
uint32_t dcThres = 3, necThres = 8, lrc1Thres = 200, lrc2Thres = 1000, dropThres = 100;
float alpha = 1.0;
// port state: 0--normal/DT  1--burst absorb/EDT  2--evacuation
std::vector<float> portState(nPorts, 0);

void TcPacketsInQueueTrace(Ptr<OutputStreamWrapper> pfile, QueueDiscContainer *qdisc, uint32_t pos, uint32_t oldValue, uint32_t newValue)
{
    // std::cout << "--------" << std::endl;
    // std::cout << "Time: " << Simulator::Now().GetSeconds() << std::endl;
    // packet dequeue event
    if (oldValue > newValue)
    {
        PortPass[pos] += oldValue - newValue;
        dequeueCounter[pos]++;
        if (dequeueCounter[pos] == dcThres)
        {
            // std::cout << "Dequeue Trigger: " << dequeueCounter[pos] << std::endl;
            netEnqueueCounter[pos] = 0;
            portState[pos] = 0;
            lineRateCounter2[pos] = 0;
            edtFlag[pos] = false;
            dropCounter[pos] = 0;
        }
        netEnqueueCounter[pos] = netEnqueueCounter[pos] > 0 ? netEnqueueCounter[pos] - 1 : 0;
        lineRateCounter1[pos]++;
        if (lineRateCounter1[pos] == lrc1Thres)
        {
            // std::cout << "lineRateCounter1 Trigger: " << lineRateCounter1[pos] << std::endl;
            // dropCounter[pos] = 0;
            lineRateCounter1[pos] = 0;
            edtFlag[pos] = true;
        }
        if (portState[pos] == 1)
        {
            lineRateCounter2[pos]++;
            if (lineRateCounter2[pos] == lrc2Thres)
            {
                // std::cout << "lineRateCounter2 Trigger: " << lineRateCounter2[pos] << std::endl;
                portState[pos] = 0;
                edtFlag[pos] = false;
            }
        }

        // std::cout << "dequeueCounter: " << dequeueCounter[pos] << std::endl;
        // std::cout << "netEnqueueCounter: " << netEnqueueCounter[pos] << std::endl;
        // std::cout << "lineRateCounter1: " << lineRateCounter1[pos] << std::endl;
        // std::cout << "lineRateCounter2: " << lineRateCounter2[pos] << std::endl;
    }
    // packet enqueue event
    else
    {
        dequeueCounter[pos] = 0;
        netEnqueueCounter[pos]++;
        if (netEnqueueCounter[pos] == necThres)
        {
            portState[pos] = 1;
            edtFlag[pos] = true;
        }
        *pfile->GetStream() << 1 << " " << pos << " " << Simulator::Now().GetSeconds() << " " << newValue - oldValue << std::endl;

        // std::cout << "dequeueCounter: " << dequeueCounter[pos] << std::endl;
        // std::cout << "netEnqueueCounter: " << netEnqueueCounter[pos] << std::endl;
        // std::cout << "lineRateCounter1: " << lineRateCounter1[pos] << std::endl;
        // std::cout << "lineRateCounter2: " << lineRateCounter2[pos] << std::endl;
    }

    if (edtFlag[pos])
    {
        netEnqueueCounter[pos] = 0;
        lineRateCounter1[pos] = 0;
        edtFlag[pos] = (portState[pos] == 1);
    }
    else
    {
        // dequeueCounter[pos] = 0;
    }

    uint32_t sumPackets = 0;
    Ptr<QueueDisc> q = qdisc[pos].Get(0);
    PacketsInPort[pos] = newValue;
    // std::cout << "Queue Length: ";
    for (uint32_t i = 0; i < nPorts; i++)
    {
        sumPackets += PacketsInPort[i];
        // std::cout << PacketsInPort[i] << " ";
    }
    // std::cout << std::endl;
    q->SetTotalBufferUse(sumPackets);

    // *pfile->GetStream() << 0 << " " << pos << " " << Simulator::Now().GetSeconds() << " " << PacketsInPort[pos] << std::endl;
    sampleCounter++;
    if (sampleCounter == sampleInterval)
    {
        sampleCounter = 0;
        for (uint32_t i = 0; i < nPorts; i++)
        {
            *pfile->GetStream() << 0 << " " << i << " " << Simulator::Now().GetSeconds() << " " << PacketsInPort[i] << std::endl;
        }
        *pfile->GetStream() << 0 << " " << nPorts << " " << Simulator::Now().GetSeconds() << " " << sumPackets << std::endl;
    }

    // uint32_t threshold = buffer;
    // if (algo == "DT" || algo == "EDT")
    // {
    //     threshold = (buffer > sumBytes) ? (alpha * (buffer - sumBytes)) : 0;
    // }
    // else if (algo == "ST")
    // {
    //     threshold = buffer / nPorts;
    // }

    uint32_t burstPorts = 0;
    for (uint32_t i = 0; i < nPorts; i++)
    {
        burstPorts += portState[i] == 1;
    }

    // std::cout << "Thres: ";
    for (uint32_t i = 0; i < nPorts; i++)
    {
        q = qdisc[i].Get(0);
        uint32_t threshold = (buffer > sumPackets) ? (alpha * (buffer - sumPackets)) : 0;
        if (portState[i] == 1)
        {
            threshold = buffer / burstPorts;
        }
        else if (portState[i] == 2)
        {
            threshold = threshold < buffer / nPorts ? threshold : buffer / nPorts;
        }
        // std::cout << "(" << threshold << ") ";
        q->SetThreshold(QueueSize(QueueSizeUnit::PACKETS, threshold));
        if (portState[i] == 2 && threshold > 2 * PacketsInPort[i])
        {
            portState[i] = 0;
            dropCounter[i] = 0;
        }
    }
    // std::cout << std::endl;
}

void Drop(Ptr<OutputStreamWrapper> pfile, Ptr<QueueDisc> q, uint32_t pos, Ptr<const QueueDiscItem> p)
{
    // std::cout << "=========" << std::endl;
    // std::cout << "Drop: " << Simulator::Now().GetSeconds() << std::endl;

    if (q->GetTotalBufferUse() == buffer)
    {
        portState[pos] = 0;
        lineRateCounter2[pos] = 0;
        edtFlag[pos] = false;
    }
    netEnqueueCounter[pos] = 0;
    dequeueCounter[pos] = 0;
    PortLoss[pos]++;
    dropCounter[pos]++;
    if (dropCounter[pos] == dropThres)
    {
        // std::cout << "Load Trigger" << std::endl;
        edtFlag[pos] = false;
        dropCounter[pos] = 0;
        portState[pos] = 2;
    }
    *pfile->GetStream() << 1 << " " << pos << " " << Simulator::Now().GetSeconds() << " " << -1 << std::endl;

    // std::cout << "dropCounter: " << dropCounter[pos] << std::endl;
    // std::cout << "dequeueCounter: " << dequeueCounter[pos] << std::endl;
    // std::cout << "netEnqueueCounter: " << netEnqueueCounter[pos] << std::endl;
    // std::cout << "lineRateCounter1: " << lineRateCounter1[pos] << std::endl;
    // std::cout << "lineRateCounter2: " << lineRateCounter2[pos] << std::endl;
    // std::cout << Simulator::Now().GetSeconds() << " packet size of " << p->GetPacket()->GetSize() << " dropped in port " << pos << std::endl;
}

int main(int argc, char *argv[])
{
    /* Log Component for debugging */
    // LogComponentEnable("PacketSink", LOG_LEVEL_ALL);
    // LogComponentEnable("UdpSocketImpl", LOG_LEVEL_ALL);
    // LogComponentEnable("Baselines", LOG_LEVEL_ALL);
    // LogComponentEnable("QueueDisc", LOG_LEVEL_ALL);
    // LogComponentEnable("PrioQueueDisc", LOG_LEVEL_ALL);
    // LogComponentEnable("Packet", LOG_LEVEL_ALL);

    // variables in simulation
    float simulationTime = 50.0;
    uint32_t servPort = 5050;
    uint32_t simSeed = 1;
    std::string transportProt = "Udp";
    std::string socketType = "ns3::UdpSocketFactory";
    std::string lineRate = "1024Mbps";
    std::string delay = "1ms";
    std::string inputFile = "data/3.in";
    std::string outputFile = "data/3.out";
    std::string plotFile = "data/plot.txt";

    CommandLine cmd;
    cmd.AddValue("algorithm", "Buffer sharing algorithm: DT, ST, CS, EDT. Default: DT", algo);
    cmd.AddValue("simSeed", "Seed for random generator. Default: 1", simSeed);
    cmd.AddValue("inFile", "Input file name", inputFile);
    cmd.AddValue("outFile", "Output file name", outputFile);
    cmd.AddValue("plotFile", "Output file for plotting the buffer-usage", plotFile);
    cmd.AddValue("buffer", "Total buffer size in packets", buffer);
    cmd.AddValue("lineRate", "Line speed of output port", lineRate);
    cmd.AddValue("delay", "Delay of the point to point channel", delay);
    cmd.AddValue("alpha", "the alpha in Dynamic Threshold", alpha);

    cmd.Parse(argc, argv);
    RngSeedManager::SetSeed(simSeed);
    dropThres = buffer;

    // output file for plotting
    AsciiTraceHelper asciiTraceHelper;
    Ptr<OutputStreamWrapper> pfile = asciiTraceHelper.CreateFileStream(plotFile);

    std::ifstream ifile(inputFile);
    ifile >> simulationTime;
    std::ofstream ofile(outputFile);

    NodeContainer nodes;
    nodes.Create(maxPorts + 1);

    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue(lineRate));
    pointToPoint.SetChannelAttribute("Delay", StringValue(delay));
    pointToPoint.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1p"));

    InternetStackHelper stack;
    stack.Install(nodes);

    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::FifoQueueDisc");

    QueueDiscContainer qdiscs[maxPorts];
    Ipv4InterfaceContainer interfaces[maxPorts];
    for (uint32_t i = 1; i <= nPorts; i++)
    {
        uint32_t src = 0, dst = i;
        NetDeviceContainer devices = pointToPoint.Install(nodes.Get(src), nodes.Get(dst));

        uint32_t pos = i - 1;
        qdiscs[pos] = tch.Install(devices);
        Ptr<QueueDisc> q = qdiscs[pos].Get(0);
        q->SetTotalBufferSize(buffer);
        q->TraceConnectWithoutContext("PacketsInQueue", MakeBoundCallback(&TcPacketsInQueueTrace, pfile, qdiscs, pos));
        q->TraceConnectWithoutContext("Drop", MakeBoundCallback(&Drop, pfile, q, pos));

        Ipv4AddressHelper address;
        char ipstring[16];
        sprintf(ipstring, "10.1.%d.0", i);
        address.SetBase(ipstring, "255.255.255.0");
        interfaces[pos] = address.Assign(devices);
    }
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    uint32_t nFlow;
    ifile >> nFlow;
    std::string fType, dataRate;
    uint32_t from, to, packetSize;
    packetSize = 1472;
    float startTime, endTime;
    for (uint32_t i = 0; i < nFlow; i++)
    {
        ifile >> fType >> from >> to >> startTime >> endTime;

        OnOffHelper onoff(socketType, Ipv4Address::GetAny());
        if (fType == "D")
        {
            ifile >> dataRate;
            onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        }
        else if (fType == "P")
        {
            dataRate = "100Gbps";
            std::string PoissonMean;
            std::string packetTimeLength = "1.1584E-7";
            ifile >> PoissonMean;
            onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=" + packetTimeLength + "]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=" + PoissonMean + "]"));
        }
        else if (fType == "LN")
        {
            std::string onMu, onSigma, offMu, offSigma, onMean, offMean;
            ifile >> dataRate >> onMu >> onSigma >> offMu >> offSigma >> onMean >> offMean;
            onoff.SetAttribute("OnTime", StringValue("ns3::LogNormalRandomVariable[Mu=" + onMu + "|Sigma=" + onSigma + "]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::LogNormalRandomVariable[Mu=" + offMu + "|Sigma=" + offSigma + "]"));
        }
        onoff.SetAttribute("PacketSize", UintegerValue(packetSize));
        onoff.SetAttribute("DataRate", StringValue(dataRate)); //bit/s
        ApplicationContainer apps;

        InetSocketAddress rmt(interfaces[to - 1].GetAddress(1), servPort + i);
        rmt.SetTos(0xb8);
        AddressValue remoteAddress(rmt);
        onoff.SetAttribute("Remote", remoteAddress);
        apps.Add(onoff.Install(nodes.Get(0)));
        apps.Start(Seconds(startTime));
        apps.Stop(Seconds(endTime));
    }
    ifile.close();

    // AsciiTraceHelper ascii;
    // pointToPoint.EnableAsciiAll (ascii.CreateFileStream ("baselines.tr"));
    // pointToPoint.EnablePcapAll ("baselines");

    FlowMonitorHelper flowmon;
    flowmon.SetMonitorAttribute("DelayBinWidth", DoubleValue(0.0001));
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();

    for (uint32_t i = 0; i < nPorts; i++)
    {
        ofile << 0 << " " << i << " " << PortPass[i] + PortLoss[i] << " " << PortPass[i] << " 0 0" << std::endl;
    }

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);

        // std::cout << "Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ") ("
        //           << t.sourcePort << "->" << t.destinationPort << ")\n";
        // std::cout << "  Tx Packets: " << i->second.txPackets << "\n";
        // std::cout << "  Tx Bytes:   " << i->second.txBytes << "\n";
        // std::cout << "  Tx Offered:  " << i->second.txBytes * 8.0 / (i->second.timeLastRxPacket - i->second.timeFirstTxPacket).GetSeconds() / 1024 / 1024 << " Mbps\n";
        // std::cout << "  Rx Packets: " << i->second.rxPackets << "\n";
        // std::cout << "  Rx Bytes:   " << i->second.rxBytes << "\n";
        // std::cout << "  Throughput: " << i->second.rxBytes * 8.0 / (i->second.timeLastRxPacket - i->second.timeFirstTxPacket).GetSeconds() /  << " Mbps\n";
        // std::cout << "  StartTime:  " << i->second.timeFirstTxPacket.GetSeconds() << "\n";
        // std::cout << "  EndTime:    " << i->second.timeLastRxPacket.GetSeconds() << "\n";
        ofile << 1 << " " << t.destinationPort - servPort << " " << i->second.timeFirstTxPacket.GetSeconds()
              << " " << i->second.timeLastTxPacket.GetSeconds() << " " << i->second.txPackets << " " << i->second.rxPackets << std::endl;

        ns3::Histogram h = i->second.delayHistogram;
        uint32_t nBins = h.GetNBins();
        for (uint32_t i = 0; i < nBins; i++)
        {
            ofile << 2 << " " << t.destinationPort - servPort << " " << i << " " << h.GetBinStart(i) << " " << h.GetBinEnd(i) << " "
                  << " " << h.GetBinCount(i) << std::endl;
        }
        ofile << 3 << " " << t.destinationPort - servPort << " " << i->second.rxBytes << " "
              << i->second.timeFirstTxPacket.GetSeconds()
              << " " << i->second.timeLastRxPacket.GetSeconds() << " "
              << " " << i->second.rxBytes * 8.0 / (i->second.timeLastRxPacket - i->second.timeFirstTxPacket).GetSeconds() << std::endl;
    }

    Simulator::Destroy();

    // std::cout << "---------------" << algo << "---------------" << std::endl;
    // std::cout << inputFile << std::endl;
    // std::cout << "Packets received in each port" << std::endl;

    // std::cout << "Packets loss in each port" << std::endl;
    for (uint32_t i = 0; i < nPorts; i++)
    {
        // std::cout << PortLoss[i] << " ";
        // ofile << PortLoss[i] << " ";
    }
    // std::cout << std::endl;
    // ofile << std::endl;
    ofile.close();

    return 0;
}
