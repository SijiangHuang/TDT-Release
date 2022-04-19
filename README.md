# Switch Buffer Management

Buffer management in shared-buffer switches

#### Overview: Queue Threshold Control

- threshold control for fifo queue is implemented in the following source files:

  1. src/network/utils/queue.h & queue.cc
  2. src/traffic-control/model/queue-disc.h & queue-disc.cc
  3. src/traffic-control/model/fifo-queue-disc.h & fifo-queue-disc.cc

- changes above enable following functions:

1. Set&Get the current threshold for a fifo queue
2. Set the overall buffer size limit
3. Set&Get current overall buffer usage

#### Configgure&Build 
- "./waf configure --disable-werror"
- "./waf build"

#### Run Code
- TDT&DT are implemented at scratch/TDT.cc
- To run the example: "./waf --run "scratch/TDT --inFile=data/trace0.txt --lineRate=1Gbps --algorithm=TDT"
- trace file (input), result file (output) in ./data
- (enbale if necessary) logging in data/plot.txt