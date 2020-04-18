# Switch Buffer Management

Buffer management in shared-buffer switches

#### Queue Threshold Control

- threshold control for fifo queue is implemented in the following source files:

  1. src/network/utils/queue.h & queue.cc
  2. src/traffic-control/model/queue-disc.h & queue-disc.cc
  3. src/traffic-control/model/fifo-queue-disc.h & fifo-queue-disc.cc

- changes above enable following functions:

1. Set&Get the current threshold for a fifo queue
2. Set the overall buffer size limit
3. Set&Get current overall buffer usage
