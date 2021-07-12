#!/usr/bin/env bash

mpirun -np 4 xterm -e gdb bin/runner -x initPipe.gdb
