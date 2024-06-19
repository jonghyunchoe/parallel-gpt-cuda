#!/bin/bash

salloc -N 4 --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 4 \
		--oversubscribe -quiet \
		./main $@