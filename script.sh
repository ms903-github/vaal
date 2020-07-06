#!/bin/bash

#initialize modules
source /etc/profile.d/modules.sh

#modules
module load python/3.6
module load cuda/10.0
module load cudnn/7.4/7.4.2
module load nccl/2.3/2.3.5-2
module load openmpi/2.1.5
source work/bin/activate

cd "/home/acb11949pt/vaal"

python main.py --cuda --dataset "office-home" --batch_size 4 --latent_dim 32 --img_dim 64
