# Play FPS game VizDoom with RL
Implemented [GA3C](https://github.com/NVlabs/GA3C) method for training FPS game VizDoom.

## System Reqirement
Must have a CUDA-supporting GPU; Multi-core CPU is preferred

On my PC (i7 3770 CPU + NVidia GTX 1070), it takes 20 hrs to train 300k mini-batches

## Current Result
After training 350k mini-batches on cig_custum map(a very simple map customized by myself) with 8 AI enemies, the test result is:
>Frag: 9.8 per 2000 steps

>kill/death: 1.15
