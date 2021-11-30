nvidia-docker run -it --mount src=/,target=/datadrive/,type=bind --net=host --shm-size=32768m bytepsimage/pytorch bash

export NVIDIA_VISIBLE_DEVICES=0
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=1
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234 # the scheduler port


bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py --model resnet50 --num-iters 1000000

