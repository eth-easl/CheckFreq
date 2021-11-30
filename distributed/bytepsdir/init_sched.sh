docker run -it --mount src=/,target=/datadrive/,type=bind --net=host bytepsimage/pytorch bash

export DMLC_NUM_WORKER=1
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1
export DMLC_PS_ROOT_PORT=1234

