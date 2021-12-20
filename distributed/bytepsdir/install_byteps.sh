sudo apt-get install libcudnn7=7.6.0.64-1+cuda10.0
sudo apt-get install  libnuma-dev
sudo apt-get install libnccl2=2.4.7-1+cuda10.0 
sudo apt-get install libnccl-dev=2.4.7-1+cuda10.0
python3 -m pip install -U numpy==1.18.1 torchvision==0.5.0 torch==1.4.0
cd /home/ubuntu/.local
git clone --recursive https://github.com/bytedance/byteps
cd byteps
python3 setup.py install --user
