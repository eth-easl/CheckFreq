- Install docker
```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
sudo usermod -aG docker $USER
newgrp docker
```

- Install nvidia-docker
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

- Get the codebase
```
git clone https://github.com/eth-easl/CheckFreq.git
cd CheckFreq && git checkout cifar
cd -
```

- Temporary hack for CIFAR-10 dataset
```
git clone https://github.com/YoongiKim/CIFAR-10-images.git cifar
mv cifar/test cifar/val
sudo mv cifar /
```

- Pull the `checkfreq-dali` container image
```
docker pull zhangks98/checkfreq-dali:hdparm
```

- Run the image
```
nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind -it --rm --network=host --privileged zhangks98/checkfreq-dali:hdparm
```

- Inside the image:
```
mkdir -p /mnt/CheckFreq
mount --bind /datadrive/home/kaiszhang/CheckFreq /mnt/CheckFreq
cd /mnt/CheckFreq
./train_checkfreq.sh
```

