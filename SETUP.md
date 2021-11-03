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

- Pull checkfreq dali image
```
docker pull zhangks98/checkfreq-dali:py36_cu10.run
```

- Run the image
```
nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind -it --rm --network=host --privileged zhangks98/checkfreq-dali:py36_cu10.run
```

