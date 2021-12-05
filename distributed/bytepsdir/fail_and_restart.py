import ray
import os
import socket
import time
import run_byteps

all_nodes=4

ray.init(address="auto")

time_int = 60

#time.sleep(time_int)
nodes_info = ray.nodes()
nodes = set([n['NodeManagerAddress'] for n in nodes_info])
local_hostname = socket.gethostbyname(socket.gethostname())
nodes.remove(local_hostname)
to_kill = list(nodes)[0]
nodes.remove(to_kill)
nodes.add(local_hostname)

os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + to_kill + " docker kill byteps")
os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + to_kill + " docker rm byteps")
# TODO; node failure here

print("node: " + to_kill +  " failed")
for i in nodes:
    os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + i + " docker kill byteps")
    os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + i + " docker rm byteps")

print("all clear!")

## simulate node removal - time interval taken from real measuerements
## used simulation because ray has issues when the node is up again. TODO: fix this
#time.sleep(40)
#run_byteps.run()

