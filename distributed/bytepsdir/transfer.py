import ray
import os
import socket

ray.init(address="auto")
nodes_info = ray.nodes()

nodes = [n['NodeManagerAddress'] for n in nodes_info]
driver_node_id = socket.gethostbyname(socket.gethostname())
nodes.remove(driver_node_id)

for x in nodes:
    print(x)
    cmd = 'scp -i /home/ubuntu/ray_bootstrap_key.pem -r /home/ubuntu/CheckFreq/distributed/bytepsdir ubuntu@' + x + ':/home/ubuntu/CheckFreq/distributed/'
    os.system(cmd)
