import ray
import os

ray.init(address="auto")
nodes_info = ray.nodes()

worker_nodes = [n['NodeManagerAddress'] for n in nodes_info if 'GPU' in n["Resources"].keys()]
server_nodes = [n['NodeManagerAddress'] for n in nodes_info if not 'GPU' in n["Resources"].keys()]
scheduler = server_nodes[0]
server_nodes = server_nodes[1:]

numw = len(worker_nodes)
nums = len(server_nodes)
print("Scheduler is: ", scheduler)

# scheduler
cmd = "ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + scheduler + " python /home/ubuntu/bytepsdir/init_sched.py numw nums scheduler 1234"
os.system(cmd)


# servers
for i in range(nums):
    cmd = "ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + server_nodes[i] + " python /home/ubuntu/bytepsdir/init_server.py i numw scheduler 1234"
    os.system(cmd)  

# workers
for i in range(numw):
    cmd = "ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + server_nodes[i] + " python /home/ubuntu/bytepsdir/init_server.py i numw nums scheduler 1234 'bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py --model resnet50 --num-iters 1000' "
    os.system(cmd)  

