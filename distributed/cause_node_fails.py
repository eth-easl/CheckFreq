import ray
import os


def remove_node(driver_node_id):
    #ray.init(address="auto")
    nodes_info = ray.nodes()
    nodes = [n['NodeManagerAddress'] for n in nodes_info]
    nodes.remove(driver_node_id)
    to_kill = nodes[0]
    cmd = "ssh -i /root/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + to_kill + " ray stop"
    os.system(cmd)

def remove_worker_node(driver_node_id):
    #ray.init(address="auto")
    nodes_info = ray.nodes()
    nodes = [n['NodeManagerAddress'] for n in nodes_info if 'GPU' in n["Resources"].keys()]
    if (driver_node_id in nodes):
        nodes.remove(driver_node_id)
    to_kill = nodes[0]

    cmd = "ssh -i /root/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no   ubuntu@" + to_kill + " ray stop"
    os.system(cmd)

def remove_server_node(driver_node_id): # considering non-colocated policy
    #ray.init(address="auto")
    nodes_info = ray.nodes()
    nodes = [n['NodeManagerAddress'] for n in nodes_info if not 'GPU' in n["Resources"].keys()]
    if (driver_node_id in nodes):
        nodes.remove(driver_node_id)
    to_kill = nodes[0]

    cmd = "ssh -i /root/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no  ubuntu@" + to_kill + " ray stop"
    os.system(cmd)
