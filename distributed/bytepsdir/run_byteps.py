import ray
import os
import socket

def run():

        #ray.init(address="auto")
        nodes_info = ray.nodes()

        worker_nodes = [n['NodeManagerAddress'] for n in nodes_info if 'GPU' in n["Resources"].keys()]
        server_nodes = [n['NodeManagerAddress'] for n in nodes_info if not 'GPU' in n["Resources"].keys()]
        scheduler = server_nodes[0]
        server_nodes = server_nodes[1:]
        if '10.138.0.101' in server_nodes:
            server_nodes.remove('10.138.0.101')
        
        print(server_nodes)
        local_hostname = socket.gethostbyname(socket.gethostname())
        worker_nodes.remove(local_hostname)
        worker_nodes = [local_hostname]+worker_nodes

        ### 1. decide on scheduler      
        nnodes = 1
        numw = 2#len(worker_nodes)
        nums = 2#len(server_nodes)

        
        print("Scheduler is: ", scheduler)
        print(numw, nums)
        ### 2. create env flles

        f1 = open('sch_list', 'w')
        f1.write('DMLC_NUM_WORKER=' + str(numw*nnodes) + '\n')
        f1.write('DMLC_NUM_SERVER=' + str(nums) + '\n')
        f1.write('DMLC_ROLE=scheduler\n')
        f1.write('DMLC_PS_ROOT_URI=' + scheduler + '\n')
        f1.write('DMLC_PS_ROOT_PORT=1234\n')
        f1.close()
        os.system("scp  -i /home/ubuntu/ray_bootstrap_key.pem sch_list ubuntu@" + scheduler + ":/home/ubuntu/")

        f2 = open('server_list', 'w')
        f2.write('DMLC_NUM_WORKER=' + str(numw*nnodes) + '\n')
        f2.write('DMLC_NUM_SERVER=' + str(nums) + '\n')
        f2.write('DMLC_ROLE=server\n')
        f2.write('DMLC_PS_ROOT_URI=' + scheduler + '\n')
        f2.write('DMLC_PS_ROOT_PORT=1234\n')
        f2.close()

        for i in range(nums):
            os.system("scp  -i /home/ubuntu/ray_bootstrap_key.pem server_list ubuntu@" + server_nodes[i] + ":/home/ubuntu/")

        for i in range(numw):
            for j in range(nnodes):
                idx = i*nnodes+j
                f = open('worker_list_' + str(idx), 'w')
                f.write('NVIDIA_VISIBLE_DEVICES=0,1,2,3\n') #+ str(j) + '\n')
                f.write('DMLC_WORKER_ID='+str(i)+'\n')
                f.write('DMLC_NUM_WORKER=' + str(numw*nnodes)+'\n')
                f.write('DMLC_NUM_SERVER=' + str(nums)+'\n')
                f.write('DMLC_ROLE=worker\n')
                f.write('DMLC_PS_ROOT_URI=' + scheduler + '\n')
                f.write('DMLC_PS_ROOT_PORT=1234\n')
                f.close()
                os.system("scp  -i /home/ubuntu/ray_bootstrap_key.pem worker_list_" + str(idx) + " ubuntu@" + worker_nodes[i] + ":/home/ubuntu/")


        ### 3. start docker on each node + start BPS
        cmd = "ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + scheduler + " docker run -d -t --name byteps --mount src=/,target=/datadrive/,type=bind  --env-file /home/ubuntu/sch_list --net=host fotstrt/checkfreq-dali-ray-byteps bash"
        os.system(cmd)
        os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + scheduler + " docker exec -d byteps bpslaunch")

        for i in range(nums):
            print(server_nodes[i])
            cmd = "ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + server_nodes[i] + " docker run -d -t --name byteps --mount src=/,target=/datadrive/,type=bind  --env-file /home/ubuntu/server_list --net=host fotstrt/checkfreq-dali-ray-byteps bash"
            os.system(cmd)
            os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + server_nodes[i] + " docker exec -d byteps bpslaunch")

        #
        for i in range(1,numw):
            print(worker_nodes[i])
            for j in range(nnodes):
                idx = i*nnodes + j
                cmd = "ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + worker_nodes[i] + " nvidia-docker run -d -t --name byteps" + str(j) + " --mount src=/,target=/datadrive/,type=bind  --env-file /home/ubuntu/worker_list_" + str(idx) + " --net=host --shm-size=32768m fotstrt/checkfreq-dali-ray-byteps bash  "
                os.system(cmd)
                runcmd = "ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + worker_nodes[i] + " docker exec -d -w /datadrive/home/ubuntu/CheckFreq/distributed/bytepsdir/  byteps"+str(j) +  " bpslaunch python3 byteps_imagenet.py --train-dir='/datadrive/home/ubuntu/ImageNet/train' --val-dir='/datadrive/home/ubuntu/ImageNet/val' --arch='vgg16' --epochs=80 --batch-size=64 --base-lr=0.0025 --cfreq=80"
                os.system(runcmd)

        for j in range(1, nnodes):
            cmd = "nvidia-docker run -d -t --name byteps" + str(j) + " --mount src=/,target=/datadrive/,type=bind  --env-file /home/ubuntu/worker_list_" + str(j) + " --net=host --shm-size=32768m fotstrt/checkfreq-dali-ray-byteps bash  "
            os.system(cmd)
            runcmd = "docker exec -d -w /datadrive/home/ubuntu/CheckFreq/distributed/bytepsdir/  byteps" + str(j) + " bpslaunch python3 byteps_imagenet.py --train-dir='/datadrive/home/ubuntu/ImageNet/train' --val-dir='/datadrive/home/ubuntu/ImageNet/val' --arch='vgg16' --epochs=80 --batch-size=64 --base-lr=0.0025 --cfreq=80"
            os.system(runcmd)

        # for head worker (rank0), run interactive
        cmd = "nvidia-docker run -d -t --name byteps0 --mount src=/,target=/datadrive/,type=bind  --env-file /home/ubuntu/worker_list_0 --net=host --shm-size=32768m fotstrt/checkfreq-dali-ray-byteps bash  "
        os.system(cmd)
        runcmd = "docker exec -w /datadrive/home/ubuntu/CheckFreq/distributed/bytepsdir/  byteps0  bpslaunch python3 byteps_imagenet.py --train-dir='/datadrive/home/ubuntu/ImageNet/train' --val-dir='/datadrive/home/ubuntu/ImageNet/val' --arch='vgg16' --epochs=80 --batch-size=64 --base-lr=0.0025 --cfreq=80"
        os.system(runcmd)

        ### clean
        #for i in range(numw):
        #    os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + worker_nodes[i] + " docker kill byteps")
        #    os.system("ssh -i /home/ubuntu/ray_bootstrap_key.pem  -o StrictHostKeyChecking=no ubuntu@" + worker_nodes[i] + " docker rm byteps")

ray.init(address="auto")
run()
