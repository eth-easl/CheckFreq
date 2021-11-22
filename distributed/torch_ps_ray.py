from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from types import SimpleNamespace
from typing import overload

import numpy as np
import os
import ray
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as torchmodels
from filelock import FileLock
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
#import ray.services
#import boto3
import io
#from boto3.s3.transfer import TransferConfig
import threading
import sys
import random
import math
import time
#import botocore

import argparse
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock

import ctypes
import socket
#import vgg_torch
#import resnet
#import config_model
#import cause_node_fails
#from dist_chk import CFCheckpoint
from statistics import mean
import torchvision.datasets as datasets


### import issues! ###
import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/datadrive/home/ubuntu/CheckFreq/models/image_classification/config_model.py")
config_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_model)

spec1 = importlib.util.spec_from_file_location("module.name", "/datadrive/home/ubuntu/CheckFreq/distributed/dist_chk.py")
dist_chk = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(dist_chk)

spec2 = importlib.util.spec_from_file_location("module.name", "/datadrive/home/ubuntu/CheckFreq/distributed/data_loader.py")
data_loader = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(data_loader)


try:
		set_start_method('spawn')
except RuntimeError:
		pass


def get_pid(name):
    n = check_output(["pidof", name])
    n = n.decode("utf-8")
    n = n[:-1]
    return n



def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # This is only set to finish evaluation faster.
            # if batch_idx * len(data) > 1024:
            #    break
            # print("--- batch idx: ", batch_idx)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100. * correct / total


def seed_everything(seed=42):
    # torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


@ray.remote(num_gpus=1, num_cpus=1, max_restarts=5, max_task_retries=-1)
class Worker(object):
    def __init__(self, idx, batch_size, td, num_ps, num_worker, rocksdb_lat, model_name, dali=False):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print('CUBLAS: ', os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        seed_everything()

        self.model_name = model_name
        self.model = models.__dict__[self.model_name](num_classes=10) 
        #self.model = config_model.get_model_by_name(self.model_name)

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.batch_size = batch_size
        #torch.cuda.set_device(0)

        #self.train_loader = ray.get(
        #    td.get_data_loader.remote(batch_size, idx, num_worker))  # data_loader
        self.td = td
        self.idx = idx
        self.train_loader = td.get_data_loader(batch_size, idx, num_worker)
        self.data_iterator = iter(self.train_loader)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        #self.batch_set = list(enumerate(self.train_loader))
        self.grad = []
        self.dali = dali

        self.num_ps = num_ps
        self.num_workers = num_worker
        self.sleep_read = 0.0
        self.sleep_write = 0.0  # 0.7
        # self.sleep_write = 0.0 #0.5 #2.4
        self.rocksdb_lat = rocksdb_lat

    def ready(self):
        return

    def num_params(self):
        return len(self.get_weights())

    def params_distribution(self):
        distribution = []
        weights = self.get_weights()
        for k, v in weights.items():
            distribution.append(v.numel())
        return distribution

    @ray.method(num_returns=3)
    def compute_gradients(self, i, *params):

        start_time = time.time()
        it = i
        for p in params:
            it = min(it, p['it'])
            p.pop('it')

        if (it < i):  # a failure
            print("----------------------- a failure is detected! roll back to: ", it)
            return None, None, it

        if (it < 0):
            return self.grad, 0, it

        weights = self.stitch_parameters(*params)

        # compute gradients

        w_dict_torch = {}
        for k in weights.keys():
            w_dict_torch[k] = torch.from_numpy(weights[k])

        try:
            batch = next(self.data_iterator)
        except StopIteration:
            print("--------------------------------- Epoch ended!!!")
            if self.dali:
                self.data_iterator.reset() # TODO: maybe change seeds here? Is this the same for native data loader?
            else:
                self.train_loader = self.td.get_data_loader(self.batch_size, self.idx, self.num_workers)
                self.data_iterator = iter(self.train_loader)
            batch = next(self.data_iterator)

        if self.dali:
            x = batch[0]["data"]
            y = batch[0]["label"].squeeze().cuda().long()
        else:
            [x, y] = batch
            x, y = x.to(self.device), y.to(self.device)

        self.set_weights(w_dict_torch)
        # self.model.eval() - TODO: fix this with resnet and vgg

        self.model.zero_grad()
        #input_var = Variable(x)
        output = self.model(x)
        self.loss = self.criterion(output, y)

        self.loss.backward()

        self.grad = self.get_gradients()
        return self.grad, self.loss.cpu().data.numpy(), it




    def split_gradients(self, grad, assignments):
        if grad is None:
            num_shards = np.unique(np.array(assignments)).size
            return [None] * num_shards

        start = time.time()
        num_shards = np.unique(np.array(assignments)).size
        shards = [dict() for i in range(num_shards)]
        for i, (k, v) in enumerate(grad.items()):
            shards[assignments[i]][k] = v
        return shards

    def split_parameters(self, assignments):

        start = time.time()
        params = self.get_weights()
        num_shards = np.unique(np.array(assignments)).size
        shards = [dict() for i in range(num_shards)]
        for i, (k, v) in enumerate(params.items()):
            shards[assignments[i]][k] = v.data.cpu()
        return shards

    def index_shard(self, shards, index):
        return shards[index]

    def stitch_parameters(self, *split_params):
        start = time.time()
        params = dict()
        for p in split_params:
            for k, v in p.items():
                params[k] = v
        return params

    def get_state(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(weights[name])
        return True

    def get_weights(self):
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param
        return param_dict

    def get_gradients(self):
        grad_dict = {}
        for name, p in self.model.named_parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grad_dict[name] = grad
        return grad_dict


@ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
class PS(object):
    def __init__(self, idx, freq, num_ps, checkp_local, rocksdb_lat, model_name, synchronous):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print('CUBLAS: ', os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        seed_everything()

        ### for training
        self.start_epoch = 0
        self.reloaded = False
        self.ready = False
        self.model_name = model_name
        self.num_ps = num_ps
        self.it = -1

        self.params = None
        self.optimizer = None
        self.idx = idx

        ### for checkpointing
        self.checkp_local = checkp_local
        self.synchronous = synchronous
        self.chk = None # set later
        self.overwrite = True
        self.mp_manager = Manager()

        self.active_snapshot = Value('i', 0)
        self.lock = Lock()
        self.in_progress_snapshot = Value('i', 0)
        self.profile_snap = Value('i', 0)

        # Handle to the process performing checkpoint
        self.chk_process = None
        self.spawned = False
        self.use_thread = False

        self.iter = Value('i', 0)
        self.epoch = Value('i', 0)
        self.last_chk_it = Value('i', -1)

        self.change = Value('i', 0)					
        self.filepath = self.mp_manager.Value(ctypes.c_wchar_p, "") #self.mp_manager.dict()
        self.additional_snapshot = self.mp_manager.dict()

        self.dirpath = '/datadrive/home/ubuntu/CheckFreq/distributed/checkpoint/'
        if not os.path.isdir(self.dirpath):
            os.mkdir(self.dirpath)
        else:
                # delete contents for now
                for filename in os.listdir(self.dirpath):
                        file_path = os.path.join(self.dirpath, filename)
                        os.remove(file_path)

        print("start from: ", self.start_epoch, self.it)

    def reset(self):
        self.it = -1

    def get_params(self, it):
        p = self.params
        wdict = {}
        for k in p.keys():
            wdict[k] = p[k].cpu().detach().numpy()
        if (self.reloaded and not self.ready):
            self.ready = True
            it = self.it
            print("-------------- ready after reloading!, it is: ", it)
        wdict['it'] = it
        return wdict

    def get_model_weights(self, it):
        p = self.params
        wdict = {}
        for k in p.keys():
            wdict[k] = p[k]
        if (self.reloaded and not self.ready):
            self.ready = True
            it = self.it
            print("-------------- ready after reloading!, it is: ", it)
        return wdict

    def set_params(self, params):
        self.params = params
        if ('resnet' in self.model_name):
            self.optimizer = torch.optim.SGD(
                self.params.values(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = torch.optim.SGD(
                self.params.values(), lr=0.01, momentum=0.9)
        self.mw_dict = self.get_model_weights(0)
        self.mw_keys = list(self.mw_dict.keys())
        self.num_keys = len(self.mw_keys)
        return True

    def apply_updates(self, epoch, itr, *list_of_gradients):

        start_time = time.time()
        while self.in_progress_snapshot.value == 1:
            continue
        print("stall for snapshot took: ", time.time()-start_time)

        assert(len(list_of_gradients) >= 1)
        summed_gradient_dict = dict()

        for k in range(self.num_keys):
            name = self.mw_keys[k]
            t = torch.zeros(self.mw_dict[name].size())
            for i in range(len(list_of_gradients)):
                t += torch.from_numpy(list_of_gradients[i][name])
            summed_gradient_dict[name] = t

	

        self.optimizer.zero_grad()

        # print(summed_gradient_dict['linear.bias'])
        self._set_gradients(summed_gradient_dict)
        self.optimizer.step()

        self.it = itr

        end_time = time.time()

        print("apply updates took: ", end_time-start_time)

        return True

    def _set_gradients(self, gradients):
        # gradients should be a stitched dict
        for name, p in self.params.items():
            if gradients[name] is not None:
                p.grad = gradients[name]

    def make_shm(self, obj):
       if obj is None:
           return
       if torch.is_tensor(obj):
           obj.share_memory_()
       elif isinstance(obj, dict):
           for name, ref in obj.items(): 
              self.make_shm(ref)
       elif isinstance(obj, list):
           for x in obj:
              self.make_shm(x)
       else:
           return


    def store_checkpoint(self, epoch, it, profile_snap=False, profile_all=False):

        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

        print('Saving. Epoch:', epoch, ' , Iter: ', it)
        self.additional_snapshot['epoch'] = epoch
        self.additional_snapshot['iter'] = it

        start = time.time()
        self.filepath.value = self.dirpath + 'chk_' + str(self.idx) + '_' + str(it) + '.chk'
        
        skeys = list(self.optimizer.state_dict()['state'].keys())
        k = skeys[-1]
        print("---- from PS, MODEL: ", 'linear.weight', self.params['linear.weight'])
        #print("---- from PS, OPT: ", k, self.optimizer.state_dict()['state'][k])

        if self.chk is None:
            print("Is NONE!")
            for name,ref in self.params.items():
                #print(ref)
                ref.share_memory_()

            #print(self.optimizer.state_dict())

            for name,ref in self.optimizer.state_dict().items():
                print(name)
                self.make_shm(ref)


            self.chk = dist_chk.CFCheckpoint(model=self.params, optimizer=self.optimizer)
            print(self.chk)

        if self.synchronous:

            self.chk._serialize_and_persist(self.filepath,  self.active_snapshot, self.in_progress_snapshot, self.lock, 1, \
                                             self.additional_snapshot, background=False, snapshot_ready=False, iter_chk=self.last_chk_it, overwrite=True)
        else:
            # Check if there's an ongoing checkpoint operation 
            if self.chk_process is not None:
                while self.change.value==1:		
                    # this means a checkpoint is on progress (wait for process doing the checkpoint to set variable to 0)
                    continue

			# Once complete, initiate the next checkpoint synchronously
            with self.lock:
                self.in_progress_snapshot.value = 1

            if profile_snap:
                self.profile_snap.value = 1
            else:
                self.profile_snap.value = 0

            if self.use_thread:
                    fn = getattr(threading, 'Thread')
            else:
                fn = Process#globals()["Process"]	
            print("Function is {}".format(fn))

            print("self spanwned is: ", self.spawned)
            with self.lock:
                self.change.value = 1

            if not self.spawned:
                keywords = { \
						'snapshot_ready': False, \
                        'profile_snap': self.profile_snap, \
						'background': True, \
						'iter_chk':self.last_chk_it, \
						'overwrite':self.overwrite}
                self.chk_process = \
					fn(target=self.chk._serialize_and_persist,	\
						args=[self.filepath, self.active_snapshot, self.in_progress_snapshot, self.lock, self.change, self.additional_snapshot], kwargs=keywords)
                self.chk_process.start()
                self.spawned = True

            # wait for the checkpoint/snapshot to complete if needed
            if profile_snap or profile_all:
                while self.change.value==1:		
                    continue

        savetime = time.time()-start
        print("store checkpoint took: ", savetime, " sec")

    def load_checkpoint(self, checkpoint_path):
        #assert os.path.isdir(checkpoint_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint_path)
        self.params = checkpoint['net']
        self.set_params(self.params)
        self.optimizer.load_state_dict(checkpoint['opt'])
        self.epoch = checkpoint['epoch']
        self.it = checkpoint['iter']

    def join_proc(self):
        if self.chk_process is not None:
            if self.chk_process.is_alive():
                pid = self.chk_process.pid
                while (self.chk_process.is_alive()):
                    os.system("kill -9 " + str(pid))
                print("-------------- Killed!")



class PSStrategy(object):
    def __init__(self,
                 num_worker,
                 num_ps,
                 batch_size, freq, dl,
                 checkp_local, rocksdb_lat, model_name, collocated, 
                 sync, profile, prof_steps, dali):

        self.num_ps = num_ps
        self.num_worker = num_worker
        self.assignments = None
        self.freq = freq
        self.checkp_local = checkp_local
        self.rocksdb_lat = rocksdb_lat
        self.model_name = model_name
        self.synchronous = sync
        self.profile = profile
        self.prof_steps = prof_steps
        self.prof_done = False
        self.steps_since_checkp = 0
        self.monitor = False
        self.monitor_iter = 0
        self.iter_dur = []
        self.prev_iter_mean = 0
        self.max_overhead = 5
        self.dali = dali

        nodes_info = ray.nodes()
        cpu_nodes = ['node:' + n['NodeManagerAddress']
                     for n in nodes_info if not 'GPU' in n["Resources"].keys()]
        gpu_nodes = ['node:' + n['NodeManagerAddress']
                     for n in nodes_info if 'GPU' in n["Resources"].keys()]

        print(cpu_nodes, gpu_nodes)
        dw = []
        for n in gpu_nodes:
            dw.append(Worker.options(resources={n: 0.01}))

        ps = []
        if (collocated):
            for n in gpu_nodes:
                ps.append(PS.options(resources={n: 0.01}))
        else:
            for n in cpu_nodes:
                ps.append(PS.options(resources={n: 0.01}))

        self.workers = []
        for j in range(self.num_worker):
            idx = j
            self.workers.append(dw[j].remote(
                idx, batch_size, dl, self.num_ps, self.num_worker, self.rocksdb_lat, self.model_name, dali=self.dali))

        # spawn ps
        self.servers = []
        for j in range(self.num_ps):
            self.servers.append(ps[j].remote(
                j, freq, self.num_ps, self.checkp_local, self.rocksdb_lat, self.model_name, self.synchronous))

        self.initialize()

    def _round_robin_sharding(self):
        """Generate the assignment of variable to servers."""
        parameter_distribution = ray.get(
            self.workers[0].params_distribution.remote())
        assignments = [0 for _ in parameter_distribution]
        loads = [0 for _ in range(self.num_ps)]
        for i, var_size in enumerate(parameter_distribution):
            min_ps_index = loads.index(min(loads))
            loads[min_ps_index] += var_size
            assignments[i] = min_ps_index
        print("Load of each ps {}".format(loads))
        self.assignments = assignments

    def initialize(self):
        init_weights_id = self.workers[0].get_weights.remote()

        self._round_robin_sharding()

        for i, worker in enumerate(self.workers):
            if i != 0:
                ray.wait([worker.set_weights.remote(init_weights_id)])

        shard_ids = self.workers[0].split_parameters.remote(self.assignments)
        for i, server in enumerate(self.servers):
            this_shard_id = self.workers[0].index_shard.remote(shard_ids, i)
            ray.wait([server.set_params.remote(this_shard_id)])

    def get_weights(self):
        model_weights = ray.get(self.workers[0].get_state.remote())
        return model_weights

    def reset(self):
        ray.get([ps.reset.remote() for ps in self.servers])

    def clean(self):
        ray.get([ps.join_proc.remote() for ps in self.servers])

    def do_prof(self, epoch, itr):
        print(self.iter_dur)
        t_i = mean(self.iter_dur)

        ## first, do a simple checkpoint call to create the background processes
        ray.get([ps.store_checkpoint.remote(epoch, itr, profile_all=True) for ps in self.servers])

        ## now measure the time the snapshot takes
        start = time.time()
        ray.get([ps.store_checkpoint.remote(epoch, itr, profile_snap=True) for ps in self.servers])
        overhead = time.time()-start

        ## finally, measure the time the actual checkpoint (snapshot + persist) takes
        start = time.time()
        ray.get([ps.store_checkpoint.remote(epoch, itr, profile_all=True) for ps in self.servers])
        t_f = time.time()-start        

        ## Check Freq:
        chk_freq = max(math.ceil((t_f - overhead)/t_i), 1)
        print("t_i: ", t_i, " , overhead: ", overhead, " , t_f: ", t_f) 
        print("------------ CheckFreq found: ", chk_freq)
        self.steps_since_checkp = 0
        self.iter_dur = []

        # start monitoring
        self.monitor = True
        self.monitor_iter = 0
        self.prev_iter_mean = t_i
        return chk_freq

    def adapt_freq(self):
        cur_iter_mean = mean(self.iter_dur)
        cur_total = sum(self.iter_dur)
        old_total = self.prev_iter_mean * len(self.iter_dur)

        overhead_full = cur_total-old_total
        overhead_perc = 100 * overhead_full/old_total

        print("--------------- Iter mean new is: ", cur_iter_mean)
        print("--------------- Overhead is: ", overhead_perc)

        if overhead_perc > self.max_overhead:
            self.freq += 1
            print("-------------------------------- New Checkpoint Freq found: ", self.freq)

        self.iter_dir = []
        self.monitor_iter = 0 

    def step(self, epoch, it, nw):
        # stitch parameters

        startit = time.time()
        param_ids = [ps.get_params.remote(it) for ps in self.servers]
        # worker compute the grads
        ps_grad_mappings = [list() for i in range(self.num_ps)]
        loss_vals = []
        for worker in self.workers[:nw]:
            grad_id, loss, itr = worker.compute_gradients.remote(
                it, *param_ids)  # stitched_param_id, it)
            loss_vals.append(loss)
            split_gradient_ids = worker.split_gradients.remote(
                grad_id, self.assignments)
            for i in range(self.num_ps):
                this_shard_id = worker.index_shard.remote(
                    split_gradient_ids, i)
                ps_grad_mappings[i].append(this_shard_id)

        ret = [ps.apply_updates.remote(
            epoch, itr, *ps_grad_mappings[i]) for i, ps in enumerate(self.servers)]

        print("----------------------------------------------------------------------------------")
        ray.get(ret)
        endit = time.time()
        check_time=0
        if (self.profile and (not self.prof_done) and it >= 5 and it < self.prof_steps): # collect statistics for the iteration time
            print("---------- Profile step: ", it)
            self.iter_dur.append(endit-startit)

        elif (self.profile and (not self.prof_done) and it == self.prof_steps):
            # do prof
            print("---------- Complete profiling")
            self.freq = self.do_prof(epoch, it)
            self.prof_done = True

        else:
            start = time.time()
            if (self.freq > 0 and self.steps_since_checkp == self.freq):
                ray.get([ps.store_checkpoint.remote(epoch, itr)
                        for ps in self.servers])
                check_time = time.time() - start
                self.steps_since_checkp=1
            elif (self.freq > 0):
                self.steps_since_checkp += 1 
        
        if self.profile and self.prof_done:
            if not self.monitor:
                self.monitor = True
            if self.monitor:
                self.monitor_iter += 1
                if self.monitor_iter <= self.freq:
                    self.iter_dur.append(time.time()-startit)
                else:
                    self.adapt_freq()
        print("Iteration took: ", time.time()-startit)
        return ray.get(itr)+1, check_time


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker processes')
    parser.add_argument('--num_servers', type=int, default=1,
                        help='Number of server processes')
    parser.add_argument('--worker_batch_size', type=int,
                        default=32, help='Per-worker minibatch size')
    parser.add_argument('--remote_check',  type=bool, default=False,
                        help='Whether to checkpoint locally or remotely (only S3 is supported for now)')
    parser.add_argument('--check_freq_iters', type=int, default=0,
                        help='Number of iterations between two consecutive checkpoints')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--num_kills', type=int, default=0,
                        help='Number of processes to kill')
    parser.add_argument('--rocksdb_lat', type=bool,
                        default=False, help='Whether or not to inject latency')
    parser.add_argument('--model', type=str, default='resnet101',
                        help='Which CV model to use. Currently supporting ResNet and VGG')
    parser.add_argument('--collocated', type=bool, default=False,
                        help='Whether to colocate servers and workers')
    parser.add_argument('--sync', type=bool, default=False,
                        help='Whether checkpoint is synchronous or not')
    parser.add_argument('--prof_steps', type=int, default=100,
                        help='Number of profling steps')
    parser.add_argument('--profile', type=bool, default=False,
                        help='Whether to profile for finding the checkpoint frequency or not') 
    parser.add_argument('--use_dali', type=bool, default=False,
                        help='Whether to use the DALI (CoorDL) library for data preprocessing') 
    parser.add_argument('--data_path', type=str, default="",
                        help='Path to dataset. It must have subdirectories named "train" and "val";')                   
    args = parser.parse_args()

    print(args)
    num_workers = args.num_workers
    num_servers = args.num_servers
    check_freq_iters = args.check_freq_iters
    train_batch_size = args.worker_batch_size
    local_check = not args.remote_check
    epochs = args.epochs
    num_kills = args.num_kills
    rocksdb_lat = args.rocksdb_lat
    model_name = args.model
    collocated = args.collocated
    sync = args.sync
    prof_steps = args.prof_steps
    profile  = args.profile
    use_dali = args.use_dali
    data_path = args.data_path

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    ray.init(address="auto")

    res = ray.cluster_resources()
    res_keys = res.keys()

    local_hostname = socket.gethostbyname(socket.gethostname())
    driver_node_id = 'node:'+local_hostname
    print(driver_node_id)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed_everything()
    #model = config_model.get_model_by_name(model_name)
    model = models.__dict__[model_name](num_classes=10)

    train_size = 50000  # for CIFAR10
    iter_per_epoch = math.ceil(train_size/train_batch_size)

    it = 0
    #loader1 = data_loader.td_loader.options(resources={driver_node_id: 0.01})
    ld=None

    if not use_dali:
        #worker_data, validation_dataset = data_loader.split_data(num_workers)
        ld = data_loader.td_loader(train_dir=traindir)
    else:
        ld = data_loader.td_loader(dali=True, train_dir=traindir)
  
    print(ld)

    workers = []

    strategy = PSStrategy(
        num_worker=num_workers,
        num_ps=num_servers,
        batch_size=train_batch_size, freq=check_freq_iters,
        dl=ld, checkp_local=local_check, rocksdb_lat=rocksdb_lat, model_name=model_name, collocated=collocated, 
        sync=sync, profile=profile, prof_steps=prof_steps, dali=use_dali)

    training_time = 0
    e_iters = math.floor(iter_per_epoch/num_workers)

    print("------- train for: ", e_iters, " iterations")
    max_iters = math.ceil(iter_per_epoch/num_workers)

    start = time.time()
    last = start
    killed = 0

    time_int = 50  # time between consecutive fails (in sec)

    # for evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #test_loader = data_loader.get_val_data_loader(validation_dataset) # TODO: FIX validation!!!!!!

    check_time = 0
    j = 0
    for i in range(epochs):
        strategy.reset()
        while j < e_iters:
            print("Iteration: ", j)
            j, chtime = strategy.step(i, j, num_workers)
            check_time += chtime

            if (killed < num_kills):
                if ((time.time() - last) >= time_int):
                    # remove a node (make sure it is not the driver)
                    cause_node_fails.remove_node(local_hostname)

                    killed += 1
                    last = time.time()

        rem = iter_per_epoch - e_iters*num_workers
        print("-------------- remaining iterations: ", rem)

        if (rem > 0):
            strategy.step(i, j, rem)

        j = 0

        current_weights = strategy.get_weights()
        training_time += (time.time()-start)
        tt = time.time()-start
        #model.set_weights(current_weights)
        start_time = time.time()
        accuracy = 0 #evaluate(model, test_loader)
        end_time = time.time()
        th = train_size/tt
        w = "Freq: " + str(check_freq_iters) + ", Num workers: " + str(num_workers) + ", Num servers: " + str(num_servers) + " , accuracy: " + \
            str(accuracy) + " , time: " + str(training_time) + " , killed: " + \
            str(killed) + " , check time: " + str(check_time) + "\n"
        start = time.time()

    strategy.clean()
    f = open('/datadrive/home/ubuntu/CheckFreq/distributed/output.txt', 'a')
    wr = "Iters: " + str(e_iters) + ", Freq: " + str(check_freq_iters) + ", Num workers: " + str(num_workers) + ", Num servers: " + str(num_servers) + \
        " , accuracy: " + str(accuracy) + " , time: " + str(training_time) + \
        " , killed: " + str(killed) + " , check time: " + \
        str(check_time) + "\n"
    f.write(wr)
    f.close()
    print("Final accuracy is {:.1f}.".format(accuracy))


if __name__ == "__main__":
    main()
