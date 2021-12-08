from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import byteps.torch as bps
import tensorboardX
import os
import math
from tqdm import tqdm
import time
import ctypes 
import sys
from statistics import mean
import glob
import logging

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

sys.path.append('../')
import dist_chk_bps

from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, freeze_support, spawn

'''
try:
    #ctx = multiprocessing.get_context("spawn")
    set_start_method('spawn')
except RuntimeError:
    print("---- error!")
    pass
'''

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}-{it}.chk',
                    help='checkpoint file format')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during pushpull')
parser.add_argument('--batches-per-pushpull', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing pushpull across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture') 
# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--cfreq', type=int, default=-1,
                    help='checkpoint frequency')

parser.add_argument('--sync', type=bool, default=False,
                    help='whether to do synchronous checkpoint or not')

parser.add_argument('--profile', type=bool, default=False,
                    help='whether to profile or not according to CheckFreq')

parser.add_argument('--prof-steps', type=int, default=50,
                    help='number of steps to profile for')

parser.add_argument('--max-overhead', type=int, default=5,
                    help='overhead (%) of checkpointing over the total execution')

parser.add_argument('--dali', type=bool, default=False,
                    help='whether to use the (CoorDL) DALI library for data loading or not')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

pushpull_batch_size = args.batch_size * args.batches_per_pushpull


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, resume_index=0, resume_epoch=0):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        world_size = bps.size()
        nnodes = 1

        #TODO: fix local and node rank when >1 GPUs are used per node
        local_rank = bps.rank()
        node_rank = 0

        shard = int(node_rank*world_size/nnodes + local_rank)
        print(shard, local_rank, world_size)
        # if args.mint:
        #     self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=args.world_size, shuffle_after_epoch=True, cache_size=args.cache_size)
        # else:
        cf_det=True
        #if not resume_index and not resume_epoch:
        #    cf_det=False
        #    self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=world_size, shuffle_after_epoch=True)
        #else:
        self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=world_size, shuffle_after_epoch=True, resume_index=resume_index, resume_epoch=resume_epoch, cf_det=cf_det)
            
        print("CF deterministic shuffling is {}".format(cf_det))


        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        #decoder_device = 'cpu' 
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.4914 * 255, 0.4822 * 255, 0.4465 * 255],
                                            std=[0.2023 * 255, 0.1994 * 255, 0.2010 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        world_size=bps.size()
        node_rank = 0
        local_rank = bps.rank()
        nnodes=1

        shard = int(node_rank*world_size/nnodes + local_rank)
        self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.res = ops.Resize(device="cpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.4914 * 255, 0.4822 * 255, 0.4465 * 255],
                                            std=[0.2023 * 255, 0.1994 * 255, 0.2010 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]



def main():  

    bps.init()

    print("--- bps size: ", bps.size())
    torch.manual_seed(args.seed)

    if args.cuda:
        # BytePS: pin GPU to local rank.
        torch.cuda.set_device(bps.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    logging.basicConfig(filename='log_file', filemode='a', level=logging.INFO)

    # BytePS: print logs on the first worker.
    verbose = 1 if bps.rank() == 0 else 0

    # BytePS: write TensorBoard logs on first worker.
    log_writer = tensorboardX.SummaryWriter(args.log_dir) if bps.rank() == 0 else None

    if args.dali:
        print("Use (CoorDL) Dali library")
        crop_size=224
        val_size = 256
        did = bps.rank() % 4 # nnodes
        print("------- device id: ", did)
        pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=4, device_id=did, data_dir=args.train_dir, crop=crop_size, dali_cpu=0)
        pipe.build()
        print("-------------- Reader size: ", pipe.epoch_size("Reader"))
        resume_size = int(pipe.epoch_size("Reader") / bps.size()) # TODO: change this in order to reload
        train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / bps.size()), fill_last_batch=False, resume_size=resume_size)         
        pipe_val = HybridValPipe(batch_size=args.batch_size, num_threads=4, device_id=did, data_dir=args.val_dir, crop=crop_size, size=val_size)
        pipe_val.build()
        val_loader = DALIClassificationIterator(pipe_val, size=int(pipe_val.epoch_size("Reader") / bps.size()))
        train_sampler = None
    else:
        print("Use the native data loader")
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
        train_dataset = \
            datasets.ImageFolder(args.train_dir,
                                    transform=transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2023, 0.1994, 0.2010])
                                ]))
            # BytePS: use DistributedSampler to partition data among workers. Manually specify
            # `num_replicas=bps.size()` and `rank=bps.rank()`.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=bps.size(), rank=bps.rank())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=pushpull_batch_size,  #shuffle=True, **kwargs)
                sampler=train_sampler, **kwargs)

        val_dataset = \
            datasets.ImageFolder(args.val_dir,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                std=[0.2023, 0.1994, 0.2010])
                                ]))
        val_sampler = torch.utils.data.distributed.DistributedSampler(
                        val_dataset, num_replicas=bps.size(), rank=bps.rank())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, #shuffle=False, **kwargs)
                                                    sampler=val_sampler, **kwargs)


    ##########################################################################################################################################
    # get model
    model = models.__dict__[args.arch](num_classes=10)

    if args.cuda:
        # Move model to GPU.
        model.cuda()

    # BytePS: scale learning rate by the number of GPUs.
    # Gradient Accumulation: scale learning rate by batches_per_pushpull
    optimizer = optim.SGD(model.parameters(),
                        lr=(args.base_lr *
                            args.batches_per_pushpull * bps.size()),
                        momentum=args.momentum, weight_decay=args.wd)

    # BytePS: (optional) compression algorithm.
    compression = bps.Compression.fp16 if args.fp16_pushpull else bps.Compression.none

    # BytePS: wrap optimizer with DistributedOptimizer.

    optimizer = bps.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_pushpull)


    # Restore from the latest (valid) checkpoint 
    # BytePS: restore on the first worker which will broadcast weights to other workers.
    start_epoch = 0
    start_it = 0
   
    ## for logging time
    ttotal=0
    tstore=0
    tload=0

    start_load = time.time()
    if bps.rank() == 0:
        dirpath = '/datadrive/home/ubuntu/CheckFreq/distributed/bytepsdir/'
        files = filter( os.path.isfile, glob.glob(dirpath + '*.chk'))
        files = sorted(files, key=os.path.getmtime) 
        if (len(files) > 0): 
           try:
              start_epoch, start_it = load_checkpoint(files[-1], model, optimizer)
           except:
              start_epoch, start_it = load_checkpoint(files[-2], model, optimizer)

           start_it += 1 # TODO; fix this for end-of-epoch checkpoints
    tload += time.time()-start_load
 	
    params = model.state_dict()
    params['it'] = torch.tensor(start_it, dtype=torch.int32)
    params['epoch'] = torch.tensor(start_epoch, dtype=torch.int32)
    
    print(bps.rank(), params['it'], params['epoch'])

    # BytePS: broadcast parameters & optimizer state.
    bps.broadcast_parameters(params, root_rank=0)
    bps.broadcast_optimizer_state(optimizer, root_rank=0)

    #print(model.state_dict())
    start_it = params['it'].item()
    start_epoch = params['epoch'].item()

    print(bps.rank(), start_it, start_epoch)
    params.pop('it')
    params.pop('epoch')

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(params[name])

    #print(model.state_dict())
    #print(train_loader[0])
    train_it = enumerate(train_loader)
    #print(len(train_it))

    ## for checkpointing, according to checkfreq ##
    # the checkpoint class
    
    for name,ref in model.state_dict().items():
        make_shm(ref)

    for name,ref in optimizer.state_dict().items():
        make_shm(ref)
    
    # are these needed?


    chk = dist_chk_bps.CFCheckpoint(model=model.state_dict(), optimizer=optimizer.state_dict())
    print(chk)
    chk_process = None
    spawned = False
    overwrite = True

    active_snapshot = Value('i', 0)
    lock = Lock()
    in_progress_snapshot = Value('i', 0)
    profile_snap = Value('i', 0)

    mp_manager = Manager()
    iter = Value('i', 0)
    epoch = Value('i', 0)
    last_chk_it = Value('i', -1)

    change = Value('i', 0)					
    filepath = mp_manager.Value(ctypes.c_wchar_p, "") #self.mp_manager.dict()
    additional_snapshot = mp_manager.dict()

    timings=[]
    for epoch in range(start_epoch, args.epochs):
        if (epoch==start_epoch):
            iteration = start_it
        else:
            iteration = 0
        
        start = time.time()
        train(epoch, tload, iteration, model, optimizer, train_sampler, train_loader, verbose, in_progress_snapshot, log_writer, \
                 filepath, additional_snapshot, chk, active_snapshot, lock, last_chk_it, change, profile_snap)
        print("Epoch ", epoch, " took: ",time.time()-start)
        timings.append(time.time()-start)
        validate(epoch,  model, val_loader, verbose, log_writer)
        #save_checkpoint(epoch)

    print(timings)


def train(epoch, tload, iteration, model, optimizer, train_sampler, train_loader, verbose, in_progress_snapshot, log_writer, \
            filepath, additional_snapshot, chk, active_snapshot, lock, last_chk_it, change, \
            profile_snap):
    print("--------------- Train at epoch ", epoch)
    model.train()

    ttotal = 0
    tstore = 0
    if args.dali:
        train_loader.reset()
        #val_loader.reset()
    else:
        # TODO: is this to restart the train loader? check this
        train_sampler.set_epoch(epoch)

    train_iter = enumerate(train_loader)

    for _ in range(iteration):
        next(train_iter)

    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    iter_dur = []
    prof_done = False
    start = time.time()
    cfreq = args.cfreq

    skip_iter = False
    steps_since_checkp = 0
    monitor = False
    monitor_iter = 0
    mean_it_time = 0

    if args.dali:
       train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
    else:
       train_loader_len = len(train_loader)

    train_loader_len -= iteration
    print("Len: ", train_loader_len)

    with tqdm(total=train_loader_len,
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        it = iteration
        for batch_idx, batch in train_iter:
            #adjust_learning_rate(epoch, batch_idx)

            if skip_iter:
                skip_iter = False

            if args.cuda:
                if args.dali:
                    data = batch[0]["data"]
                    target = batch[0]["label"].squeeze().cuda().long()
                else:
                    data, target = batch[0].cuda(), batch[1].cuda()
            #print(data, target)
            # TODO: could this affect checkpoint?
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                #print(output, target)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss.item())
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            if (bps.rank()==0):
                ## wait for fine-grained chec
                start_time = time.time()
                while in_progress_snapshot.value == 1:
                    continue
                tstore += time.time()-start_time
                print("stall for snapshot took: ", time.time()-start_time)
            optimizer.step()

            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})

            if (bps.rank()==0):

                if (bps.rank()==0 and args.profile and epoch==0 and it >=5 and it < args.prof_steps):
                    print("---------- Profile step: ", it)
                    iter_dur.append(time.time()-start)

                elif (args.profile and not prof_done and epoch==0 and it == args.prof_steps):
                    print("---------- Complete profiling")
                    mean_it_time, cfreq = do_prof(iter_dur,filepath, model, optimizer, additional_snapshot, chk, active_snapshot, \
                                    in_progress_snapshot, lock, epoch, it, last_chk_it, change, profile_snap)
                    prof_done = True
                    skip_iter = True
                    steps_since_checkp=0
                    iter_dur = []

                else:
                    if (cfreq > 0 and steps_since_checkp == cfreq):
                        chk_start = time.time()
                        save_checkpoint(filepath, model, optimizer, additional_snapshot, chk, active_snapshot, in_progress_snapshot, lock, \
                                            epoch, it, last_chk_it, change, profile_snap, sync=args.sync)
                        steps_since_checkp=1
                        tstore += time.time()-chk_start
                    elif (cfreq > 0):
                        steps_since_checkp+=1
                
                if args.profile and prof_done and epoch==0:
                    if not monitor:
                        monitor=True
                    if monitor and not skip_iter:
                        monitor_iter += 1
                        if monitor_iter <= cfreq:
                            iter_dur.append(time.time()-start)
                        else:
                            cfreq = adapt_freq(iter_dur, mean_it_time, cfreq)
                            iter_dur = []
                            monitor_iter = 0 
                   
            it_time = time.time()-start
            ttotal += it_time
            
            if bps.rank() == 0:
            	print("it: ", it, " time: ", it_time, len(data))
            	logging.info("it: %s, ts: %s, total time: %.3f, store time: %.3f, load time: %.3f" % (it, time.time(), ttotal, tstore, tload))
            t.update(1)
            start = time.time()
            it += 1


        # do a synchronous checkpoint at the end of the epoch
        if bps.rank() == 0:
             save_checkpoint(filepath, model, optimizer, additional_snapshot, chk, active_snapshot, in_progress_snapshot, lock, \
                                    epoch, it, last_chk_it, change, profile_snap, sync=True)
        steps_since_checkp = 0

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch, model, val_loader, verbose, log_writer):
    model.eval()
    # TODO: need to reset the val loader?
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    if args.dali:
       val_loader_len = int(math.ceil(val_loader._size / args.batch_size))
    else:
       val_loader_len = len(val_loader)

    with tqdm(total=val_loader_len,
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for batch in val_loader:
                #print(data, target)
                if args.cuda:
                    if args.dali:
                        data = batch[0]["data"]
                        target = batch[0]["label"].squeeze().cuda().long()
                    else:
                        data, target = batch[0].cuda(), batch[1].cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                #print(output, target)
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch) 
     
    print("val/loss: ", val_loss.avg, epoch)
    print("val/accuracy: ", val_accuracy.avg, epoch)
 


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ep = checkpoint['epoch']
    iteration = checkpoint['iter']
    return ep, iteration

# BytePS: using `lr = base_lr * bps.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * bps.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, train_loader, optimizer):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / bps.size() * (epoch * (bps.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * bps.size() * args.batches_per_pushpull * lr_adj


def make_shm(obj):
    if obj is None:
        return
    if torch.is_tensor(obj):
        obj.share_memory_()
    elif isinstance(obj, dict):
        for name, ref in obj.items(): 
            make_shm(ref)
    elif isinstance(obj, list):
        for x in obj:
            make_shm(x)
    else:
        return

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()

def hello():
    print("hello")

def do_prof(iter_dur,filepath, model, optimizer, additional_snapshot, chk, active_snapshot, \
                in_progress_snapshot, lock, epoch, it, last_chk_it, change, profile_snap):

    print(iter_dur)
    t_i = mean(iter_dur)

    ## first, do a simple checkpoint call to create the background processes
    save_checkpoint(filepath, model,optimizer, additional_snapshot, chk, active_snapshot, in_progress_snapshot, lock, \
                        epoch, it, last_chk_it, change, profile_snap, prof_all=True)

    ## now measure the time the snapshot takes
    start = time.time()
    save_checkpoint(filepath, model,optimizer, additional_snapshot, chk, active_snapshot, in_progress_snapshot, lock, \
                        epoch, it, last_chk_it, change, profile_snap, prof_snap=True)
    overhead = time.time()-start

    ## finally, measure the time the actual checkpoint (snapshot + persist) takes
    start = time.time()
    save_checkpoint(filepath, model,optimizer, additional_snapshot, chk, active_snapshot, in_progress_snapshot, lock, \
                        epoch, it, last_chk_it, change, profile_snap, prof_all=True)
    t_f = time.time()-start        

    ## Check Freq:
    chk_freq = max(math.ceil((t_f - overhead)/t_i), 1)
    print("t_i: ", t_i, " , overhead: ", overhead, " , t_f: ", t_f) 
    print("------------ CheckFreq found: ", chk_freq)

    return t_i, chk_freq


def adapt_freq(iter_dur, mean_it_time, cfreq):
    cur_iter_mean = mean(iter_dur)
    cur_total = sum(iter_dur)
    old_total = mean_it_time * len(iter_dur)

    overhead_full = cur_total-old_total
    overhead_perc = 100 * overhead_full/old_total

    print("--------------- Iter mean new is: ", cur_iter_mean)
    print("--------------- Overhead is: ", overhead_perc)

    if overhead_perc > args.max_overhead:
        cfreq += 2
        print("-------------------------------- New Checkpoint Freq found: ", cfreq)

    return cfreq
        

def save_checkpoint(filepath, model, optimizer, additional_snapshot, chk, active_snapshot, in_progress_snapshot, lock, \
                        epoch, it, last_chk_it, change, profile_snap,  sync=False, prof_snap=False, prof_all=False):

    '''
    d = {"model": model.state_dict(), "opt": optimizer.state_dict()}
    torch.save(d, 'test.chk')
    f = open('test.chk', 'a+')
    os.fsync(f.fileno())
    f.close()
    '''
    start = time.time()
    filepath.value = args.checkpoint_format.format(epoch=epoch, it=it)
    additional_snapshot['epoch'] = epoch
    additional_snapshot['iter'] = it
    print(chk.chk_process)

    #print("----------- FROM WORKER, MODEL: fc.bias: ", model.state_dict()['fc.bias'])
    #skeys = list(optimizer.state_dict()['state'].keys())
    #k = skeys[-1]
    #print("---- from WORKER, OPT: ", k, optimizer.state_dict()['state'][k])

    if sync:
        chk._serialize_and_persist(filepath,  active_snapshot, in_progress_snapshot, lock, 1, \
                additional_snapshot, background=False, snapshot_ready=False, iter_chk=last_chk_it, overwrite=True)
    else:
        if chk.chk_process is not None:
            while change.value==1:		
                # this means a checkpoint is on progress (wait for process doing the checkpoint to set variable to 0)
                continue

		# Once complete, initiate the next checkpoint synchronously
        with lock:
                in_progress_snapshot.value = 1

        if prof_snap:
                profile_snap.value = 1
        else:
                profile_snap.value = 0

        fn = Process #globals()["Process"]	

        with lock:
                change.value = 1

    
    
        if not chk.spawned:
            print("------------- START A NEW PROCESS!! ------------")
            keywords = { \
					'snapshot_ready': False, \
                    'profile_snap': profile_snap, \
					'background': True, \
					'iter_chk':last_chk_it, \
					'overwrite':True}
            chk.chk_process = \
					fn(target=chk._serialize_and_persist,	\
						args=[filepath, active_snapshot, in_progress_snapshot, lock, change, additional_snapshot], kwargs=keywords)
            freeze_support()
            chk.chk_process.start()
            chk.spawned = True

        # wait for the checkpoint/snapshot to complete if needed
        if prof_snap or prof_all or sync:
            while change.value==1:		
                continue
    
    print("store checkp took: ", time.time() - start)  

# BytePS: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

if __name__ == '__main__':


    #freeze_support()
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        print("------------ error!")
    pass
    main()

  
