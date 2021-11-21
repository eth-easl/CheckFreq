import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import ray

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

def split_data(n_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(
        root="~/data", train=True, download=True, transform=transform_train)
    validation_dataset = CIFAR10(
        root="~/data", train=False, download=True, transform=transform_test)

    temp = tuple([len(train_dataset)//n_workers for i in range(n_workers)])
    print(temp)
    worker_data = torch.utils.data.random_split(
        train_dataset, temp) #generator=torch.Generator().manual_seed(42)) # Comment for running with PyTorch 1.1
    print(len(worker_data))
    return worker_data, validation_dataset


def get_train_data_loader(worker_data, train_batch_size, idx):

    print(len(worker_data), idx)
    train_loader = DataLoader(
        worker_data[idx], batch_size=train_batch_size, shuffle=True, num_workers=0, worker_init_fn=random.seed(42))  # should add num_workers here?
    return train_loader


def get_val_data_loader(validation_dataset):
    validation_loader = DataLoader(
        validation_dataset, batch_size=100, shuffle=False, num_workers=0, worker_init_fn=random.seed(42))
    return validation_loader


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, local_rank, world_size, node_rank=0, nnodes=1, dali_cpu=False, resume_index=0, resume_epoch=0):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        shard = int(node_rank*world_size/nnodes + local_rank)
        #if args.mint:
        #    self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=args.world_size, shuffle_after_epoch=True, cache_size=args.cache_size)
        #else:
        cf_det=True
        #if not resume_index and not resume_epoch:
        #    cf_det=False
        #    self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=world_size, shuffle_after_epoch=True)
        #else:
        # TODO: check if this is what we want:
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


@ray.remote
class td_loader(object):
    def __init__(self, worker_data=None, train_dir=None, dali=False):
        self.worker_data = worker_data
        self.train_dir = train_dir
        self.dali = dali

    def get_data_loader(self, train_batch_size, idx, world_size):
        if self.dali:
            crop_size = 224
            val_size = 256
            pipe = HybridTrainPipe(batch_size=train_batch_size, num_threads=0, device_id=idx, data_dir=self.train_dir, crop=crop_size)
            #resume_size = int(pipe.epoch_size("Reader") / world_size) - args.start_index
            start_index = 0 # TODO: fix this for resume!!
            resume_size = int(pipe.epoch_size("Reader") / world_size) - start_index
            train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / world_size), fill_last_batch=False, resume_size=resume_size)
        else:
            train_loader = DataLoader(self.worker_data[idx], batch_size=train_batch_size, shuffle=True,
                                        num_workers=0, worker_init_fn=random.seed(42))  # should add num_workers here?
        return train_loader

