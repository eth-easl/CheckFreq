#!/bin/bash
set -e

## Use torch.distributed
# python -m torch.distributed.launch --nproc_per_node=4 models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 3  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --data /datadrive/cifar > stdout.out 2>&1

## Use checkfreq
# python models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 1  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --data /datadrive/cifar > stdout.out 2>&1 

res_base=/datadrive/home/kaiszhang/results

# for freq in {0,10,20,40,80}; do
#   echo "frequency: $freq"
#   res_dir=$res_base/chk-$freq
#   mkdir -p $res_dir
#   # python models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 1  --deterministic --noeval --barrier --checkfreq --chk-freq $freq --chk_mode_baseline --chk-prefix ./chk/ --cf_iterator --data /datadrive/cifar > stdout.out 2>&1 
#   python -m torch.distributed.launch --nproc_per_node=4 models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64  --workers 3 --epochs 3  --deterministic --noeval --barrier --checkfreq --chk-freq $freq --chk_mode_baseline --chk-prefix ./chk/ --cf_iterator --data /datadrive/cifar > stdout.out 2>&1

#   mv stdout.out time-split*.csv acc-progress*.csv $res_dir
# done

for ovh in {0,1,5,10,15,20}; do
  echo "overhead: $ovh"
  res_dir=$res_base/ovh-$ovh
  mkdir -p $res_dir

  python -m torch.distributed.launch --nproc_per_node=4 models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 3  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --overhead $ovh --data /datadrive/cifar > stdout.out 2>&1

  mv stdout.out time-split*.csv acc-progress*.csv $res_dir
done
