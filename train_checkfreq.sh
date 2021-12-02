#!/bin/bash
set -e
set -x

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

# for ovh in {0,1,5,10,15,20}; do
#   echo "overhead: $ovh"
#   res_dir=$res_base/ovh-$ovh
#   mkdir -p $res_dir
# 
#   python -m torch.distributed.launch --nproc_per_node=4 models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 3  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --overhead $ovh --data /datadrive/cifar > stdout.out 2>&1
# 
#   mv stdout.out time-split*.csv acc-progress*.csv $res_dir
# done

echo "opt, 100 interrupt, checkfreq"
res_dir=$res_base/opt-100-checkfreq
mkdir -p $res_dir
python crashes/proc_crashes.py --interval 100 -- python -m torch.distributed.launch --nproc_per_node=4 models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 3  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --data /datadrive/cifar --resume --print-freq 1
mv *.csv stdout.* stderr.* $res_dir
rm -rf *.log chk

for freq in {2,19,40}; do
  echo "opt, 100 interrupt, frequency: $freq"
  res_dir=$res_base/opt-100-$freq
  mkdir -p $res_dir
  python crashes/proc_crashes.py --interval 100 -- python -m torch.distributed.launch --nproc_per_node=4 models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 3  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --data /datadrive/cifar --resume --print-freq 1 --chk-freq $freq --chk_mode_baseline
  mv *.csv stdout.* stderr.* $res_dir
  rm -rf *.log chk
done

for freq in {0,2,19,40}; do
  echo "sync, 100 interrupt, frequency: $freq"
  res_dir=$res_base/sync-100-$freq
  mkdir -p $res_dir
  python crashes/proc_crashes.py --interval 100 -- python -m torch.distributed.launch --nproc_per_node=4 models/image_classification/pytorch-imagenet-cf.py --dali -a vgg16 -b 64 --workers 3 --epochs 3  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --data /datadrive/cifar --resume --print-freq 1 --chk-freq $freq --chk_mode_baseline --noop
  mv *.csv stdout.* stderr.* $res_dir
  rm -rf *.log chk
done

