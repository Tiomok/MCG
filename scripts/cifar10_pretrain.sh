#!/bin/bash
xhidden=64
xsize=32
yhidden=256
depth=8
level=3
batchsize=64
epochs=40
testgap=100000
loggap=10
savegap=1000
infergap=100000
lr=0.001
grad_clip=0
grad_norm=10
regularizer=0
adv_loss=True
name='cifar10_train'
learn_top=False
only=True
tanh=False
clamp=True
#model_path='/home/chen/ghh/MCG-Blackbox/checkpoints/cifar10_target_models/ResNet18_ckpt.t7'
model_path=''
support_set='resnet18'
query_set='resnet18'
margin=5.0
down_sample_x=2
down_sample_y=2
logroot="./save/cifar10/"
train_dataset_root="./data/cifar10"
valid_dataset_root="./data/cifar10"

CUDA_VISIBLE_DEVICES=0 \
python train.py --dataset_name=cifar10 \
  --train_dataset_root=${train_dataset_root} --valid_dataset_root=${valid_dataset_root} \
  --log_root=${logroot} --x_hidden_channels=${xhidden} --y_hidden_channels=${yhidden} \
  --x_hidden_size=${xsize} --flow_depth=${depth} --num_levels=${level} --num_epochs=${epochs} --batch_size=${batchsize} \
  --test_gap=${testgap} --log_gap=${loggap} --inference_gap=${infergap} --lr=${lr} --max_grad_clip=${grad_clip} \
  --max_grad_norm=${grad_norm} --save_gap=${savegap} --regularizer=${regularizer} --adv_loss=${adv_loss} \
  --learn_top=${learn_top} --model_path=${model_path} --tanh=${tanh} --only=${only} --margin=${margin} --clamp=${clamp} \
  --name=${name} --support_set=${support_set} --query_set=${query_set} --down_sample_x 8 --down_sample_y 8 --meta_iteration=3 \
  --down_sample_x=${down_sample_x} --down_sample_y=${down_sample_y}
