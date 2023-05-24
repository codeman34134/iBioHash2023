CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  src/main_ddp.py \
  --source_path /home/hdd/iBioHash_Train \
  --loss recallatk \
  --dataset Inaturalist \
  --mixup 1 \
  --samples_per_class 4 \
  --embed_dim 48 \
  --fc_lr_mul 0 \
  --arch SwinL \
