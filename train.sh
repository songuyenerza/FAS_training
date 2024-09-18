# nproc_per_node=$(python3.8 -c "from config import ApplicationConfig as cfg; print(cfg.nproc_per_node)")
# echo "nproc_per_node: $nproc_per_node"
# python3.8 -m torch.distributed.run --nproc_per_node=$nproc_per_node --master_port=12356 \
#   train.py


CUDA_VISIBLE_DEVICES=0 python3.8 -m torch.distributed.run --nproc_per_node=1 --master_port=12355 train.py
