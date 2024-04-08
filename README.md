
# Face_iso

1. Download dataset to XX and modify "./isot/config/env.yml": 
   data_dir: XX/data/ 
   result_dir: XX/result/ 
2. Train model:
   python train.py --dataset lfw_gender  --save --lr_scheduler --epoch 50 --lr 0.001 --momentum 0.9 --weight_decay 0.0005  --batch_size 16  --model alexnet
3. Auditing:
   python audit_iso.py --dataset lfw_gender  --attack iso_boundary_simplified --epoch 10 --lr 0.001 --save --verbose  --model alexnet --batch_size 16  --pretrain 