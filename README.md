# DRLCDR


CUDA_VISIBLE_DEVICES=0 nohup python train_rec.py --dataset cloth_phone --id cloth_phone_lv0d3 --num_epoch 500 --condi_weight 0.3 --condi_non_weight 10 --condi_condi_weight 10
