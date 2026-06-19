export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./temp \
  --model_id LEAD \
  --model Autoformer \
  --data LEAD \
  --features S \
  --seq_len 168 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 2 \
  --enc_in 1 \
  --c_out 1 \
  --anomaly_ratio 2.3 \
  --batch_size 32 \
  --train_epochs 2 
  # --learning_rate 0.0005 