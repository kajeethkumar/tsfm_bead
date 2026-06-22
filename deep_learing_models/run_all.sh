#!/bin/bash

# ======================================================
# Models to evaluate
# ======================================================
MODELS=(
    "Crossformer"
    "Informer"
    "KANDAD"
    "Nonstationary_Transformer"
    "iTransformer"
)

# ======================================================
# Get valid building IDs
# ======================================================
BUILDINGS=$(python - <<'PY'
import pandas as pd

df = pd.read_csv('../../dataset/train.csv')

df1 = pd.read_csv('../../dataset/train_features.csv')
valid_buildings = set(df1['building_id'].unique())

for bid in sorted(df['building_id'].unique()):
    if bid in valid_buildings:
        print(int(bid))
PY
)

mkdir -p temp

# ======================================================
# Run all models on all buildings
# ======================================================
for MODEL in "${MODELS[@]}"
do
    echo "======================================"
    echo "Running model: ${MODEL}"
    echo "======================================"

    for BID in ${BUILDINGS}
    
    do
        echo "Processing Building ${BID}"

        python - <<PY
import pandas as pd
import os

b_id = ${BID}

df = pd.read_csv('../../dataset/train.csv')

df1 = pd.read_csv('../../dataset/train_features.csv')
valid_buildings = df1['building_id'].unique()
df = df[df['building_id'].isin(valid_buildings)]

df_b = df[df['building_id'] == b_id].copy()

if len(df_b) == 0:
    raise ValueError(f"No data found for building {b_id}")

actual_ratio = df_b['anomaly'].mean() * 100
print(f"Building {b_id} anomaly ratio: {actual_ratio:.2f}%")

df_b['meter_reading'] = df_b['meter_reading'].fillna(
    df_b['meter_reading'].median()
)

os.makedirs('temp', exist_ok=True)
df_b.to_csv('temp/train.csv', index=False)
PY

        python -u run.py \
            --task_name anomaly_detection \
            --is_training 1 \
            --root_path ./temp \
            --data_path train.csv \
            --model_id LEAD \
            --model "${MODEL}" \
            --data LEAD \
            --features S \
            --seq_len 168 \
            --pred_len 0 \
            --d_model 64 \
            --d_ff 64 \
            --e_layers 2 \
            --enc_in 1 \
            --c_out 1 \
            --batch_size 64 \
            --train_epochs 20 \
            --building_id "${BID}" \
            --learning_rate 0.0001 \
            --patience 5

        echo "Finished Building ${BID}"
        echo
    done
done

echo "All experiments completed."
