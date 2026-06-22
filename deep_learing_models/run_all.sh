#!/bin/bash

# =========================
# CONFIG
# =========================
MODELS=(
    Crossformer Informer KANDAD
    Nonstationary_Transformer iTransformer
)

DATA_PATH="../dataset/train.csv"
TEMP_DIR="./temp"

mkdir -p $TEMP_DIR

# =========================
# GET UNIQUE BUILDING IDS
# =========================
BUILDINGS=$(python - <<END
import pandas as pd
df = pd.read_csv("$DATA_PATH")
print(" ".join(map(str, df["building_id"].unique())))
END
)

# =========================
# LOOP BUILDINGS
# =========================
for b_id in $BUILDINGS; do
    # Check if building_id is less than or equal to 1226
    if [ "$b_id" -le 1226 ]; then
        continue # Skip to the next building in the loop
    fi

    echo "========== Processing building_id: $b_id =========="

    # Prepare data using Python
    python3 - <<END
import pandas as pd

# Now we know for sure this Python block only runs for b_id > 1226
df = pd.read_csv("$DATA_PATH")

df_b = df[df["building_id"] == int("$b_id")].copy()

ratio = df_b["anomaly"].mean() * 100

df_b["meter_reading"] = df_b["meter_reading"].fillna(df_b["meter_reading"].median())
df_b.to_csv("$TEMP_DIR/train.csv", index=False)

print(f"Anomaly Ratio: {ratio}")
END

    # Extract ratio again (clean way)
    RATIO=$(python - <<END
import pandas as pd
df = pd.read_csv("$TEMP_DIR/train.csv")
print(df["anomaly"].mean() * 100)
END
)

    echo "Anomaly Ratio for building $b_id: $RATIO"

    # =========================
    # LOOP MODELS
    # =========================
    for MODEL in "${MODELS[@]}"; do
        echo "--- Running Model: $MODEL for Building: $b_id ---"

        python -u run.py \
            --task_name anomaly_detection \
            --is_training 1 \
            --root_path $TEMP_DIR \
            --data_path train.csv \
            --model_id LEAD \
            --model $MODEL \
            --data LEAD \
            --features S \
            --seq_len 168 \
            --pred_len 0 \
            --d_model 64 \
            --d_ff 64 \
            --e_layers 2 \
            --enc_in 1 \
            --c_out 1 \
            --anomaly_ratio $RATIO \
            --batch_size 64 \
            --train_epochs 20 \
            --building_id $b_id \
            --learning_rate 0.0001 \
            --patience 5
    done

    echo "Finished all models for building_id: $b_id"
done
