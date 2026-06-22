# TSFM-BEAD: Time Series Foundation Models for Building Energy Anomaly Detection

## Overview

Buildings account for nearly 40% of global energy consumption, and roughly 20% of that energy is wasted due to equipment faults, system misconfigurations, and operational inefficiencies. Detecting anomalies in smart meter and building management system data is essential for catching equipment faults, sensor failures, inefficient energy use, and abnormal occupancy patterns. However, traditional anomaly detection approaches (statistical, ML, and deep learning) typically depend on handcrafted features, building-specific tuning, or large labeled datasets — limiting their scalability and transferability across diverse buildings.

Time Series Foundation Models, pretrained on large and heterogeneous temporal corpora, offer a promising alternative: transferable representations that support strong zero-shot and few-shot performance with little or no labeled data.

This study benchmarks pretrained TSFMs (**MOMENT**, **UniTS**, **TSPulse**) under both **zero-shot** and **fine-tuning** settings on the **LEAD (Large-scale Energy Anomaly Detection)** dataset — 192 non-residential buildings across 15 building types and sites — and compares them against:

- **Statistical baselines:** Modified Z-Score (mZ-Score), IQR
- **Machine learning models:** Local Outlier Factor (LOF), Isolation Forest (IForest), K-Means
- **Deep learning models:** CNN, LSTM-AD, Crossformer, KANAD, Informer, iTransformer, Non-Stationary Transformer (NST)

Performance is assessed using both point-wise metrics (AUC-ROC, PA-F1, Event-F1) and range-aware metrics (VUS-F1), to capture both classification accuracy and the temporal/event-level quality of detected anomalies.

The overall system architecture illustrated below.

![System Architecture](images/architecture_diagram.pdf)

## How to Run

All sub-modules expect the preprocessed LEAD dataset (see [Dataset](#dataset)) to be available, and use the shared preprocessing and evaluation utilities in `utils/` and `evaluation/`.

### 1. Statistical Models

Located in `statistical/`. These provide simple thresholding baselines (mZ-Score, IQR) with no training step required.

```bash
cd statistical
python mz_score.py --input_csv "../dataset/train.csv"
python iqr.py --input_csv "../dataset/train.csv"
```

### 2. Machine Learning Models

Located in `ml_models/`. Includes LOF, Isolation Forest, and K-Means.

```bash
cd ml_models
python lof.py --input_csv "../dataset/train.csv"
python iforest.py --input_csv "../dataset/train.csv"
python k_means.py --input_csv "../dataset/train.csv"
```

Each script loads the preprocessed LEAD dataset, fits the corresponding model per building, computes anomaly scores, and reports Point-F1, Event-F1, VUS-F1, and AUC-ROC via the `evaluation/` module.

### 3. Deep Learning Models

Located in `deep_learing_models/`. Includes CNN, LSTM-AD, Crossformer, KANAD, Informer, iTransformer, and Non-Stationary Transformer (NST).

A) To run Crossformer, KANAD, Informer, iTransformer and Non-Stationary Transformer (NST) sequentially, use the provided shell scripts:

```bash
cd deep_learning_models
bash run_all.sh
```

B) To run CNN and LSTM, use the following commands:

```bash
cd deep_learning_models/models
python cnn.py --input_csv '../../dataset/train.csv'
python lstm.py --input_csv '../../dataset/train.csv'
```

### 4. Time Series Foundation Models (TSFMs)

Located in `tsfm/`, organized by model: `moment/`, `units/`, and `tspulse/`. Each TSFM supports two settings — **zero-shot** (frozen pretrained model) and **fine-tuning** (adapted to the LEAD dataset).

**MOMENT**

```bash
cd tsfm/moment
python moment_zeroshot.py  --input_csv '../../dataset/train.csv'           # Zero-shot evaluation
python moment_finetuning.py --input_csv '../../dataset/train.csv'          # Fine-tuning on LEAD dataset
```

**UniTS**

```bash
cd tsfm/units
python units_zeroshot.py --input_csv '../../dataset/train.csv'             # Zero-shot evaluation
python units_finetuning.py --input_csv '../../dataset/train.csv'           # Fine-tuning on LEAD dataset
```

**TSPulse**

```bash
cd tsfm/tspulse
python tspulse_zeroshot.py --input_csv '../../dataset/train.csv'           # Zero-shot evaluation
python tspulse_finetuning_hp.py --input_csv '../../dataset/train.csv'      # Fine-tuning (with hyperparameter settings) on LEAD dataset
```

## Project Repository Structure

```
├── .gitignore
├── dataset
│   └── README.md
├── deep_learing_models
│   ├── data_provider/
│   ├── exp/
│   ├── layers/
│   ├── models
│   │   ├── base.py
│   │   ├── cnn.py
│   │   ├── Crossformer.py
│   │   ├── feature.py
│   │   ├── Informer.py
│   │   ├── iTransformer.py
│   │   ├── KANAD.py
│   │   ├── lstm.py
│   │   ├── Nonstationary_Transformer.py
│   │   └── __init__.py
│   ├── run.py
│   ├── run_all.sh
│   └── utils/
├── evaluation
│   ├── affiliation/
│   ├── basic_metrics.py
│   ├── metrics.py
│   ├── metrics_combined.py
│   ├── uns_metrics.py
│   ├── visualize.py
│   └── __init__.py
├── LICENSE
├── ml_models
│   ├── iforest.py
│   ├── k_means.py
│   └── lof.py
├── README.md
├── statistical
│   ├── iqr.py
│   └── mz_score.py
├── tsfm
│   ├── moment
│   │   ├── moment_finetuning.py
│   │   └── moment_zeroshot.py
│   ├── tspulse
│   │   ├── tspulse_finetuning_hp.py
│   │   └── tspulse_zeroshot.py
│   └── units
│       ├── UniTS.py
│       ├── units_finetuning.py
│       └── units_zeroshot.py
└── utils/
```
