# Dataset

This project uses the **LEAD (Large-scale Energy Anomaly Detection)** dataset, which contains one year of hourly electricity consumption data from 200 non-residential buildings across 15 sites and 15 building types.

## Download

The dataset is publicly available on Kaggle:

🔗 [https://www.kaggle.com/competitions/energy-anomaly-detection/data?select=train.csv]([https://www.kaggle.com/competitions/energy-anomaly-detection](https://www.kaggle.com/competitions/energy-anomaly-detection/data?select=train.csv))

Download the dataset from the link above and place all the files inside this `dataset/` folder.

## Excluded Buildings

After partitioning the data chronologically (first 6 months for training, next 2 months for validation, and the last 4 months for testing), we found that the following **8 building IDs** had **no anomaly events in the last 4 months (test period)**:

```
32
534
653
693
739
970
1147
1264
```

Since meaningful evaluation requires positive anomaly instances in the test set, these buildings were **excluded** from the benchmark. This reduces the dataset from 200 buildings to the **192 buildings** used in all experiments and results reported in this study.

## Folder Structure

After downloading, your `dataset/` folder should look like this:

```
dataset/
├── README.md
├── train.csv
├── train_features.csv
```
