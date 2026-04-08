# 📥 Datasets Guide

## Where to Download

| Disease | Dataset | Link |
|---------|---------|------|
| Diabetes | Pima Indians Diabetes Dataset | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Heart Disease | UCI Heart Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| Kidney Disease | UCI Chronic Kidney Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease) |

---

## Where to Place Files

After downloading, place the CSV files inside the `data/raw/` folder and rename them exactly as shown:

```
data/
└── raw/
    ├── diabetes.csv
    ├── heart.csv
    └── kidney.csv
```

> ⚠️ These files are in .gitignore and will NOT be committed to GitHub.
> Share the raw data folder with your team via Google Drive or WhatsApp.

---

## Quick Look at Each Dataset

### Diabetes (diabetes.csv)
- 768 rows, 9 columns
- Target column: `Outcome` (0 = No Diabetes, 1 = Diabetes)
- Key features: Glucose, BMI, Age, Insulin, BloodPressure

### Heart Disease (heart.csv)
- 303 rows, 14 columns
- Target column: `target` (0 = No Disease, 1 = Disease)
- Key features: age, sex, cp (chest pain), thalach (max heart rate), chol

### Kidney Disease (kidney.csv)
- 400 rows, 26 columns
- Target column: `classification` (ckd = sick, notckd = healthy)
- Key features: blood glucose, hemoglobin, serum creatinine, blood urea
- ⚠️ This dataset has the most missing values — handle carefully
