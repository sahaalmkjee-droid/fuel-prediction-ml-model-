# 🚜 Mindshift Analytics: Haulmark Challenge — Heavy Equipment Fuel Prediction Pipeline

An end-to-end, leak-free machine learning and feature engineering pipeline designed to predict shift-level fuel consumption for mining dump trucks using high-frequency GPS, telemetry sensors, and operator behavior metrics.

---

## 📌 Project Overview

Predicting shift-level fuel consumption in heavy open-pit mining operations requires modeling complex interactions between vehicle physics, terrain topography, operational downtime, and driver habits[cite: 1]. This repository processes millions of high-frequency telemetry rows and engineers domain-specific features to feed a calibrated multi-model ensemble (LightGBM, XGBoost, and Ridge Regression)[cite: 1].

---

## 🛠️ Key Pipeline Innovations

### 1. **Telemetry Processing & Quality Assurance**
* **Scalable Ingestion:** Handles millions of raw telemetry rows efficiently from Parquet files using `pyarrow`[cite: 1].
* **Signal Filtering:** Filters noisy GPS/GNSS data by enforcing satellite count constraints (`satellites >= 4`) and Dilution of Precision bounds (`HDOP <= 1.5`, `PDOP <= 2.0`)[cite: 1].
* **Shift Mapping:** Maps UTC timestamps to local timezone (`Asia/Kolkata`) and classifies telemetry into operational daily shifts (Shifts A, B, and C)[cite: 1].

### 2. **Activity-Weighted Target Disaggregation**
* Daily fuel targets are disaggregated down to individual 8-hour operational shifts using **equivalent flat distance weighting**[cite: 1].
* Accurately prevents target distortion across high-activity vs. low-activity shifts on the same operational day[cite: 1].

### 3. **Domain-Driven Feature Engineering (25 Total Features)**
* **Physics & Kinematics:** Equivalent flat distance (applying elevation grade penalties), loaded climbing work, speed variance, and acceleration event proxies[cite: 1].
* **Equipment & Operations:** Sensor-detected dumping cycles via analog voltage thresholds (`analog_input_1`), idle vs. travel time splits, and shift-to-shift lag indicators[cite: 1].
* **Operator Profiling:** Aggression score (accel/hr), moving average speed, idle ratios, and dumping efficiency grouped per operator[cite: 1].
* **Terrain Severity:** Grade severity indices per mine and route intensity (haul distance per engine-on hour)[cite: 1].

### 4. **Leak-Free Target Encoding & Rolling Priors**
* **Out-Of-Fold (OOF) Target Encoding:** Applied to categorical features (`vehicle`, `mine_str`, `operator_str`, and interaction keys) using K-Fold cross-validation on train splits to prevent target leakage[cite: 1].
* **Exponentially Weighted Moving Average (EWMA):** Captures rolling historical fuel priors per vehicle to preserve persistent physical consumption trends[cite: 1].

---

## 🤖 Model Architecture & Ensembling

The core modeling engine operates on log-transformed targets (`log1p` / `expm1`) using a multi-model ensemble[cite: 1]:

* **LightGBM Regressor (RMSE Objective)**[cite: 1]
* **LightGBM Quantile Regressor ($\alpha = 0.5$)**[cite: 1]
* **XGBoost Regressor (Histogram Tree Method)**[cite: 1]
* **Ridge Regression (Standardized Pipeline)**[cite: 1]

### Optimization & Post-Processing
* **Walk-Forward Cross-Validation:** Evaluates models across strict chronological slices (`CV_FOLDS`) matching real-world deployment[cite: 1].
* **Bayesian Optimization:** Automates hyperparameter search per model fold using **Optuna** (TPE Sampler)[cite: 1].
* **Blend Weight Optimization:** Uses Nelder-Mead optimization to discover optimal model blend weights on out-of-fold predictions[cite: 1].
* **Isotonic Calibration & Scale Correction:** Applies isotonic regression calibration and vehicle-shift specific scaling factors to align overall distribution scale[cite: 1].

---

## 💻 Tech Stack & Libraries

* **Language:** Python 3.12[cite: 1]
* **Data Processing:** pandas, NumPy, SciPy, PyArrow[cite: 1]
* **Machine Learning:** LightGBM, XGBoost, scikit-learn[cite: 1]
* **Optimization:** Optuna[cite: 1]

---

## 📁 Repository Structure

```text
.
├── sahaal_.ipynb          # Main pipeline notebook containing data processing & ML execution
├── submission.csv            # Final shift-level test set predictions
└── README.md                 # Project documentation
