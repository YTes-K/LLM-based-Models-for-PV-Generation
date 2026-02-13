# LLM-based-Models-for-PV-Generation

# PV Generation Forecasting using Prophet

This project implements a photovoltaic (PV) power generation forecasting model using **Meta Prophet**, **XGBoost**, **Lag LLaMA**.  
The objective is to model seasonal patterns (daily, weekly, yearly) and evaluate forecast performance using normalized error metrics suitable for energy systems.

---

## Prophet Overview

Photovoltaic generation exhibits strong:

- Daily seasonality (sunrise–sunset pattern)
- Yearly seasonality (summer–winter irradiance variation)
- Long-term trend (capacity or environmental changes)

Prophet is used as a baseline interpretable forecasting model before applying advanced ML models such as XGBoost or Lag LLaMA.

---

## Data Description

- Resolution: **15-minute intervals**
- Variable:
  - `ds` → timestamp (UTC)
  - `y` → aggregated solar generation (MW)
- Source: ENTSO-E Transparency Platform
- Values represent **average power (MW)** over each 15-minute interval.
- The dataset is consist of the diverse resources of power geneneration. Solar power is extracted from dataset from 2022 January to 2025 Octorber.


---

## Model Configuration

```python
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=5.0,
    changepoint_range=0.8,
    n_changepoints=15,
    seasonality_mode='additive',
    growth='linear',
    interval_width=0.95,
    mcmc_samples=0,
    uncertainty_samples=1000
)
```
## XGBoost

# PV Power Generation Forecasting using XGBoost

This project implements an **XGBoost regression model** for photovoltaic (PV) power generation forecasting.

The model combines:
- Time-series lag features
- Cyclical time encoding (sin/cos)
- Solar geometry features
- Clear-sky irradiance proxy
- Feature importance analysis

---

## Overview

Photovoltaic power generation depends heavily on:

- Time of day
- Seasonality
- Solar position
- Weather variability

Instead of feeding raw timestamps directly into the model, we engineer **physics-informed and cyclical features** to improve predictive performance.

---

## Model Configuration

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=3000,
    max_depth=12,
    learning_rate=0.05,
    random_state=42
    subsample=0.9
    colsample\_bytree=0.9
    colsample\_bylevel=0.9
    min\_child\_weight=1
    n\_jobs=-1
    tree\_method=hist
    early\_stopping\_rounds=150
    eval\_metric=rmse
    max\_bin=512
    )
```

# Lag-Llama PV Forecasting (Zero-shot + Fine-tuning)

This repository provides a clean workflow to run **Lag-Llama** for **hourly PV generation forecasting**, including:
- **Zero-shot inference** using a pre-trained checkpoint
- **Fine-tuning** on your local PV dataset
- **Long-horizon forecasting** via **iterative (rolling) prediction**
- **Model persistence** (save/load) to avoid re-training every time

---

## Lag-Llama

**Lag-Llama** is a transformer-based **time-series foundation model** that supports:
- **Zero-shot forecasting**: forecast a new time series without training on it
- **Fine-tuning**: adapt the pre-trained model to domain-specific patterns (e.g., PV generation)
- **Probabilistic forecasts**: generates multiple samples so you can compute mean/quantiles (uncertainty)

---

## Model Configuration


import torch
from lag_llama.gluon.estimator import LagLlamaEstimator

prediction_length = 168
context_length = 336

estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=prediction_length,
    context_length=context_length,

    # Must match the checkpoint architecture
    input_size=1,
    n_layer=1,
    n_embd_per_head=36,
    n_head=4,

    scaling="std",
    time_feat=True,

    trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "max_epochs": TBD for user
        ## for "lagllama_PV_hourly_try_74epoch.ipynb": max_epochs is 80 and using version_21, gluonts_checkpoint_pattern = "lightning_logs/version_21/checkpoints/*.ckpt"
        ## for "lagllama_PV_hourly_try_85epoch.ipynb": max_epochs is 100 and using version_23, gluonts_checkpoint_pattern = "lightning_logs/version_23/checkpoints/*.ckpt"
    }
)
