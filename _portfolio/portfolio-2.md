---
title: "Land Surface Temperature Forecasting using ARIMA model"
excerpt: "This project explores the application of the ARIMA model to forecast Land Surface Temperature (LST), a critical variable in understanding climate variability and supporting decision-making in sectors such as agriculture, urban planning, and public health. By analyzing historical satellite-derived temperature records, the model aims to capture temporal trends and seasonality, providing short- to medium-term forecasts. The resulting insights can inform adaptive strategies in climate-sensitive domains, highlighting the potential of time-series approaches in environmental modeling. <br/><img src='/images/500x300.png'>"
collection: portfolio
---

## Introduction
Land Surface Temperature (LST) is a key climate indicator that influences and reflects land–atmosphere interactions. Accurate forecasting of LST is critical in disciplines ranging from agriculture and hydrology to urban planning and epidemiological modeling. This project focuses on applying the `AutoRegressive Integrated Moving Average (ARIMA)` model to predict short-to-medium term changes in LST, drawing from historical satellite-derived temperature data.

## Objectives
- To collect and preprocess historical LST data over a defined spatial-temporal window.
- To test and implement the ARIMA model for forecasting LST.
- To evaluate the model's performance and identify seasonal, trend, and cyclic temperature components.
- To discuss the model’s potential integration with environmental, epidemiological, or agricultural models.

## Data Acquisition
- Source: Moderate Resolution Imaging Spectroradiometer (MODIS) Terra/Aqua LST datasets or other open-access repositories.
- Coverage: Daily or 8-day composite LST values; spatial focus on a defined region (e.g., Nairobi, Kenya).
- Preprocessing: Handling missing values, smoothing, and converting geospatial grids to time-series format.

## Methodology / How to go about it
- Stationarity Testing: Employ Augmented Dickey-Fuller (ADF) test to assess stationarity.
- Model Selection: Use ACF and PACF plots to guide identification of ARIMA(p,d,q) parameters.
- Model Fitting: Apply Maximum Likelihood Estimation (MLE) for parameter optimization.
- Diagnostics: Residual analysis, Ljung-Box test, and forecasting accuracy using RMSE and MAE.
- Validation: Use walk-forward validation or holdout datasets to test forecasting performance.

## Applications
- Urban Heat Island Effect: Anticipating urban thermal stress in fast-growing cities.
- Public Health: Supporting vector-borne disease models (e.g., malaria) via temperature-informed risk mapping.
- Agriculture: Enabling proactive planning in irrigation and planting schedules.
- Disaster Management: Contributing to drought early warning systems.

