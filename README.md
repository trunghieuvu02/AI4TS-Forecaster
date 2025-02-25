# AI4TS-Forecaster

We provide a comprehensive AI for time series forecasting task (AI4TS-Forecaster) that includes a wide range of AI models, data preprocessing, and evaluation metrics. The AI4TS-Forecaster is designed to facilitate the development of AI models for time series forecasting tasks.


:triangular_flag_on_post:**News** (2025.02) Multivariate time series datasets has been included in [Google Drive](https://drive.google.com/drive/folders/1vYFgM5Po3RekCIg-74ZTp_LhgeS_mbVW?usp=sharing). Code has been uploaded into this repo github. 

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all from [Google Drive](https://drive.google.com/drive/folders/1vYFgM5Po3RekCIg-74ZTp_LhgeS_mbVW?usp=sharing). **All the datasets can be used easily.**
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

## Models 

Models can be used for time series forecasting tasks include: Statistical, Machine Learning, and Deep Learning approaches.

### Statistical Models

- [X] **LR**: Regression Model - Darts Library [[code]](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html)

### Machine Learning Models

- [X] **XGBoost** - Darts Library [[code]](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html)
- [X] **LightGBM** - Darts Library [[code]](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html)

### Deep Learning Models

- [X] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
- [X] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).
- [X] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py).
- [X] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
- [X] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
- [X] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
- [X] **FAT** - FAT: Fusion-Attention Transformer for Remaining Useful Life Prediction [[ICPR 2024]](https://link.springer.com/chapter/10.1007/978-3-031-78192-6_19)

## Usage

1. Install the requirements.txt:

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the datasets from [[Google Drive]](https://drive.google.com/drive/folders/1vYFgM5Po3RekCIg-74ZTp_LhgeS_mbVW?usp=sharing). Put the datasets under the folder `./dataset/`.
3. Train and evaluate model. We provide the experiment scripts for TS4DL under the folder `./scripts/`. AI4ML under the folder `./scripts_ML/`.

- DL models:
```
# long-term forecasting
bash ./scripts_DL/long_term_forecast/ETT_script/Autoformer_ETTh1.sh
```

- ML models:
```
# long-term forecasting
bash ./scripts_ML/long_term_forecast/ETT_script/RegressionModel_ETTh1.sh
```
4. Develop your own DL model.

- Add the model file to the folder `./models`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.


## Contact

If you have any questions or want to use the code, please contact h.vu1@rgu.ac.uk.

