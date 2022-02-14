# InterpretTime: a new approach for the systematic evaluation of neural-network interpretability in time series classification

This repository is the implementation code for :
[InterpretTime: a new approach for the systematic evaluation of neural-network interpretability in time series classification](http://arxiv.org/abs/2202.05656).

If you find this code or idea useful, please consider citing our work:

```
@misc{turbé2022interprettime,
      title={InterpretTime: a new approach for the systematic evaluation of neural-network interpretability in time series classification}, 
      author={Hugues Turbé and Mina Bjelogrlic and Christian Lovis and Gianmarco Mengaldo},
      year={2022},
      eprint={2202.05656},
}
```

#### Table of Contents
* [Overview](#overview)
  + [Family of synthetic datasets](#family-of-synthetic-datasets)
    - [CNN](#cnn)
    - [Bi-LSTM](#bi-lstm)
    - [Transformer](#transformer)
  + [ECG dataset](#ecg-dataset)
* [Usage](#usage)
  + [Requirements](#requirements)
  + [Setup](#setup)
  + [Pre-trained Models](#pre-trained-models)
  + [Post-processing](#post-processing)
* [Example for Dynamical system dataset](#example-for-dynamical-system-dataset)
* [Example for ECG dataset](#example-for-ecg-dataset)
  
## Overview

This repository presents the framework used in the paper "InterpretTime: a new approach for the systematic evaluation of neural-network interpretability in time series classification". The framework allows 
- evaluating and ranking of interpretability methods' performance via the AUCSE metric; 
- assessing the overlap between a human expert and a neural network in terms of data interpretation, 
for time series classification. 

The paper also introduces a new family of synthetic datasets with tunable complexity that can be used to assess the performance of interpretability methods, and that is able to reproduce time-series classification tasks of arbitrary complexity. 

In the tables below we show the ranking of 6 common interpretabilty methods using the AUCSE metric for 3 different neural-network architectures and for both the family of synthetic datasets, and for an ECG dataset.


### Family of synthetic datasets

#### CNN

| Method               | SD1        | SD2       | SD3        | Average   | Ranking |
|----------------------|------------|-----------|------------|-----------|---------|
| DeepLift             | 0.877      | 0.904     | 0.878      | 0.886     | 2       |
| GradShap             | 0.816      | 0.836     | 0.835      | 0.829     | 4       |
| Integrated Gradients | 0.877      | 0.904     | **0.879**  | 0.886     | 2       |
| KernelShap           | 0.629      | 0.61      | 0.652      | 0.63      | 6       |
| Saliency             | 0.799      | 0.755     | 0.806      | 0.787     | 5       |
| **Shapley Sampling** | **0.883**  | **0.907** | 0.873      | **0.888** | **1**   |
| Random               | 0.606      | 0.578     | 0.634      | 0.606     | 7       |


#### Bi-LSTM

| Method               | SD1        | SD2       | SD3       | Average   | Ranking |
|----------------------|------------|-----------|-----------|-----------|---------|
| DeepLift             | 0.512      | 0.566     | 0.607     | 0.562     | 5       |
| GradShap             | 0.587      | 0.557     | 0.598     | 0.581     | 4       |
| Integrated Gradients | 0.658      | 0.738     | 0.725     | 0.707     | 2       |
| KernelShap           | 0.437      | 0.364     | 0.408     | 0.403     | 6       |
| Saliency             | 0.651      | 0.539     | 0.583     | 0.591     | 3       |
| **Shapley Sampling** | **0.704**  | **0.825** | **0.749** | **0.759** | **1**   |
| Random               | 0.415      | 0.388     | 0.381     | 0.394     | 7       |

#### Transformer

| Method                   | SD1        | SD2       | SD3       | Average   | Ranking |
|--------------------------|-------     |-------    |-------    |---------  |---------|
| DeepLift                 | 0.827      | 0.847     | 0.774     | 0.816     | 4       |
| GradShap                 | 0.867      | 0.858     | 0.78      | 0.835     | 3       |
| **Integrated Gradients** | **0.916**  | **0.916** | 0.816     | **0.883** | **1**   |
| KernelShap               | 0.43       | 0.406     | 0.470     | 0.435     | 6       |
| Saliency                 | 0.463      | 0.526     | 0.417     | 0.469     | 5       |
| Shapley Sampling         | 0.898      | 0.909     | **0.830** | 0.879     | 2       |
| Random                   | 0.295      | 0.314     | 0.293     | 0.301     | 7       |

### ECG dataset

The result for the classification task on the ECG dataset with the AUCSE metric are presented in the table below.

|                          | CNN        | Bi-LSTM       | Transformer |
|--------------------------|------------|---------------|-------------|
| DeepLift                 | 0.47       | 0.603         | 0.63        |
| GradShap                 | 0.456      | 0.561         | 0.714       |
| Integrated Gradients     | 0.477      | 0.649         | 0.759       |
| KernelShap               | 0.336      | 0.173         | 0.444       |
| Saliency                 | 0.388      | 0.332         | 0.659       |
| **Shapley Sampling**     | **0.484**  | **0.652**     | **0.802**   |

## Usage

### Requirements

Setup tested with python 3.7.0 and Linux

To install requirements:
 
```setup
pip install -r requirements.txt
```

### Setup

Data are stored in the zip file called `datasets.zip` and should be unzip to obtain the following structure:
```
ts_robust_interpretability
    │
    ├── src
    ├──trained_model
    |  ├── dynamics 
    |  └── ecg
    │ 
    └──datasets
       ├── dynamics 
       └── ecg
```

  
### Pre-trained Models

Pre-trained models are provided in the *trained_model* folder. There are 9 models related to the synthetic dataset family (dynamics) and 3 models related to the ECG dataset.
Each folder with a model is organised as follows:
```
Simulation
    ├── results 
    ├── classes_encoder.npy 
    ├── config__***.yaml 
    └── best_model.ckpt
```

-  `results`: folder with the results of the simulation

-  `classes_encoder.npy`: class used for the encoder

-  `config__***.yaml`: config file with model's hyperparameters

-  `best_model.ckpt`: saved model 



### Post-processing

Relevance and metrics presented in the paper can be computed using the following command from within the src/operations folder:

```relevance
python3 __exec_postprocessing.py --results_path --method_relevance
```


-  `results_path`: path to the folder with the trained model

-  `method_relevance`: list of interpretability method to be evaluated. Can be one (or a list) of [shapleyvalue, integrated_gradients, deeplift, gradshap, saliency, kernelshap]

Interpretability evaluation metrics are saved in `results_path/interpretability_results`

## Example for Dynamical system dataset

```relevance
python3 __exec_postprocessing.py --results_path ../../trained_models/dynamics/SD1_CNN
```


## Example for ECG dataset

```relevance
python3 __exec_postprocessing.py --results_path ../../trained_models/ecg/CNN
```
