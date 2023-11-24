
# nanoDPO: Direct Preference Optimization for Time Series Data

[![PyPI](https://img.shields.io/pypi/v/nanoDPO.svg)](https://pypi.org/project/nanoDPO/)
[![Changelog](https://img.shields.io/github/v/release/jamesliu/nanoDPO?include_prereleases&label=changelog)](https://github.com/jamesliu/nanoDPO/releases)
[![Tests](https://github.com/jamesliu/nanoDPO/workflows/Test/badge.svg)](https://github.com/jamesliu/nanoDPO/actions?query=workflow%3ATest)
[![Documentation Status](https://readthedocs.org/projects/nanoDPO/badge/?version=stable)](http://nanoDPO.readthedocs.org/en/stable/?badge=stable)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jamesliu/nanoDPO/blob/main/LICENSE)

## Introduction
Welcome to `nanoDPO` – a novel, cutting-edge library for Direct Preference Optimization (DPO) tailored for time series data. Inspired by the concept of utilizing DPO in fine-tuning unsupervised Language Models (LMs) to align with user preferences, `nanoDPO` pivots this approach to the realm of time series analysis. This library offers a unique perspective and toolset for time series forecasting, leveraging the principles of DPO to model and predict preferences in sequential data.

## Installation
To get started with `nanoDPO`, simply install the package using pip:

```bash
pip install nanoDPO
```

## Key Features

* Causal Transformer & Simple Sequence Model: Incorporates both a Causal Transformer and a LSTM-based Simple Sequence Model for diverse modeling needs.
* Preference Data Simulation: Utilizes a custom function, simulate_dpo_dataset_noise, to generate synthetic preference-based time series data.
* Sequence Data Preparation: Prepares data for training with prepare_sequence_datasets, aligning time series data with the DPO framework.
* DPO Training with PyTorch: Leverages the power of PyTorch for efficient and effective model training, complete with customizable parameters.
* MulticlassTrainer provides an additional approach to handle time series data, focusing on traditional multiclass classification tasks. 
* Cross-Entropy Loss for Multiclass Classification: Optimized for handling multiple classes in time series data.
* Customizable Training and Evaluation: Flexible parameters for epochs, batch size, and learning rate.
* Model Evaluation and Visualization: Offers tools for model evaluation and metrics visualization, ensuring an insightful analysis of performance.

## Usage

Import the necessary modules from nanoDPO, including the CausalTransformer, SimpleSequenceModel, and dataset preparation functions. Utilize the DPOOneModelTrainer for Direct Preference Optimization or MulticlassTrainer for conventional multiclass training.

```python
import torch
from nanodpo.causal_transformer import CausalTransformer
from nanodpo.simple_sequence_model import SimpleSequenceModel
from nanodpo.preference_data import simulate_dpo_dataset_noise
from nanodpo.sequence_data import prepare_sequence_datasets
from nanodpo.dpo_onemodel_trainer import DPOOneModelTrainer
from nanodpo.multiclass_trainer import MulticlassTrainer

# Initialize and train your model
...
model = CausalTransformer(d_feature= feature_dim, d_model=d_model, n_head=n_head, n_layer=n_layer, 
                          num_actions=num_actions, max_len=sequence_len,
                          device=device).to(device)
...
trainer = DPOOneModelTrainer(model=model, model_dir=f"dpo_{model_type}_model/", device=device,
                             learning_rate=learning_rate, batch_size=batch_size)
trainer.train(train_dataset, test_dataset, epochs=epochs, eval_interval=eval_interval)
# Evaluate and visualize the results
trainer.plot_metrics()
trainer.evaluate(test_dataset)
```

![wandb dpo causal_transformer](https://github.com/jamesliu/nanoDPO/blob/main/assets/dpo_causal_transformer_wandb.png)

## License

nanoDPO is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/jamesliu/nanoDPO/blob/main/LICENSE) file for more details.

## Acknowledgments

Inspired by the paper "[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)," nanoDPO extends the concept of DPO to the domain of time series data, opening new avenues for research and application.

## Citation

```bibtex
@misc{rafailov2023direct,
  title   = {Direct Preference Optimization: Your Language Model is Secretly a Reward Model}, 
  author  = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon and Christopher D. Manning and Chelsea Finn},
  year    = {2023},
  eprint  = {2305.18290},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```
