.. _changelog:

===========
 Changelog
===========

.. _v0_1:

0.1 (2023-11-01)
----------------

- Initial release to PyPI
- Causal Transformer & Simple Sequence Model: Incorporates both a Causal Transformer and a LSTM-based Simple Sequence Model for diverse modeling needs.
- Preference Data Simulation: Utilizes a custom function, simulate_dpo_dataset_noise, to generate synthetic preference-based time series data.
- Sequence Data Preparation: Prepares data for training with prepare_sequence_datasets, aligning time series data with the DPO framework.
- DPO Training with PyTorch: Leverages the power of PyTorch for efficient and effective model training, complete with customizable parameters.
- MulticlassTrainer provides an additional approach to handle time series data, focusing on traditional multiclass classification tasks. 
- Cross-Entropy Loss for Multiclass Classification: Optimized for handling multiple classes in time series data.
- Customizable Training and Evaluation: Flexible parameters for epochs, batch size, and learning rate.
- Model Evaluation and Visualization: Offers tools for model evaluation and metrics visualization, ensuring an insightful analysis of performance.
