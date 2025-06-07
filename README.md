# Fraud Detection and Rule Generation Framework

This repository contains a Databricks notebook designed for end-to-end fraud detection, anomaly identification using autoencoders and Isolation Forests, and the generation of new fraud rules using Decision Trees and Genetic Algorithms.

## Table of Contents

-   [Features](#features)
-   [Installation](#installation)
-   [Usage](#usage)
    -   [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    -   [Logistic Regression Model](#logistic-regression-model)
    -   [Autoencoder for Anomaly Detection](#autoencoder-for-anomaly-detection)
    -   [Isolation Forest for Anomaly Tagging](#isolation-forest-for-anomaly-tagging)
    -   [Rule Identification with Decision Trees](#rule-identification-with-decision-trees)
    -   [Genetic Algorithm for Rule Optimization](#genetic-algorithm-for-rule-optimization)
-   [Artefacts](#artefacts)
-   [Configuration](#configuration)
-   [Dependencies](#dependencies)

## Features

This framework provides the following functionalities:

-   **Data Loading and Splitting**: Reads transactional data from a Databricks table and splits it into training and testing sets, ensuring customer separation.
-   **Logistic Regression**: Trains a logistic regression model as a "feeder model" to identify potential fraud.
-   **Autoencoder**: Implements and trains a deep learning autoencoder for unsupervised anomaly detection, generating embeddings for transactions.
-   **Isolation Forest**: Applies Isolation Forest on autoencoder embeddings to tag anomalies (potential fraud) in the "Below The Line" (BTL) transactions (those not flagged by the logistic regression model).
-   **Rule Identification**: Leverages Decision Trees to extract interpretable rules from the identified anomalies.
-   **Rule Evaluation**: Calculates performance statistics for both suggested and existing fraud rules.
-   **Genetic Algorithm for Rule Optimization**: Optimizes a combination of suggested and existing rules based on defined fitness criteria (e.g., maximizing fraud volume and value while minimizing the number of rules).
-   **Artefact Saving**: Persists trained models (Logistic Regression, Isolation Forest, Autoencoder scaler and weights), and intermediate dataframes.

## Installation

This code is designed to run within a Databricks environment.

1.  **Install Libraries**: The first cell in the Databricks notebook installs the necessary Python packages:
    ```python
    !pip install torch
    !pip install deap
    ```

## Usage

The notebook is structured into several sections, each performing a specific task in the fraud detection and rule generation pipeline.

### Data Loading and Preprocessing

The `read_data` function loads the dataset from the specified Databricks table. The `custom_train_test_split` function ensures that customers are not split across training and testing sets, maintaining data integrity.

```python
# Standard Variables
TABLE_LOCATION = 'hackathon.were_just_here_for_the_snacks.synthetic_features_data_small_lowfraud'
# ... other standard variables

feature_pdf = read_data(TABLE_LOCATION)

# Data split by time periods
feature_pdf_apr = feature_pdf[(feature_pdf['TX_DATETIME'] >= '2025-04-01') & (feature_pdf['TX_DATETIME'] < '2025-05-01')]
feature_pdf_may_first3wk = feature_pdf[(feature_pdf['TX_DATETIME'] >= '2025-05-01') & (feature_pdf['TX_DATETIME'] < '2025-05-24')]
feature_pdf_may_lastwk = feature_pdf[(feature_pdf['TX_DATETIME'] >= '2025-05-24') & (feature_pdf['TX_DATETIME'] < '2025-06-01')]