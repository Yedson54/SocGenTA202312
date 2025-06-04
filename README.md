# Economies on Energy Consumption Prediction
This project predicts potential energy consumption reduction for households. It contains data exploration utilities, modelling notebooks and trained models.

Below is a guide to help you navigate through the repository:

## Dataset

The raw dataset comes from a real-world energy consumption challenge. It
consists of two CSV files (`training_inputs.csv` and `training_outputs.csv`)
representing around 85k households with more than one hundred variables
(categorical and numeric). The target column `REDUCTION_POTENTIAL` indicates
whether a household is expected to lower its energy consumption.

Important feature families are:

- `C1..C19` socio-economic indicators
- `S1..S12` meter and subscription information
- `Q1..Q75` answers to a survey

Processed data is stored under `data/processed` and is used for modelling.

## Folder Structure

- **data/raw:** Raw data straight from the source.
- **data/processed:** Cleaned and preprocessed data for analysis.
- **data/external:** Additional data from external sources.
- **model:** Storehouse for serialized models. All artefacts are saved as pickle/joblib.
- **notebooks:** Playground — explore, analyze, and draft your model.
- **src:** Codebase — utility functions and the final runnable version.
- **tests:** Unit tests for the code base.

## Workflow

### Data Exploration
1. Load data.
2. Check variables types and shapes.
3. Handle missing values.
4. Identify outliers.
5. Tackle low- and high-cardinality variables.
6. Examine leakage variables.
7. Dive into univariate and bivariate distributions.

### Model Development
1. Define task and target.
2. Check target balance.
3. Set metrics and validation strategy.
4. Split data for training and testing.
5. Create a feature engineering pipeline.
6. Establish a baseline model.
7. Train, tune, and evaluate your model.
8. Explore feature contributions.

### Model Evaluation
The best performing model is a RandomForest classifier with an accuracy around
0.81 on the hold-out set. Logistic regression with calibrated probabilities
achieves a comparable score.

### Communication (TBD)
1. Summarize and compare models.
2. Showcase feature contributions.
3. Utilize interactive widgets.
4. Dream big — an app might be in the future!

## Installation

```bash
pip install -r requirements.txt
```

The `Makefile` provides shortcuts to execute the notebooks and run the unit
tests:

```bash
make notebooks
make tests
```

## Get Started
1. notebook/data_wrangling.ipynb
2. notebook/model_training.ipynb
