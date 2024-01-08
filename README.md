# Economies on Energy Consumption Prediction
This is my submission for the technical assignment. Here's a quick guide to help you navigate through the project:

## Folder Structure

- **data/raw:** Raw data straight from the source.
- **data/processed:** Cleaned and preprocessed data for analysis.
- **data/external:** Additional data from external sources.
- **model:** Storehouse for serialized models.
- **notebooks:** Playground — explore, analyze, and draft your model.
- **src:** Codebase — utility functions and the final runnable version.
- **tests:** Unit tests—TBD for now.

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

### Communication (TBD)
1. Summarize and compare models.
2. Showcase feature contributions.
3. Utilize interactive widgets.
4. Dream big — an app might be in the future!

## Get Started
1. notebook/data_wrangling.ipynb
2. notebook/model_training.ipynb
