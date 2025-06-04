# Import libraries
import glob
import warnings
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')




def load_data(data_rootpath: str) -> pd.DataFrame:
    """
    Load data from CSV files in the specified directory and merge them.

    Args:
        data_rootpath (str): The root path of the data files.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Get all filenames in the folder and load each of them.
    files = glob.glob(rf"{data_rootpath}/*")
    dfs = [pd.read_csv(file, sep=";", decimal=".") for file in files]
    # Merge DataFrames based on the 'ID' column.
    merged_data = pd.merge(dfs[0], dfs[1], how="inner", on="ID")
    
    return merged_data


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all columns in a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    data = data.copy()
    data = (
        data.fillna(np.nan) # consistent missing values
        .replace({True: "True", False: "False"})
    )
    
    # Normalize string columns.
    catcols = data.select_dtypes(object).columns
    data[catcols] = data[catcols].apply(lambda x: x.str.upper().str.strip().replace({"TRUE": "True", "FALSE": "False"}))

    # Normalize date columns
    s_cols = data.filter(regex='^S[3-7]').columns
    # # Enforce numeric type when possible.
    data[s_cols] = data[s_cols].apply(lambda x: pd.to_datetime(x, errors="ignore", format="%Y-%m-%d"))
    
    
    data = data.convert_dtypes(convert_string=False).replace(pd.NA, np.nan) 
    return data


def wrangle(filepath: str) -> pd.DataFrame:
    """
    Load, normalize, and wrangle data from CSV files.

    Args:
        filepath (str): The path to the CSV files.

    Returns:
        pd.DataFrame: Wrangled DataFrame.
    """
    # Load data from CSV files and merge them.
    data = load_data(filepath)
    # Normalize columns data types.
    data = normalize_columns(data)

    return data


def entropy(column_series: pd.Series, normalize: bool=False) -> float:
    """Compute the entropy.

    Args:
        column_series (pd.Series): a series
        normalize (bool, optional): whether to normalize the entropy by 
            the number of class in the series. Defaults to False.

    Returns:
        float: entropy of the series
    """
    n_classes = column_series.nunique()
    if n_classes <= 1:
        return 0 
    
    # Compute frequencies then entropy.
    value_counts = column_series.value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log2(value_counts)) 
    
    if normalize:
        return entropy / np.log2(n_classes)
    return entropy

def compute_entropies(df: pd.DataFrame, normalize=False):
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    other_binary_columns = df.select_dtypes("number").columns[df.select_dtypes("number").nunique() == 2]
    categorical_columns = categorical_columns.append(other_binary_columns) 
    entropies = {column: entropy(df[column], normalize=normalize) for column in categorical_columns}
    return entropies


def plot_kde_grid(data, columns, grid_cols=3, hue=None, target=None):
    """
    Plot KDE plots for specified columns in a grid.

    Parameters:
    - data: DataFrame
    - columns: List of column names to plot
    - grid_cols: Number of columns in the grid
    - hue: Column name to use for color encoding

    Returns:
    - None (displays the grid of KDE plots)
    """
    # Calculate the number of rows needed in the grid
    if isinstance(columns, (pd.Index, list)):
        columns = list(columns)
    if target and target in columns:
        columns.remove(target)
    grid_rows, reminder = divmod(len(columns) - (1 - (not hue)), grid_cols)
    grid_rows += (reminder > 0)  # assess if there remain some columns

    # Create a grid of subplots
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()

    # Loop through each column and plot KDE
    for i, column in enumerate(columns):
        ax = axes[i]
        if not hue:
            sns.kdeplot(data[column], ax=ax, fill=True)
        else:
            for hue_level in data[hue].unique():
                subset_data = data[data[hue] == hue_level]
                sns.kdeplot(subset_data[column], ax=ax, fill=True, label=f"{hue_level}", common_norm=False)

        ax.set_title(column)
        ax.set_xlabel('')
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def test_distribution_difference(data, target_variable, feature_variable, equal_var=False):
    """
    Test the difference of distribution between two classes of a target variable for a continuous feature variable.

    Parameters:
    - data: DataFrame
    - target_variable: Column name of the target variable (binary)
    - feature_variable: Column name of the continuous feature variable

    Returns:
    - test_statistic: Test statistic
    - p_value: p-value
    """
    class_0 = data[data[target_variable] == 0][feature_variable]
    class_1 = data[data[target_variable] == 1][feature_variable]

    # # Assuming the data is approximately normally distributed
    # test_statistic, p_value = stats.ttest_ind(class_0, class_1, equal_var=equal_var)

    # # Cramér-von Mises test
    # test_statistic, p_value = stats.cramervonmises(class_0, class_1)

    # Kolmogorov-Smirnov test
    test_statistic, p_value = stats.ks_2samp(class_0, class_1)

    return test_statistic, p_value

def test_distribution_difference_categorical(data, target_variable, categorical_variable):
    """
    Test the difference of distribution between two classes of a target variable for a categorical variable.

    Parameters:
    - data: DataFrame
    - target_variable: Column name of the target variable (binary)
    - categorical_variable: Column name of the categorical variable

    Returns:
    - chi2_statistic: Chi-squared statistic
    - p_value: p-value
    """
    contingency_table = pd.crosstab(data[target_variable], data[categorical_variable])
    chi2_statistic, p_value, _, _ = stats.chi2_contingency(contingency_table)

    return chi2_statistic, p_value


def test_distribution_difference_all(data: pd.DataFrame, target_variable: str, feature_variables: List[str] | pd.Index):
    """
    Test the difference of distribution between two classes of a target variable for a list of feature variables.

    Parameters:
    - data: DataFrame
    - target_variable: Column name of the target variable (binary)
    - feature_variables: List of column names of feature variables

    Returns:
    - result_dict: Dictionary containing test statistics and p-values for each feature variable
    """
    result_dict = {}
    for feature in feature_variables:
        if feature == target_variable:
            continue
        if pd.api.types.is_numeric_dtype(data[feature]):
            result_dict[feature] = test_distribution_difference(
                data, target_variable, feature
            )
        else:
            result_dict[feature] = test_distribution_difference_categorical(
                data, target_variable, feature
            )

    # for feature in feature_variables:
    #     if data[feature].dtype == 'O':
    #         # Categorical variable
    #         chi2_stat, p_value = test_distribution_difference_categorical(data, target_variable, feature)
    #     else:
    #         # Continuous variable
    #         test_stat, p_value = test_distribution_difference(data, target_variable, feature)

    #     result_dict[feature] = [test_stat, p_value]

    return result_dict

def corr_matrix_threshold(df: pd.DataFrame,
                          cols: List[str],
                          method: str = "pearson",
                          threshold: float = 0.7,
                          cmap: str="coolwarm"):
    # Correlation matrix
    corr_matrix_pearson = df[cols].corr(method=method)
    np.fill_diagonal(corr_matrix_pearson.values, np.nan)
    # Select columns where at least one coefficient is above the threshold
    selected_cols = (corr_matrix_pearson.abs() > threshold).any()
    # Display the matrix formed only by these columns
    high_corr_matrix = corr_matrix_pearson.loc[selected_cols, selected_cols]
    # Display the matrix
    return high_corr_matrix.style.background_gradient(cmap=cmap)