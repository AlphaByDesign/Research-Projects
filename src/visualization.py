import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

# Create plotting distributions to visualize various types of distributions within the DF.

# Generates a pair plot for specified features and displays it.

def plot_feature_distribution(df: pl.DataFrame, features: List[str])-> None:

# Transform the Polars df into Pandas for use of Seaborn

    pd_df = df.select(features).to_pandas()

# Create the pair plot

    sns.pairplot(pd_df,diag_kind="kde")
    plt.suptitle("Feature Distribution", y=1.05)
    plt.show()
    plt.close()

# Create a correlation matrix to identify feature importance to target variable.

def display_feature_corr(
        df: pl.DataFrame,
        target_col: str = "close_log_return") -> pl.DataFrame:

# Arguments:
    # df: The Polars Df containing all features and the target.
    # target_col: the name of the target column for the correlation.
# returns a Polars data frame containing the correlation values.

        corr_df = df.corr()

        target_corr = corr_df.select(pl.col(target_col))
        target_corr = target_corr.with_columns(pl.Series(name="Feature",values=corr_df.columns))

        target_corr = target_corr.select([pl.col("Feature"),
        pl.col(target_col).abs().alias("Abs_correlation")]).sort("Abs_correlation",descending=True)

        return target_corr


