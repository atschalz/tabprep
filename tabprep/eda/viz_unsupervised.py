import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set default plotting style
sns.set(style="whitegrid")

def plot_missing_values(df):
    """
    Visualize missing values in a dataset as a heatmap.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()


def plot_histograms(df, columns=None, bins=30):
    """
    Plot histograms for numerical columns in the dataset.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns
    
    df[columns].hist(bins=bins, figsize=(15, 10), edgecolor="black")
    plt.suptitle("Histograms of Numeric Features")
    plt.show()


def plot_correlation_matrix(df):
    """
    Plot a correlation heatmap for numeric columns.
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


def plot_boxplots(df, columns=None):
    """
    Plot boxplots for numerical columns to check for outliers.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns
    
    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()


def plot_categorical_counts(df, columns=None):
    """
    Plot countplots for categorical columns.
    """
    if columns is None:
        columns = df.select_dtypes(include="object").columns
    
    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Count Plot of {col}")
        plt.xticks(rotation=45)
        plt.show()

def analyze_cat_feature(x,y):
    # NOTE: histograms look different depending on whether we treat a feature as string or float - could somehow use this information
    fig,ax = plt.subplots(1,4, figsize=(10, 5))
    ax[0].scatter(x, y)

    y_by_x = y.groupby(x, observed=False).mean()
    ax[1].scatter(y_by_x.index, y_by_x.values)

    ax[2].hist(x, bins=100)
    pd.Series(x.astype(float).sort_values()).plot(kind='hist', bins=100, ax=ax[3])


def plot_pairwise_relationships(df, columns=None, hue=None):
    """
    Plot pairwise relationships (scatterplots + histograms) for numeric data.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns
    
    sns.pairplot(df[columns], hue=hue, diag_kind="kde")
    plt.suptitle("Pairwise Relationships", y=1.02)
    plt.show()

def plot_pairwise_relationships_fast(
    df,
    columns=None,
    hue=None,
    sample=8000,          # downsample rows for speed (None to disable)
    corner=True,          # plot only lower triangle
    kind="scatter",       # 'scatter' is much faster than 'kde'
    diag_kind="hist",     # 'hist' is much faster than 'kde'
    bins=20,
    height=1.8,
    alpha=0.6,
    marker_size=8,
    drop_na=True,
):
    """
    Much faster alternative to seaborn.pairplot.

    Speed-ups:
      - Optional row downsampling.
      - corner=True (plots only half the matrix).
      - diag_kind='hist' instead of 'kde'.
      - kind='scatter' (no density estimation).
      - rasterized markers for quicker rendering.
    """
    import seaborn as sns

    # choose numeric columns by default
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    # keep hue if present
    cols = columns + ([hue] if hue and hue not in columns else [])
    data = df[cols].copy()

    if drop_na:
        data = data.dropna(subset=columns + ([hue] if hue else []))

    # optional downsampling for speed
    if sample is not None and len(data) > sample:
        data = data.sample(sample, random_state=42)

    # build pairplot with fast settings
    g = sns.pairplot(
        data=data,
        vars=columns,
        hue=hue,
        kind=kind,
        diag_kind=diag_kind,
        corner=corner,
        height=height,
        plot_kws={
            "s": marker_size,
            "alpha": alpha,
            "rasterized": True,   # big performance win on large scatters
            "linewidth": 0,
        },
        diag_kws={"bins": bins, "edgecolor": "black", "linewidth": 0.25},
    )
    g.fig.suptitle("Pairwise Relationships (fast)", y=1.02)
    return g

def scatter_matrix_ultrafast(
    df,
    columns=None,
    sample=10000,       # downsample for speed
    bins=20,
    figsize=(10, 10),
    alpha=0.6,
):
    """
    Ultra-fast scatter matrix using pandas/matplotlib only.
    No hue support, but very quick even on larger data.
    """
    import numpy as np
    import pandas as pd
    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    data = df[columns].dropna()
    if sample is not None and len(data) > sample:
        data = data.sample(sample, random_state=42)

    axes = scatter_matrix(
        data,
        figsize=figsize,
        diagonal="hist",
        hist_kwds={"bins": bins, "edgecolor": "black", "linewidth": 0.25},
        alpha=alpha,
        range_padding=0.05,
    )

    # make the diagonal hist axes tight and off-grid for neatness
    for i in range(len(columns)):
        ax = axes[i, i]
        ax.grid(False)

    plt.suptitle("Scatter Matrix (ultra-fast matplotlib)", y=1.02)
    plt.tight_layout()
    return axes


# Example usage
if __name__ == "__main__":
    # Load sample dataset
    df = sns.load_dataset("titanic")

    plot_missing_values(df)
    plot_histograms(df, bins=20)
    plot_correlation_matrix(df)
    plot_boxplots(df, columns=["age", "fare"])
    plot_categorical_counts(df, columns=["sex", "class", "embarked"])
    plot_pairwise_relationships(df, columns=["age", "fare"], hue="survived")
