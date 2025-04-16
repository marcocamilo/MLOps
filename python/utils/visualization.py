import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

def time_hist(df: pd.DataFrame, time_cols: list, colors: list, bins: int, title: str, mean: bool = False, limit: bool = None, save_path: str = None, format: str = 'png'):
    """
    Plot histograms of time columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - time_cols (list): The list of time columns to plot.
    - colors (list): The list of colors for each time column.
    - bins (int): The number of bins for the histogram.
    - title (str): The title of the plot.
    - mean (bool): Display the mean value if True.
    - limit (bool): Set a limit for the histogram.
    - save_path (str): The path to save the plot.
    - format (str): The format of the saved plot.

    Returns:
    histogram of time columns
    """
    for col, color in zip(time_cols, colors):
        if limit:
            plt.hist(df[df[col] < limit][col], alpha=0.5, bins=bins, color=color)
            if mean:
                plt.axvline(df[df[col] < limit][col].mean(), color=color, linestyle='dashed', linewidth=2)
        else:
            plt.hist(df[col], alpha=0.5, bins=bins, color=color)
            if mean:
                plt.axvline(df[col].mean(), color=color, linestyle='dashed', linewidth=2)
    plt.legend(time_cols)
    plt.title(title)
    if np.issubdtype(df[col].dtype, np.datetime64):
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path, format=format)
    plt.show()

def jointplot(df: pd.DataFrame, x: str, y: str, kind: str ='boxplot', hue: str = None, title: str = None, save_path: str = None, format: str = 'png'):
    """
    Plot jointplot of two columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - x (str): The name of the first column.
    - y (str): The name of the second column.
    - kind (str): The kind of plot to display (boxplot, violinplot, histplot, kdeplot).
    - hue (str): The name of the column to color by.
    - title (str): The title of the plot.
    - save_path (str): The path to save the plot.
    - format (str): The format of the saved plot.

    Returns:
    jointplot of two columns
    """
    corr = df[[x, y]].corr().iloc[0, 1]
    display(f'Pearson correlation coefficient: {corr}')

    gridspecs = {'hspace': 0,
                 'wspace': 0,
                 'width_ratios': [5, 1],
                 'height_ratios': [1, 5]}
    
    fig, axes = plt.subplots(2, 2, gridspec_kw=gridspecs)
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[1, 1].axis("off")

    sns.scatterplot(x=x, y=y, data=df, ax=axes[1, 0], alpha=0.25, hue=hue)
    
    if kind == 'boxplot':
        sns.boxplot(x=x, data=df, ax=axes[0, 0], hue=hue)
        sns.boxplot(y=y, data=df, ax=axes[1, 1], hue=hue)
    elif kind == 'violinplot':
        sns.violinplot(x=x, data=df, ax=axes[0, 0], hue=hue)
        sns.violinplot(y=y, data=df, ax=axes[1, 1], hue=hue)
    elif kind == 'histplot':
        sns.histplot(x=x, data=df, ax=axes[0, 0], hue=hue)
        sns.histplot(y=y, data=df, ax=axes[1, 1], hue=hue)
    elif kind == 'kdeplot':
        sns.kdeplot(x=x, data=df, ax=axes[0, 0], hue=hue)
        sns.kdeplot(y=y, data=df, ax=axes[1, 1], hue=hue)
    else:
        raise ValueError('Invalid kind of plot. Must be one of: boxplot, violinplot, histplot, kdeplot')

    plt.suptitle(title)
    plt.subplots_adjust(top=0.95)
    if save_path:
        plt.savefig(save_path, format=format)
    plt.show()

def plot_with_cumulative_sum(df: pd.DataFrame, labels_column: str, values_column: str, save_path: str = None, format: str = 'png'):
        # Write the docstring
    """
    Plot a bar plot with cumulative sum overlay.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - labels_column (str): The name of the column containing labels.
    - values_column (str): The name of the column containing values.
    - save_path (str): The path to save the plot.
    - format (str): The format of the saved plot.

    Returns:
    bar plot with cumulative sum overlay
    """

    df_sorted = df.sort_values(by=values_column, ascending=False)

    df_sorted['cumulative_sum'] = df_sorted[values_column].cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_sorted, x=values_column, y=labels_column, ax=ax1)
    ax1.bar_label(ax1.containers[0], fmt="%0.2f%%")

    ax2 = ax1.twiny()
    ax2.plot(df_sorted['cumulative_sum'], df_sorted[labels_column], marker='o', color='red')
    ax2.set_xlabel('Cumulative Sum')

    ax1.set_xlabel('Values')
    ax1.set_ylabel('Labels')
    ax1.set_title('Bar Plot with Cumulative Sum Overlay')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, format=format)
    plt.show()
