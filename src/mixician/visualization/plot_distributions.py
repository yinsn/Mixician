from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DistributionViewer:
    """A class to visualize distributions of scores in various forms.

    This class provides methods to plot distributions of score data from a pandas DataFrame
    or a numpy array. It supports plotting in both linear and logarithmic scales.

    Attributes:
        dataframe (pd.DataFrame): The dataframe containing score data.
        score_columns (List[str]): The list of column names in the dataframe to plot.

    """

    def __init__(self, dataframe: pd.DataFrame, score_columns: List[str]) -> None:
        """Initializes the DistributionViewer with a dataframe and score columns.

        Args:
            dataframe (pd.DataFrame): The dataframe containing the score data.
            score_columns (List[str]): The list of column names in the dataframe that contain
                                       score data to be visualized.
        """
        self.dataframe = dataframe
        self.score_columns = score_columns

    def plot_logarithm_distributions(self) -> None:
        """Plots the logarithmic distribution of the scores in the dataframe.

        This method plots the distributions of scores on a logarithmic scale. It is particularly
        useful for visualizing data that spans several orders of magnitude.
        """
        DistributionViewer._common_settings()
        palette = sns.color_palette("Spectral", len(self.score_columns))
        for column, color in zip(self.score_columns, palette):
            sns.kdeplot(
                np.log10(self.dataframe[column]), label=column, fill=True, color=color
            )
            plt.yscale("symlog")
        plt.legend(loc="center left", framealpha=0, edgecolor="none")
        plt.xlabel(r"$\log_{10}(\text{Scores})$")
        plt.ylabel("Density (symlog)")
        plt.show()

    @staticmethod
    def _common_settings() -> None:
        """Applies common settings for all plots.

        This method sets the theme and the figure size for the plots.
        """
        sns.set_theme(style="ticks")
        plt.figure(figsize=(10, 3))

    @staticmethod
    def plot_logarithm_array_distribution(scores: np.ndarray, legend: str) -> None:
        """Plots the logarithmic distribution of scores from a numpy array.

        Args:
            scores (np.ndarray): The array of scores to be plotted.
            legend (str): The legend label for the plot.
        """
        DistributionViewer._common_settings()
        sns.kdeplot(np.log10(scores), fill=True, color="darkorange")
        plt.yscale("symlog")
        plt.legend([legend], framealpha=0, edgecolor="none")
        plt.xlabel(r"$\log_{10}(\text{Scores})$")
        plt.ylabel("Density (symlog)")
        plt.show()

    @staticmethod
    def plot_array_distribution(scores: np.ndarray, legend: str) -> None:
        """Plots the distribution of scores from a numpy array in linear scale.

        Args:
            scores (np.ndarray): The array of scores to be plotted.
            legend (str): The legend label for the plot.
        """
        DistributionViewer._common_settings()
        sns.kdeplot(scores, fill=True, color="darkorange")
        plt.yscale("linear")
        plt.legend([legend], framealpha=0, edgecolor="none")
        plt.xlabel(r"$\text{Scores}$")
        plt.ylabel("Density (linear)")
        plt.show()
