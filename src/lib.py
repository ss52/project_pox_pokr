from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math


def plot_moving_average(
    series: pd.DataFrame,
    window: int,
    plot_intervals: bool = False,
    scale: int = 1.96,
    plot_anomalies: bool = False,
) -> None:
    """Функция поиска выбросов и доверительного диапазона

    Args:
        series (pd.DataFrame): dataframe with timeseries
        window (int): rolling window size
        plot_intervals (bool, optional): show confidence intervals. Defaults to False.
        scale (int, optional): sigma value for confidential interval. Defaults to 1.96 (95 %).
        plot_anomalies (bool, optional): show anomalies. Defaults to False.
    """

    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(20, 8))
    plt.title("Скользящее среднее\n размер окна = {}".format(window))
    plt.plot(rolling_mean, "r", label="Скользящее среднее")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "g--", label="Верхняя / нижняя граница")
        plt.plot(lower_bond, "g--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Истинные значения")
    plt.legend(loc="upper left")
    plt.grid(True)


def file_save(df: pd.DataFrame, folder: str, name: str) -> None:
    """Функция сохранения данных в csv с zip компрессией.

    Args:
        df (pd.DataFrame): dataframe to save
        folder (str): folder where save df
        name (str): file name
    """

    comp = {"method": "zip", "archive_name": "out.csv"}

    filepath = Path(f"./{folder}/{name}.zip")

    df.to_csv(filepath, compression=comp)


def plot_data_rolling(
    data: pd.DataFrame, title: str, rolling: bool = False, window: int = 24
) -> None:
    """Print data with rolling curve

    Args:
        data (pd.DataFrame): original dataframe to print
        title (str): title of the figure
        rolling (bool, optional): bool print rolling or not. Defaults to False.
        window (int, optional): rolling window. Defaults to 24.
    """

    f, ax = plt.subplots(figsize=(15, 5))

    ax.plot(data.values)

    if rolling:
        rolling_mean = data.rolling(window=window).mean()
        ax.plot(rolling_mean.values, c="red")

    f.tight_layout()
    ax.set_ylabel(title)

    plt.show()


def smooth_data_moving_average(
    series: pd.DataFrame, window: int, scale: int = 1.96, fill: bool = True
) -> None:
    """Smooth data INPLACE!!!!!

    Args:
        series (pd.DataFrame): DateFrame with ONE column
        window (int): rolling window size
        scale (int, optional): sigma value for confidential interval. Defaults to 1.96 (95 %).
        fill (bool, optional): filling values with ffill method. Defaults to True.
    """

    rolling_mean = series.rolling(window=window).mean()

    mae = mean_absolute_error(series[window:], rolling_mean[window:])
    deviation = np.std(series[window:] - rolling_mean[window:])

    lower_bond = rolling_mean - (mae + scale * deviation)
    upper_bond = rolling_mean + (mae + scale * deviation)

    series.loc[(series < lower_bond) | (series > upper_bond)] = np.NaN

    if fill:
        series.ffill(inplace=True)


def model_score(model, x_test: np.array, y_test: np.array) -> None:
    """Функция рассчитывает метрики модели по RMSE, MAE и R2.
    Выводит график зависимости предсказанных и реальных значений.

    Args:S
        model (_type_): model for calculation
        x_test (np.array): X test np array
        y_test (np.array): y test np array
    """

    pred = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print("Метрики качества")
    print("RMSE: {:.4f}".format(rmse))
    print("MAE: {:.4f}".format(mae))
    print("R2: {:.4f}".format(r2))

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test.values, pred)

    scale = math.ceil(max(y_test.values.max(), pred.max()))
    plt.plot([0, scale], [0, scale], "r--")

    plt.xlabel("Данные реальные")
    plt.ylabel("Данные модели")

    plt.show()
