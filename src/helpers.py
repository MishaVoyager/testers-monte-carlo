from typing import Optional

import matplotlib
from numpy import ndarray

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.stats import norm


class NormalVar:
    """
    На основе оценки min и max с 90% уверенностью
    Рассчитывает среднее и стандартное отклонение
    """

    def __init__(self, min: float, max: float, name: str):
        if max < min:
            raise ValueError(f"Max значение {max} меньше min {min}")
        self.min: float = min
        self.max: float = max
        self.average: float = (max - min) // 2
        self.std_90: float = (max - min) / 3.29
        self.name: str = name

    def gen_normal_values(self, n: int) -> ndarray:
        """
        Генерирует значения в рамках нормального распределения
        Для метода Монте-Карло нужен именно такой рандом
        """
        return np.random.normal(self.average, self.std_90, n)

    def gen_normal_values_dict(self, n: int) -> dict[str, ndarray]:
        """
        Генерирует значения в рамках нормального распределения
        Для метода Монте-Карло нужен именно такой рандом
        """
        return {self.name: np.random.normal(self.average, self.std_90, n)}


def draw_confidence_interval(df: DataFrame, series_name: str = "result") -> None:
    """Нарисовать две вертикали - двойное стандартное отклонение в обе стороны"""
    mean = np.mean(df[series_name])
    std = np.std(df[series_name])
    plt.axvline(mean - 2 * std, color='blue', linestyle='--')
    plt.axvline(mean + 2 * std, color='blue', linestyle='--')


def draw_normal_distribution(df: DataFrame, series_name: str = "result") -> None:
    """
    Нарисовать график нормального распределения для сравнения с фактическим распределением
    """
    x_axis = np.arange(df[series_name].min(), df[series_name].max(), 1)
    y_axis = norm.pdf(x_axis, df[series_name].mean(), df[series_name].std())
    plt.plot(x_axis, y_axis, "r", label="Норм. распределение")


def draw_normal_distribution_over_current_graph(df: DataFrame, series_name: str = "result") -> None:
    """
    Рисует график нормального распределения поверх (!) основного графика.
    Ничего не нарисует, если вызвать до создания основного графика.
    """
    mu, std = norm.fit(df[series_name])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, df[series_name].count())
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "b--", label="Normal_over")


def draw_hist_with_average(
        df: DataFrame,
        series_name: str,
        label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show: bool = False
) -> None:
    """Рисует гистограмму с линией среднего значения"""
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.hist(df[series_name],
             range=(df[series_name].min(), df[series_name].max()),
             color='green',
             bins=100,
             label=label
             )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axvline(df[series_name].mean(), color='grey', linestyle='--', label="Среднее")
    if show:
        plt.legend()
        plt.show()


def output_fail_chance(min_profit: int, df: DataFrame, series_name: str = "result", msg: str = "Шанс провала") -> None:
    """Выводит вероятность, что проект убыточный"""
    fail_chance = (df[series_name].size - df[df[series_name] > min_profit][series_name].count()) / df[
        series_name].size
    print(f"{msg}: {fail_chance}")


def print_stats(df: DataFrame, column_name: str) -> None:
    """Выводит основные статистические показатели"""
    print(f"Количество: {df[column_name].size}")
    print(f"Минимум: {df[column_name].min()}")
    print(f"Максимум: {df[column_name].max()}")
    print(f"Среднее: {df[column_name].mean()}")


def adjust_non_positive_values(df: DataFrame, positive_cols: list[str], non_negative_cols: list[str]) -> DataFrame:
    """Приведение нереальных непозитивных значений к минимальным"""
    for col in positive_cols:
        df.loc[df[col] <= 0, col] = 1
    for col in non_negative_cols:
        df.loc[df[col] < 0, col] = 0
    return df
