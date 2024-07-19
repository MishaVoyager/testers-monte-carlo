import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import helpers


def main():
    case_1_economies_per_product(10000, 400000, [(15, 10), (3, 10), (6, 6)], (25000, 20000))


def case_1_economies_per_product(
        n: int,
        min_profit: int,
        economies_per_product: list[tuple[int, int]],
        products_count: tuple[int, int]
) -> None:
    """
    Рассчитывает вероятность провала, когда прикидываем экономию на единицу продукции
    :param economies_per_product: список кортежей, где первый элемент - среднее,
    второй - разброс с уверенностью 90%
    """
    result_column_name = "result"
    df = prepare_df_for_economies_per_product(n, economies_per_product, products_count, result_column_name)
    helpers.draw_hist_with_average(df, result_column_name, "Потрачено", "Траты в р.", "Количество")
    helpers.draw_confidence_interval(df, result_column_name)
    helpers.draw_normal_distribution(df, result_column_name)

    plt.legend()
    plt.savefig("case_one.jpg")
    plt.show()

    helpers.output_fail_chance(min_profit, df, "result")


def prepare_df_for_economies_per_product(
        n: int,
        economies_per_product: list[tuple[int, int]],
        products_count: tuple[int, int],
        result_column_name: str
) -> DataFrame:
    """
    Готовит dataframe для кейса, когда прикидываем экономию на единицу продукции
    :param economies_per_product: список кортежей, где первый элемент - среднее,
    второй - разброс с уверенностью 90%
    """
    data = dict(
        enumerate([np.random.normal(economy, diapason / 3.29, n) for economy, diapason in economies_per_product]))
    data.update({len(economies_per_product): np.random.normal(products_count[0], products_count[1] / 3.29, n)})
    df = pd.DataFrame(data)
    df[result_column_name] = df.apply(
        lambda row: (sum([economy for economy in row[:len(data) - 1]])) * row[len(data) - 1], axis=1)
    return df


if __name__ == "__main__":
    main()
