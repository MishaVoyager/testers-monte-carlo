import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
import pandas as pd
from pandas import DataFrame
from enum import Enum
import helpers
from helpers import NormalVar


class Beneficiary(Enum):
    """Получатель пользы от улучшения покрытия"""
    Tester = 1,
    Developer = 2,
    Both = 3


class ColumnName:
    """Строковые константы для названий колонок в таблице"""
    scenarios_per_task = "scenarios_per_task"
    automate_scenario_in_minutes = "automate_scenario_in_minutes"
    manual_check_scenario_in_minutes = "manual_check_scenario_in_minutes"
    additional_checks_per_year = "additional_checks_per_year"
    uncovered_scenario_fixes_per_year = "uncovered_scenario_fixes_per_year"
    fix_time_in_minutes = "fix_time_in_minutes"
    result_column = "total_economy"
    maintain_test_per_year_in_minutes = "maintain_test_per_year_in_minutes"


def main():
    get_test_coverage_economy(100000, 200, 5, 2, Beneficiary.Both)


def get_test_coverage_economy(n: int, tasks_per_year: int, years: int, min_checks: int, beneficiary: Beneficiary):
    """
    Рассчитывает шанс потратить на автоматизацию больше времени, чем сэкономить
    :param n: количество рандомных случаев (в нормальном распределении) для метода Монте-Карло
    :param tasks_per_year: количество задач в году
    :param years: количество лет, которые в среднем проживают задачи
    :param min_checks: минимальное количество проверок сценария во время тестирования задачи
    :param beneficiary: рассчитываем пользу для разработчиков, тестировщиков или всех
    """
    df = pd.DataFrame({
        ColumnName.scenarios_per_task: NormalVar(1, 21).gen_normal_values(n),
        ColumnName.automate_scenario_in_minutes: NormalVar(20, 100).gen_normal_values(n),
        ColumnName.maintain_test_per_year_in_minutes: NormalVar(5, 40).gen_normal_values(n)
    })
    if beneficiary.Tester or beneficiary.Both:
        df[ColumnName.manual_check_scenario_in_minutes] = NormalVar(5, 60).gen_normal_values(n)
        df[ColumnName.additional_checks_per_year] = NormalVar(0, 2).gen_normal_values(n)
    if beneficiary.Developer or Beneficiary.Both:
        df[ColumnName.uncovered_scenario_fixes_per_year] = NormalVar(0.1, 0.3).gen_normal_values(n)
        df[ColumnName.fix_time_in_minutes] = NormalVar(30, 100).gen_normal_values(n)
    df = adjust_nonpositive_values(df, beneficiary)

    match beneficiary:
        case Beneficiary.Tester:
            df = calculate_testers_economy(df, tasks_per_year, years, min_checks, ColumnName.result_column)
        case Beneficiary.Developer:
            df = calculate_developer_economy(df, tasks_per_year, years, ColumnName.result_column)
        case Beneficiary.Both:
            df = calculate_both_economy(df, tasks_per_year, years, min_checks, ColumnName.result_column)
        case _:
            raise ValueError(f"Некорректное значение Beneficiary: {beneficiary}")

    helpers.print_stats(df, ColumnName.result_column)
    helpers.draw_hist_with_average(
        df=df,
        series_name=ColumnName.result_column,
        label="Экономия",
        xlabel="Экономия в минутах",
        ylabel="Количество вариантов",
        show=True
    )
    plt.savefig("monte-coverage.jpg")
    working_hours_in_month = 159
    print(f"Среднее в рабочих месяцах: {df[ColumnName.result_column].mean() / 60 / working_hours_in_month}")
    helpers.output_fail_chance(0, df, ColumnName.result_column, "Шанс НЕ сэкономить")


def calculate_testers_economy(
        df: DataFrame, tasks_per_year: int, years: int, min_checks: int, result_column: str
) -> DataFrame:
    """Рассчитывает экономию только на ручных проверках"""
    testers_spendings_per_scenario = "testers_spendings_per_scenario"
    df[testers_spendings_per_scenario] = df.apply(
        lambda row:
        row[ColumnName.manual_check_scenario_in_minutes] *
        (min_checks + row[ColumnName.additional_checks_per_year] * years),
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row[testers_spendings_per_scenario] - row[ColumnName.automate_scenario_in_minutes] - row[
            ColumnName.maintain_test_per_year_in_minutes])
        * row[ColumnName.scenarios_per_task] * tasks_per_year * years,
        axis=1
    )
    return df


def calculate_developer_economy(
        df: DataFrame, tasks_per_year: int, years: int, result_column: str
) -> DataFrame:
    """Рассчитывает экономию только на фиксах багов в непокрытой функциональности"""
    dev_spendings_per_scenario = "dev_spendings_per_scenario"
    df[dev_spendings_per_scenario] = df.apply(
        lambda row: row[ColumnName.uncovered_scenario_fixes_per_year] * row[ColumnName.fix_time_in_minutes] * years,
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row[dev_spendings_per_scenario] - row[ColumnName.automate_scenario_in_minutes] - row[
            ColumnName.maintain_test_per_year_in_minutes])
        * row[ColumnName.scenarios_per_task] * tasks_per_year * years,
        axis=1
    )
    return df


def calculate_both_economy(
        df: DataFrame, tasks_per_year: int, years: int, min_checks: int, result_column: str
) -> DataFrame:
    """Рассчитывает экономию и на ручных проверках и на фиксах багов в непокрытой функциональности"""
    df["testers_spendings_per_scenario"] = df.apply(
        lambda row:
        row[ColumnName.manual_check_scenario_in_minutes] *
        (min_checks + row[ColumnName.additional_checks_per_year] * years), axis=1
    )
    df["dev_spendings_per_scenario"] = df.apply(
        lambda row: row[ColumnName.uncovered_scenario_fixes_per_year] * row[ColumnName.fix_time_in_minutes] * years,
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row["testers_spendings_per_scenario"] + row["dev_spendings_per_scenario"] - row[
            ColumnName.automate_scenario_in_minutes])
        * row[ColumnName.scenarios_per_task] * tasks_per_year * years,
        axis=1
    )
    return df


def adjust_nonpositive_values(df: DataFrame, beneficiary: Beneficiary) -> DataFrame:
    """Приведение нереальных непозитивных значений к минимальным"""
    df.loc[df[ColumnName.automate_scenario_in_minutes] <= 0, ColumnName.automate_scenario_in_minutes] = 1
    df.loc[df[ColumnName.scenarios_per_task] <= 0, ColumnName.scenarios_per_task] = 1
    df.loc[df[ColumnName.maintain_test_per_year_in_minutes] < 0, ColumnName.maintain_test_per_year_in_minutes] = 0
    if beneficiary.Tester or beneficiary.Both:
        df.loc[df[ColumnName.manual_check_scenario_in_minutes] <= 0, ColumnName.manual_check_scenario_in_minutes] = 1
        df.loc[df[ColumnName.additional_checks_per_year] < 0, ColumnName.additional_checks_per_year] = 0
    if beneficiary.Developer or beneficiary.Both:
        df.loc[df[ColumnName.uncovered_scenario_fixes_per_year] < 0, ColumnName.uncovered_scenario_fixes_per_year] = 0
        df.loc[df[ColumnName.fix_time_in_minutes] <= 0, ColumnName.fix_time_in_minutes] = 1
    return df


if __name__ == "__main__":
    main()
