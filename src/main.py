import matplotlib
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, field_validator

matplotlib.use('TkAgg')
import pandas as pd
from pandas import DataFrame
from enum import Enum
import helpers
from helpers import NormalVar, adjust_non_positive_values


class Beneficiary(Enum):
    """Получатель пользы от улучшения покрытия"""
    Tester = 1,
    Developer = 2,
    Both = 3


NON_NEGATIVE_COLS: list[str] = [
    "maintain_test_per_year_in_minutes",
    "additional_checks_per_year",
    "uncovered_scenario_fixes_per_year",
    "additional_fix_time_in_minutes"
]

POSITIVE_COLS: list[str] = [
    "scenarios_per_task",
    "automate_scenario_in_minutes",
    "manual_check_scenario_in_minutes"
]


class TestCoverageApproximations(BaseModel):
    """
    Значения, определенные на основе min и max с 90% уверенностью,
    необходимые для оценки времени, которое поможет сэкономить автоматизация
    :param scenarios_per_task: количество сценариев на автоматизацию в задаче
    :param automate_scenario_in_minutes: сколько минут займет автоматизация сценария
    :param manual_check_scenario_in_minutes: сколько минут займет ручная проверка сценария
    :param additional_checks_per_year: сколько раз в году приходится выполнять регрессионные проверки сценария
    :param uncovered_scenario_fixes_per_year: сколько раз в году ломается непокрытый сценарий
    :param additional_fix_time_in_minutes: доп. время на починку непокрытого сценария - из-за
    необходимости выяснять причину через логи, переключать контекст (чинить покрытый сценарий проще)
    :param maintain_test_per_year_in_minutes: сколько минут в год уходит на поддержку теста
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenarios_per_task: NormalVar
    automate_scenario_in_minutes: NormalVar
    manual_check_scenario_in_minutes: NormalVar
    additional_checks_per_year: NormalVar
    uncovered_scenario_fixes_per_year: NormalVar
    additional_fix_time_in_minutes: NormalVar
    maintain_test_per_year_in_minutes: NormalVar

    @field_validator(*POSITIVE_COLS)
    def check_positive_values(cls, value: NormalVar) -> NormalVar:
        if value.min <= 0:
            raise ValueError("Значение должно быть > 0")
        return value

    @field_validator(*NON_NEGATIVE_COLS)
    def check_nonnegative_values(cls, value: NormalVar) -> NormalVar:
        if value.min < 0:
            raise ValueError("Значение должно быть >= 0")
        return value


def main():
    data = TestCoverageApproximations(
        scenarios_per_task=NormalVar(1, 21, "scenarios_per_task"),
        automate_scenario_in_minutes=NormalVar(20, 100, "automate_scenario_in_minutes"),
        maintain_test_per_year_in_minutes=NormalVar(5, 40, "maintain_test_per_year_in_minutes"),
        manual_check_scenario_in_minutes=NormalVar(5, 60, "manual_check_scenario_in_minutes"),
        additional_checks_per_year=NormalVar(0, 2, "additional_checks_per_year"),
        uncovered_scenario_fixes_per_year=NormalVar(0.1, 0.3, "uncovered_scenario_fixes_per_year"),
        additional_fix_time_in_minutes=NormalVar(20, 100, "additional_fix_time_in_minutes")
    )
    get_test_coverage_economy(data, 10000, 200, 5, 2, Beneficiary.Both)


def get_test_coverage_economy(
        data: TestCoverageApproximations,
        n: int,
        tasks_per_year: int,
        years: int,
        min_checks: int,
        beneficiary: Beneficiary
) -> None:
    """
    Рассчитывает шанс потратить на автоматизацию больше времени, чем сэкономить
    :param data: информация о переменных, которые используются для расчета
    :param n: количество рандомных случаев (в нормальном распределении) для метода Монте-Карло
    :param tasks_per_year: количество задач в году, для которых можно написать хоть один тест
    :param years: количество лет, которые в среднем проживают задачи
    :param min_checks: минимальное количество проверок сценария во время тестирования задачи, например, 2,
    если выполняется 2 круга тестирования (можно сэкономить на обоих, если написать тест до ручного тестирования)
    :param beneficiary: рассчитываем пользу для разработчиков, тестировщиков или всех
    """
    df = prepare_df(data, beneficiary, n)
    df = adjust_non_positive_values(df, POSITIVE_COLS, NON_NEGATIVE_COLS)
    result_column = "total_economy"
    df = calculate_economy(df, tasks_per_year, years, min_checks, result_column, data, beneficiary)
    draw_and_print_coverage_case_result(df, result_column)


def prepare_df(data: TestCoverageApproximations, beneficiary: Beneficiary, n: int) -> DataFrame:
    """Создает dataframe с нормально распределенными данными для наших колонок"""
    df = pd.DataFrame(
        data.scenarios_per_task.gen_normal_values_dict(n) |
        data.automate_scenario_in_minutes.gen_normal_values_dict(n) |
        data.maintain_test_per_year_in_minutes.gen_normal_values_dict(n)
    )
    if beneficiary.Tester or beneficiary.Both:
        df[data.manual_check_scenario_in_minutes.name] = data.manual_check_scenario_in_minutes.gen_normal_values(n)
        df[data.additional_checks_per_year.name] = data.additional_checks_per_year.gen_normal_values(n)
    if beneficiary.Developer or Beneficiary.Both:
        df[data.uncovered_scenario_fixes_per_year.name] = \
            data.uncovered_scenario_fixes_per_year.gen_normal_values(n)
        df[data.additional_fix_time_in_minutes.name] = data.additional_fix_time_in_minutes.gen_normal_values(n)
    return df


def calculate_economy(
        df: DataFrame,
        tasks_per_year: int,
        years: int,
        min_checks: int,
        result_column: str,
        data: TestCoverageApproximations,
        beneficiary: Beneficiary
) -> DataFrame:
    """
    Рассчитывают экономию для тестировщиков (на ручных проверках),
    для разработчиков (на починке дежурных багов по непокрытым сценариям),
    или для тех и других
    """
    match beneficiary:
        case Beneficiary.Tester:
            df = calculate_testers_economy(df, tasks_per_year, years, min_checks, result_column, data)
        case Beneficiary.Developer:
            df = calculate_developer_economy(df, tasks_per_year, years, result_column, data)
        case Beneficiary.Both:
            df = calculate_both_economy(df, tasks_per_year, years, min_checks, result_column, data)
        case _:
            raise ValueError(f"Некорректное значение Beneficiary: {beneficiary}")
    return df


def calculate_testers_economy(
        df: DataFrame,
        tasks_per_year: int,
        years: int,
        min_checks: int,
        result_column: str,
        data: TestCoverageApproximations
) -> DataFrame:
    """Рассчитывает экономию только на ручных проверках"""
    testers_spendings_per_scenario = "testers_spendings_per_scenario"
    df[testers_spendings_per_scenario] = df.apply(
        lambda row:
        row[data.manual_check_scenario_in_minutes.name] *
        (min_checks + row[data.additional_checks_per_year.name] * years),
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row[testers_spendings_per_scenario] - row[data.automate_scenario_in_minutes.name] - row[
            data.maintain_test_per_year_in_minutes.name])
        * row[data.scenarios_per_task.name] * tasks_per_year * years,
        axis=1
    )
    return df


def calculate_developer_economy(
        df: DataFrame,
        tasks_per_year: int,
        years: int,
        result_column: str,
        data: TestCoverageApproximations
) -> DataFrame:
    """Рассчитывает экономию только на фиксах багов в непокрытой функциональности"""
    dev_spendings_per_scenario = "dev_spendings_per_scenario"
    df[dev_spendings_per_scenario] = df.apply(
        lambda row:
        row[data.uncovered_scenario_fixes_per_year.name] * row[data.additional_fix_time_in_minutes.name] * years,
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row[dev_spendings_per_scenario] - row[data.automate_scenario_in_minutes.name]
         - row[data.maintain_test_per_year_in_minutes.name])
        * row[data.scenarios_per_task.name] * tasks_per_year * years,
        axis=1
    )
    return df


def calculate_both_economy(
        df: DataFrame,
        tasks_per_year: int,
        years: int,
        min_checks: int,
        result_column: str,
        data: TestCoverageApproximations
) -> DataFrame:
    """Рассчитывает экономию и на ручных проверках и на фиксах багов в непокрытой функциональности"""
    df["testers_spendings_per_scenario"] = df.apply(
        lambda row:
        row[data.manual_check_scenario_in_minutes.name] *
        (min_checks + row[data.additional_checks_per_year.name] * years), axis=1
    )
    df["dev_spendings_per_scenario"] = df.apply(
        lambda row:
        row[data.uncovered_scenario_fixes_per_year.name]
        * row[data.additional_fix_time_in_minutes.name] * years,
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row["testers_spendings_per_scenario"] + row["dev_spendings_per_scenario"] - row[
            data.automate_scenario_in_minutes.name])
        * row[data.scenarios_per_task.name] * tasks_per_year * years,
        axis=1
    )
    return df


def draw_and_print_coverage_case_result(df: DataFrame, result_column: str) -> None:
    """Рисует гистограмму с результатами и выводит шанс НЕ сэкономить на покрытии"""
    helpers.print_stats(df, result_column)
    helpers.draw_hist_with_average(
        df=df,
        series_name=result_column,
        label="Экономия",
        xlabel="Экономия в минутах",
        ylabel="Количество вариантов",
        show=True
    )
    plt.savefig("monte-coverage.jpg")
    working_hours_in_month = 159
    print(f"Среднее в рабочих месяцах: {df[result_column].mean() / 60 / working_hours_in_month}")
    helpers.output_fail_chance(0, df, result_column, "Шанс НЕ сэкономить")


if __name__ == "__main__":
    main()
