import matplotlib
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, field_validator

matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from pandas import DataFrame
from enum import Enum
import helpers
from helpers import NormalVar, adjust_non_positive_values
import seaborn


class Beneficiary(Enum):
    """Получатель пользы от улучшения покрытия"""
    Tester = 1,
    Developer = 2,
    Both = 3


NON_NEGATIVE_COLS: list[str] = [
    "maintain_test_per_year_in_minutes",
    "additional_checks_per_year",
    "uncovered_scenario_fixes_per_year",
    "additional_fix_time_in_minutes",
    "scenario_checks_before_release"
]

POSITIVE_COLS: list[str] = [
    "scenarios_per_task",
    "automate_scenario_in_minutes",
    "manual_check_scenario_in_minutes",
    "tasks_per_year",
    "feature_lifespan_in_years"
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
    :param scenario_checks_before_release: минимальное количество проверок сценария во время тестирования задачи, например, 2 (это если предполагается написание автотестов до ручного тестирования, а если после - то 0, потому что автоматизация не поможет сократить эти проверки)
    :param tasks_per_year: количество задач в год, в которых есть хоть один сценарий, поддающийся автоматизации
    :param feature_lifespan_in_years: количество лет, которые в среднем проживают фичи
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenarios_per_task: NormalVar
    automate_scenario_in_minutes: NormalVar
    manual_check_scenario_in_minutes: NormalVar
    additional_checks_per_year: NormalVar
    uncovered_scenario_fixes_per_year: NormalVar
    additional_fix_time_in_minutes: NormalVar
    maintain_test_per_year_in_minutes: NormalVar
    scenario_checks_before_release: NormalVar
    tasks_per_year: NormalVar
    feature_lifespan_in_years: NormalVar

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
    input_data = TestCoverageApproximations(
        scenarios_per_task=NormalVar(1, 21, "scenarios_per_task"),
        automate_scenario_in_minutes=NormalVar(20, 100, "automate_scenario_in_minutes"),
        maintain_test_per_year_in_minutes=NormalVar(5, 40, "maintain_test_per_year_in_minutes"),
        manual_check_scenario_in_minutes=NormalVar(5, 60, "manual_check_scenario_in_minutes"),
        additional_checks_per_year=NormalVar(0, 2, "additional_checks_per_year"),
        uncovered_scenario_fixes_per_year=NormalVar(0.1, 0.3, "uncovered_scenario_fixes_per_year"),
        additional_fix_time_in_minutes=NormalVar(20, 100, "additional_fix_time_in_minutes"),
        tasks_per_year=NormalVar(150, 250, "tasks_per_year"),
        feature_lifespan_in_years=NormalVar(4, 6, "feature_lifespan_in_years"),
        scenario_checks_before_release=NormalVar(1, 3, "scenario_checks_before_release")
    )
    np.random.seed(42)
    beneficiary = Beneficiary.Both
    df = prepare_df(input_data, beneficiary, 10000)
    df = adjust_non_positive_values(
        df,
        list(set(NON_NEGATIVE_COLS) - {input_data.scenario_checks_before_release.name}),
        list(set(POSITIVE_COLS) - {input_data.tasks_per_year.name, input_data.feature_lifespan_in_years.name})
    )

    economy_in_minutes = "economy_in_minutes"
    df = calculate_economy_in_minutes(df, economy_in_minutes, input_data, beneficiary)

    economy_in_working_months = "economy_in_working_months"
    working_hours_in_month = 159
    minutes_in_hour = 60
    df[economy_in_working_months] = df.apply(
        lambda row: row[economy_in_minutes] / minutes_in_hour / working_hours_in_month, axis=1
    )
    seaborn.histplot(data=df[economy_in_working_months])
    plt.axvline(df[economy_in_working_months].mean(), color="grey", linestyle='--', label="Среднее")
    plt.xlim(None, df[economy_in_working_months].mean() + 4 * df[economy_in_working_months].std())
    plt.savefig("coverage.jpg")
    plt.show()
    print(f"Среднее: {df[economy_in_working_months].mean()}")
    print(f"Медиана: {df[economy_in_working_months].median()}")
    print(f"Шанс НЕ сэкономить: {helpers.calculate_fail_chance(df, 0, economy_in_working_months)}%")


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


def calculate_economy_in_minutes(
        df: DataFrame,
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
            df = calculate_testers_economy(df, result_column, data)
        case Beneficiary.Developer:
            df = calculate_developer_economy(df, result_column, data)
        case Beneficiary.Both:
            df = calculate_both_economy(df, result_column, data)
        case _:
            raise ValueError(f"Некорректное значение Beneficiary: {beneficiary}")
    return df


def calculate_testers_economy(
        df: DataFrame,
        result_column: str,
        data: TestCoverageApproximations
) -> DataFrame:
    """Рассчитывает экономию только на ручных проверках"""
    testers_spendings_per_scenario = "testers_spendings_per_scenario"
    df[testers_spendings_per_scenario] = df.apply(
        lambda row:
        row[data.manual_check_scenario_in_minutes.name] *
        (data.scenario_checks_before_release.average + row[
            data.additional_checks_per_year.name] * data.feature_lifespan_in_years.average),
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row[testers_spendings_per_scenario] - row[data.automate_scenario_in_minutes.name] - row[
            data.maintain_test_per_year_in_minutes.name])
        * row[data.scenarios_per_task.name] * data.tasks_per_year.average * data.feature_lifespan_in_years.average,
        axis=1
    )
    return df


def calculate_developer_economy(
        df: DataFrame,
        result_column: str,
        data: TestCoverageApproximations
) -> DataFrame:
    """Рассчитывает экономию только на фиксах багов в непокрытой функциональности"""
    dev_spendings_per_scenario = "dev_spendings_per_scenario"
    df[dev_spendings_per_scenario] = df.apply(
        lambda row:
        row[data.uncovered_scenario_fixes_per_year.name] * row[
            data.additional_fix_time_in_minutes.name] * data.feature_lifespan_in_years.average,
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row[dev_spendings_per_scenario] - row[data.automate_scenario_in_minutes.name]
         - row[data.maintain_test_per_year_in_minutes.name])
        * row[data.scenarios_per_task.name] * data.tasks_per_year.average * data.feature_lifespan_in_years.average,
        axis=1
    )
    return df


def calculate_both_economy(
        df: DataFrame,
        result_column: str,
        data: TestCoverageApproximations
) -> DataFrame:
    """Рассчитывает экономию и на ручных проверках и на фиксах багов в непокрытой функциональности"""
    df["testers_spendings_per_scenario"] = df.apply(
        lambda row:
        row[data.manual_check_scenario_in_minutes.name] *
        (data.scenario_checks_before_release.average + row[
            data.additional_checks_per_year.name] * data.feature_lifespan_in_years.average),
        axis=1
    )
    df["dev_spendings_per_scenario"] = df.apply(
        lambda row:
        row[data.uncovered_scenario_fixes_per_year.name]
        * row[data.additional_fix_time_in_minutes.name] * data.feature_lifespan_in_years.average,
        axis=1
    )
    df[result_column] = df.apply(
        lambda row:
        (row["testers_spendings_per_scenario"] + row["dev_spendings_per_scenario"] - row[
            data.automate_scenario_in_minutes.name])
        * row[data.scenarios_per_task.name] * data.tasks_per_year.average * data.feature_lifespan_in_years.average,
        axis=1
    )
    return df


if __name__ == "__main__":
    main()
