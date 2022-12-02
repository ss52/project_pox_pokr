import pandas as pd
# from dotenv import load_dotenv
# import os
import click
import numpy as np
# from pathlib import Path


# load_dotenv(override=True)
# RAW_DF_PATH = os.getenv("RAW_DIR")
# CLEAN_DF_PATH = os.getenv("CLEAN_DIR")
#
# RAW_FILE_NAME = os.getenv("RAW_DF_ALL_NAME")
# CLEAN_FILE_NAME = os.getenv("CLEAN_DF_ALL_NAME")
#
# in_file_path = f'{RAW_DF_PATH}\\{RAW_FILE_NAME}'
# out_file_path = f'{CLEAN_DF_PATH}\\{CLEAN_FILE_NAME}'


@click.command()
@click.argument('in_file', type=click.Path())
@click.argument('out_file', type=click.Path())
def clean_df(in_file: str, out_file: str) -> None:
    """
    Function for cleaning dataset

    Описание правил:
    0. Убираем то время, что автоклав не работал
    1. Убираем все данные с концентрацией железа выше 20 г/л. 20 - константа
    2. Убираем все данные с концентрацией железа = 0. - Попробуем заменить на Nan, потом интерполировать
    3. Колонку SO4 удалим, данных мало, анализы странные
    4. По Сорг удалим все данные больше 2 %
    5. По Стот удалим все данные больше 5 %
    6. Удалить все значения мышьяка больше 16 %
    7. Почистить данные по плотности пульпы. Константа определена в EDA

    Args:
        in_file: file path for raw df
        out_file: file path for clean df

    Returns: new df file

    """
    FE2_MAX = 20
    FE2_MIN = 0
    CORG_MAX = 2
    CTOT_MAX = 5
    AS_S_MAX = 16
    D_SL_H_MAX = 1868

    # read file
    df = pd.read_csv(in_file)

    # clean file
    # df = df.drop(df[df['Fe2+'] == FE2_MIN].index)
    df = df.drop(df[df['Fe2+'] > FE2_MAX].index)
    df = df.replace({'Fe2+': {0, np.NaN}})

    df = df.drop(df[df['Corg'] > CORG_MAX].index)
    df = df.drop(df[df['Ctot'] > CTOT_MAX].index)
    df = df.drop(df[df['As'] > AS_S_MAX].index)

    df = df.drop(df[df['D_SL_H'] > D_SL_H_MAX].index)

    # удаляем SO4
    df = df.drop('SO4', axis=1)

    # only work data
    fel_sum = df['Fel_1'] + df['Fel_2']
    df = df.assign(Fel_sum=fel_sum.values)

    df['work'] = np.where((df['Fel_sum'] >= 30), 1, 0)
    # df = df.reset_index()
    df = df.drop(df[df['work'] == 0].index)
    df = df.drop('work', axis=1)

    # save file
    comp = {
        'method': 'zip',
        'archive_name': 'out.csv'
    }

    df.to_csv(out_file, compression=comp, index=False)


if __name__ == "__main__":
    clean_df()
