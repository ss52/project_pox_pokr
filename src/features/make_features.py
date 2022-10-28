import pandas as pd
from dotenv import load_dotenv
import os
import click
from pathlib import Path
import numpy as np

# load_dotenv(override=True)
#
# CLEAN_DF_PATH = os.getenv("CLEAN_DIR")
# WORK_DF_PATH = os.getenv("DATA_DIR")
#
# CLEAN_FILE_NAME = os.getenv("CLEAN_DF_ALL_NAME")
# WORK_FILE_NAME = os.getenv("WORK_DF_ALL_NAME")
#
# in_file_path = f'{CLEAN_DF_PATH}\\{CLEAN_FILE_NAME}'
# out_file_path = f'{WORK_DF_PATH}\\{WORK_FILE_NAME}'


@click.command()
@click.argument('in_file', type=click.Path())
@click.argument('out_file', type=click.Path())
def make_features(in_file: str, out_file: str) -> None:
    """
    Add features to df

    Нужно добавить следующие данные:
        + интерполяция ХА
        + Содержание твердого - C_S
        + Расход пульпы - Q_SL
        + Расход твердого - G_S
        + Расход серного эквивалента - G_Seq
        - количество пирита, т/ч
        - количество арсенопирита, т/ч
        - стехиометрический расход кислорода на полное окисление, нм³/т
        - отношение реально поданного кислорода к стехиометрии, доли
        - степень окисления материала по секциям по расходу кислорода от общего кол-ва, доли
        -

    Константы:
        - Максимальная производительность насоса Feluwa 17,45 м³/ч
        - Максимальное число шагов 51
        - Мышьяк в СЭ - 0,371


    Args:
        in_file: File path to clean df
        out_file: File path to work df

    Returns: new df file
    """
    FELUWA_MAX_Q = 17.45
    FELUWA_MAX_N = 51
    COEF_AS_SEQ = 0.371
    D_WATER = 1000

    df = pd.read_csv(in_file, index_col=0, parse_dates=True)

    # chem interpolate
    int_cols = [
        'Fe',
        'Stot',
        'As',
        'Corg',
        'Ctot',
        'D_S',
        'D_SL_H'
    ]

    df[int_cols] = df.loc[:, int_cols].interpolate(method='time')

    # new features
    fel_sum = df['Fel_1'] + df['Fel_2']
    df = df.assign(Fel_sum=fel_sum.values)

    df = df.assign(C_S=(df['D_S'] * D_WATER - df['D_SL_H'] * df['D_S']) / (
                df['D_SL_H'] * D_WATER - df['D_SL_H'] * df['D_S']) * 100)
    df = df.assign(Q_SL=df['Fel_sum'] / FELUWA_MAX_N * FELUWA_MAX_Q)
    df = df.assign(G_S=df['Q_SL'] * df['D_SL_H'] / 1000 * (df['C_S'] / 100))
    df = df.assign(G_Seq=(COEF_AS_SEQ * df['As'] / 100 + df['Stot'] / 100) * df['G_S'])

    # delete columns
    columns_drop = [
        "Fel_1",
        "Fel_2",
        "D_SL",
        "O2_tot",
        "Sl_tot",
        "AC_rbk_open",
        "AC_valve_open",
        "P_O2_in",
        "P_tot_2",
        "FT1_level",
        "FT1_P",
        "FT1_T_in",
        "FT1_T",
        "FT2_level",
        "FT2_P",
        "FT2_T",
        "Cond_Q",
        "Cond_level",
        "Cond_valve",
        "O2_conc_1",
        "O2_conc_2",
        'T_C1',
        'T_C2',
        'T_C3',
        'T_C4',
        'T_C5'
    ]

    df = df.drop(columns_drop, axis=1)
    df = df.dropna()

    # save file
    comp = {
        'method': 'zip',
        'archive_name': 'out.csv'
    }

    df.to_csv(out_file, compression=comp)


if __name__ == "__main__":
    make_features()
