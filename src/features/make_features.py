import pandas as pd
# from dotenv import load_dotenv
# import os
import click
# from pathlib import Path
# import numpy as np

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


def add_lags_inplace(df: pd.DataFrame, num_lags: int, for_ac: bool = True) -> None:

    if for_ac:
        for lag in range(1, num_lags + 1):
            lag_name = f'lag_{lag}'
            df[lag_name] = df.groupby('ac')['Fe2+'].shift(lag)
    else:
        for lag in range(1, num_lags + 1):
            lag_name = f'lag_{lag}'
            df[lag_name] = df['Fe2+'].shift(lag)

    df.dropna(axis=0, inplace=True)


@click.command()
@click.argument('in_file', type=click.Path())
@click.argument('out_file', type=click.Path())
def make_features(in_file: str, out_file: str) -> None:
    """
    Add features to df

    Нужно добавить следующие данные:
        + интерполяция ХА
        + Содержание твердого - C_Solid
        + Расход пульпы - Q_SL
        + Расход твердого - G_Solid
        + Расход серного эквивалента - G_Seq
        + количество пирита, т/ч
        + количество арсенопирита, т/ч
        + стехиометрический расход кислорода на полное окисление, нм³/т
        + отношение реально поданного кислорода к стехиометрии, доли
        + степень окисления материала по секциям по расходу кислорода от общего кол-ва, доли - BettaS_O2_X
        + степень окисления материала по секциям по расходу ОВ, доли - BettaS_QW_X
        + добавим признаки сдвижки по времени, данные по концентрации железа за прошлые 3 часа

    Константы:
        - Максимальная производительность насоса Feluwa 17,45 м³/ч
        - Максимальное число шагов 51
        - Мышьяк в СЭ - 0,371
        - Молярная масса Fe - 55,85 г/моль
        - Молярная масса S - 32,06 г/моль
        - Молярная масса As - 74,92 г/моль


    Args:
        in_file: File path to clean df
        out_file: File path to work df

    Returns: new df file
    """
    FELUWA_MAX_Q = 17.45
    FELUWA_MAX_N = 51
    COEF_AS_SEQ = 0.371
    D_WATER = 1000

    M_As = 74.92
    M_Fe = 55.85
    M_S = 32.06

    M_FeS2 = M_Fe + 2 * M_S
    M_FeAsS = M_Fe + M_As + M_S

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
    # fel_sum = df['Fel_1'] + df['Fel_2']
    # df = df.assign(Fel_sum=fel_sum.values)

    df = df.assign(C_Solid=(df['D_S'] * D_WATER - df['D_SL_H'] * df['D_S']) / (
                df['D_SL_H'] * D_WATER - df['D_SL_H'] * df['D_S']) * 100)
    df = df.assign(Q_SL=df['Fel_sum'] / FELUWA_MAX_N * FELUWA_MAX_Q)
    df = df.assign(G_Solid=df['Q_SL'] * df['D_SL_H'] / 1000 * (df['C_Solid'] / 100))
    df = df.assign(G_Seq=(COEF_AS_SEQ * df['As'] / 100 + df['Stot'] / 100) * df['G_Solid'])

    df = df.assign(G_FeS2=((df['Stot'] - df['As'] / M_As * M_S) / 2 / M_S * M_FeS2) / 100 * df['G_Solid'])
    df = df.assign(G_FeAsS=(df['As'] / M_As * M_FeAsS) / 100 * df['G_Solid'])

    df = df.assign(G_O2_st=df['G_FeS2'] / 2 / M_FeS2 * 7.5 * 22.4 + df['G_FeAsS'] / M_FeAsS * 3.5 * 22.4)
    df = df.assign(O2_part=df['G_O2_st'] / df['O2_tot'])

    df = df.assign(BettaS_O2_1=df['O2_C1'] / df['O2_tot'])
    df = df.assign(BettaS_O2_2=df['O2_C2'] / df['O2_tot'])
    df = df.assign(BettaS_O2_3=df['O2_C3'] / df['O2_tot'])
    df = df.assign(BettaS_O2_4=df['O2_C4'] / df['O2_tot'])
    df = df.assign(BettaS_O2_5=df['O2_C5'] / df['O2_tot'])

    df = df.assign(QQ_tot_sl=df['QQ_tot'] + + df['Q_SL'] - df['G_Solid'] / df['D_S'])

    df = df.assign(BettaS_QW_1=(df['QQ_C1'] + df['Q_SL'] - df['G_Solid'] / df['D_S']) / df['QQ_tot_sl'])
    df = df.assign(BettaS_QW_2=df['QQ_C2'] / df['QQ_tot_sl'])
    df = df.assign(BettaS_QW_3=df['QQ_C3'] / df['QQ_tot_sl'])
    df = df.assign(BettaS_QW_4=df['QQ_C4'] / df['QQ_tot_sl'])
    df = df.assign(BettaS_QW_5=df['QQ_C5'] / df['QQ_tot_sl'])

    # сдвижка по времени
    add_lags_inplace(df, 3)

    # delete columns
    columns_drop = [
        "Fel_1",
        "Fel_2",
        "D_SL",
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
