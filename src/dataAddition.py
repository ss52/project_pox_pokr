#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import seaborn as sns

from pathlib import Path

from datetime import datetime

import matplotlib.dates as mdates
import seaborn as sns

from pandas import testing as tm

from sklearn.metrics import mean_absolute_error

import lib



dataPath = "C:\\Users\\Laevskiy\\Desktop\\05-22\\AC101"
savePath = "C:\\Users\\Laevskiy\\Desktop\\05-22\\AC101"

def dataAddition (dataPath, savePath) -> None:
   
    WORK_DIR = dataPath
    #Константы
    FELUWA_MAX_Q = 17.45
    FELUWA_MIN_N= 20
    FELUWA_MAX_N = 51
    COEF_AS_SEQ = 0.371
    D_WATER = 1000
    Fe_MAX= 50
    As_MAX  = 30
    Stot_MAX = 50
    Corg_MAX = 5
    Ctot_MAX = 5
    Fe2_MAX= 35.0
    P_H2O_MAX= 30
    P_O2_MAX = 10
    T_MAX = 250
    DENSITY_MAX = 4001
    
    def file_save(df: pd.DataFrame, folder: str, name: str) -> None:
        comp = {
            'method': 'zip',
            'archive_name': 'out.csv'}

        filepath = Path(f'./{folder}/{name}.zip')

        df.to_csv(filepath, compression=comp)
       
    df_work = pd.read_csv(f'{WORK_DIR}\df_work_ds.zip', index_col=0, parse_dates=True)
    
#     добавляем:
#     Содержание твердого - C_S
#     Расход пульпы - Q_SL
#     Расход твердого - G_S
#     Расход серного эквивалента - G_Seq
    
    fel_sum = df_work['Fel_1'] + df_work['Fel_2']

    df_work = df_work.assign(Fel_sum = fel_sum.values)
    df_work = df_work.assign(C_S = (df_work['D_S'] * D_WATER - df_work['D_SL_H'] * df_work['D_S']) / (df_work['D_SL_H'] * D_WATER - df_work['D_SL_H'] * df_work['D_S']) * 100)
    df_work = df_work.assign(Q_SL = df_work['Fel_sum'] / FELUWA_MAX_N * FELUWA_MAX_Q)
    df_work = df_work.assign(G_S = df_work['Q_SL'] * df_work['D_SL_H'] / 1000 * (df_work['C_S'] / 100))
    df_work = df_work.assign(G_Seq = (COEF_AS_SEQ * df_work['As'] / 100 + df_work['Stot'] / 100) * df_work['G_S'])
    df_work.drop('SO4', axis=1, inplace=True)
    df_work.dropna(axis=0, inplace=True)
    
    file_save(df_work, WORK_DIR, 'df_work_ds')
    
dataAddition(dataPath, savePath)
