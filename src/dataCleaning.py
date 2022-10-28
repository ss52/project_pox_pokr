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

def dataCleaning (dataPath, savePath) -> None:
   
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
        
    #Fe2+
    a = np.array(df_work['Fe2+'].values.tolist())
    df_work['Fe2+'] = np.where(a > Fe2_MAX, np.NaN, a).tolist()
    df_work['Fe2+'].interpolate(method='time', inplace=True)

    df_work.loc[df_work['Fe2+'] == 0, :]
    df_work.loc[df_work['Fe2+'] == 0, :]= df_work.replace(0, np.nan)
    df_work.loc[df_work['Fe2+'] == 0, :]= df_work.dropna(how='all', axis=0)
    df_work['Fe2+'].interpolate(method='time', inplace=True)
    
    #исходное сырьё
    df_work.loc[(df_work['Fe'] <= 10) & (df_work['Stot'] <= 10), ['Fe', 'Stot']].count()
    int_cols = [
    'Fe',
    'Stot',
    'As',
    'Corg',
    'Ctot']
    df_work.loc[df_work['Corg'] > df_work['Ctot'], 'Corg'] = df_work.loc[df_work['Corg'] > df_work['Ctot'], 'Ctot']
    #Fe
    b = np.array(df_work['Fe'].values.tolist())
    df_work['Fe'] = np.where(b > Fe_MAX, np.NaN, b).tolist()
    df_work['Fe'] = np.where(b <= 0, np.NaN, b).tolist()
    df_work['Fe'].interpolate(method='time', inplace=True)
    #Stot
    c = np.array(df_work['Stot'].values.tolist())
    df_work['Stot'] = np.where(c > Stot_MAX, np.NaN, c).tolist()
    df_work['Stot'] = np.where(c <= 0, np.NaN, c).tolist()
    df_work['Stot'].interpolate(method='time', inplace=True)
    #As
    d = np.array(df_work['As'].values.tolist())
    df_work['As'] = np.where(d > As_MAX, np.NaN, d).tolist()
    df_work['As'] = np.where(d <= 0, np.NaN, d).tolist()
    df_work['As'].interpolate(method='time', inplace=True)
    #Corg
    e = np.array(df_work['Corg'].values.tolist())
    df_work['Corg'] = np.where(e > Corg_MAX, np.NaN, e).tolist()
    df_work['Corg'] = np.where(e <= 0, np.NaN, e).tolist()
    df_work['Corg'].interpolate(method='time', inplace=True)
    #Ctot
    f = np.array(df_work['Ctot'].values.tolist())
    df_work['Ctot'] = np.where(f > Ctot_MAX, np.NaN, f).tolist()
    df_work['Ctot'] = np.where(f <= 0, np.NaN, f).tolist()
    df_work['Ctot'].interpolate(method='time', inplace=True)
    
    #Расход воды по секциям и общий расход
    QQ_cols = [
    'QQ_C1',
    'QQ_C2',
    'QQ_C3',
    'QQ_C4',
    'QQ_C5',
    'QQ_tot',]
    q1 = np.array(df_work['QQ_C1'].values.tolist())
    df_work['QQ_C1'] = np.where(q1 <= 0, np.NaN, q1).tolist()
    df_work['QQ_C1'].interpolate(method='time', inplace=True)

    q2 = np.array(df_work['QQ_C2'].values.tolist())
    df_work['QQ_C2'] = np.where(q2 <= 0, np.NaN, q2).tolist()
    df_work['QQ_C2'].interpolate(method='time', inplace=True)

    q3 = np.array(df_work['QQ_C3'].values.tolist())
    df_work['QQ_C3'] = np.where(q3 <= 0, np.NaN, q3).tolist()
    df_work['QQ_C3'].interpolate(method='time', inplace=True)
    
    q4 = np.array(df_work['QQ_C4'].values.tolist())
    df_work['QQ_C4'] = np.where(q4 <= 0, np.NaN, q4).tolist()
    df_work['QQ_C4'].interpolate(method='time', inplace=True)
    
    q5 = np.array(df_work['QQ_C5'].values.tolist())
    df_work['QQ_C5'] = np.where(q5 <= 0, np.NaN, q5).tolist()
    df_work['QQ_C5'].interpolate(method='time', inplace=True)
    #Температура
    #можно смотреть только не температуру абгаза
    T_cols = [
    'T_C1',
    'T_C2',
    'T_C3',
    'T_C4',
    'T_C5',
    'T_abg']
    
    t = np.array(df_work['T_abg'].values.tolist())
    df_work['T_abg'] = np.where(t <= 0, np.NaN, t).tolist()
    df_work['T_abg'] = np.where(t > T_MAX, np.NaN, t).tolist()
    df_work['T_abg'].interpolate(method='time', inplace=True)
    
    #Давление в автоклаве и расход кислорода
    O2_cols = [
    'O2_C1',
    'O2_C2',
    'O2_C3',
    'O2_C4',
    'O2_C5',
    'O2_conc_1',
    'O2_conc_2',]

    P_cols = [
    'P_H2O',
    'P_O2',
    'P_tot',
    'P_valve']
    p1 = np.array(df_work['O2_C1'].values.tolist())
    df_work['O2_C1'] = np.where(p1 <= 0, np.NaN, p1).tolist()
    df_work['O2_C1'].interpolate(method='time', inplace=True)

    p2 = np.array(df_work['O2_C2'].values.tolist())
    df_work['O2_C2'] = np.where(p2 <= 0, np.NaN, p2).tolist()
    df_work['O2_C2'].interpolate(method='time', inplace=True)

    p3 = np.array(df_work['O2_C3'].values.tolist())
    df_work['O2_C3'] = np.where(p3 <= 0, np.NaN, p3).tolist()
    df_work['O2_C3'].interpolate(method='time', inplace=True)

    p4 = np.array(df_work['O2_C4'].values.tolist())
    df_work['O2_C4'] = np.where(p4 <= 0, np.NaN, p4).tolist()
    df_work['O2_C4'].interpolate(method='time', inplace=True)

    p5 = np.array(df_work['O2_C5'].values.tolist())
    df_work['O2_C5'] = np.where(p5 <= 0, np.NaN, p5).tolist()
    df_work['O2_C5'].interpolate(method='time', inplace=True)

    #Данные по чистоте кислорода использовать, скорее всего, нецелесообразно ввиду их практически постоянства.

    #Уровень в автоклаве и клапан сброса
    l_cols = [
    'AC_level',
    'AC_valve_open']
    
    ###
    df_work[df_work < 0] = 0
    ###
    file_save(df_work, WORK_DIR, 'df_work_ds_final')
    
dataCleaning (dataPath, savePath)   

