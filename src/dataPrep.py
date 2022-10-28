#!/usr/bin/env python
# coding: utf-8

# In[9]:


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

FELUWA_MAX_Q = 17.45
FELUWA_MIN_N= 20
FELUWA_MAX_N = 51
COEF_AS_SEQ = 0.371
D_WATER = 1000

def dataPrep (dataPath, savePath) -> None:
    ac_data = dataPath+"\\raw data\\mergeSortedbyDate.xlsx"
    chem_in = dataPath+"\\raw data\\!chem баки питания.xlsx"
    chem_out = dataPath+"\\raw data\\!chem окисленная пульпа.xlsx"
    chem_fe2 = dataPath+"\\raw data\\!chem Fe2+.xlsx"
    path_19 = dataPath+"\\raw data\\Nikitos19.xlsx"
    path_20 = dataPath+"\\raw data\\Nikitos20.xlsx"
    path_21 = dataPath+"\\raw data\\Nikitos21.xlsx"

    ds_19 = pd.read_excel(path_19,  usecols='B:D', index_col=0, parse_dates=True)
    ds_20 = pd.read_excel(path_20,  usecols='B:D', index_col=0, parse_dates=True)
    ds_21 = pd.read_excel(path_21,  usecols='B:D', index_col=0, parse_dates=True)
    
    WORK_DIR = dataPath
       
    def file_save(df: pd.DataFrame, folder: str, name: str) -> None:
        comp = {
            'method': 'zip',
            'archive_name': 'out.csv'}

        filepath = Path(f'./{folder}/{name}.zip')

        df.to_csv(filepath, compression=comp)
    
    
    #Плотность
    ds_full = pd.concat([ds_19, ds_20, ds_21])
    ds_full = ds_full.apply(pd.to_numeric, errors='coerce')
    ds_full_1h = ds_full.resample('H', convention='start').first()
    
    #Данные АСУТП
    df_ac = pd.read_excel(ac_data, index_col=1, parse_dates=True)
    df_ac.drop("Unnamed: 0", axis=1, inplace=True)
    df_ac.dropna(inplace=True)
    df_ac_1h = df_ac.resample('H').mean()
    df_ac_1h.dropna(inplace=True)
    
    #Баки питания
    df_chem_in = pd.read_excel(chem_in, index_col=0, parse_dates=True, usecols="B:H")
    df_chem_in = df_chem_in.apply(pd.to_numeric, errors='coerce')
    df_chem_in.fillna(0, inplace=True)
    df_chem_in_1h = df_chem_in.resample('H', convention='start').first()
    cols = ['Fe', 'Stot', 'SO4', 'As', 'Corg', 'Ctot']
    df_chem_in_1h.columns = cols
    
    min_date = datetime.fromisoformat('2019-01-01 00:00:00')
    max_date = datetime.fromisoformat('2021-12-31 17:00:00')

    df_chem_in_1h = df_chem_in_1h[min_date:max_date]
    df_chem_in_1h.replace({0: np.NaN}, inplace=True)
    
    #Fe2+
    
    df_chem_fe2 = pd.read_excel(chem_fe2, index_col=0, parse_dates=True, usecols="B,C")
    df_chem_fe2 = df_chem_fe2.apply(pd.to_numeric, errors='coerce')
    df_chem_fe2.fillna(0, inplace=True)
    df_chem_fe2_1h = df_chem_fe2.resample('H').first()
    df_chem_fe2_1h = df_chem_fe2_1h[min_date:max_date]
    df_chem_fe2_1h.fillna(method='ffill', inplace=True)
    cols = ['Fe2+']
    df_chem_fe2_1h.columns = cols
    
    ###Свадная таблица
    df1 = df_ac_1h.merge(df_chem_in_1h, how='inner', left_index=True, right_index=True)
    df1 = df1.merge(df_chem_fe2_1h, how='inner', left_index=True, right_index=True)
    df1 = df1.merge(ds_full_1h, how='inner', left_index=True, right_index=True)
    # Сдвиг на два часа
    df1['Fe2+'] = df1['Fe2+'].shift(-2)
    
    file_save(df1, WORK_DIR, 'df_total_ds_1h')
        
    #Подготовка данных

    filepath = Path(f'./{WORK_DIR}/df_total_ds_1h.zip')
    df = pd.read_csv(filepath, index_col=0)
    df.index.name = None
    df.index = pd.to_datetime(df.index)
    
    cols = [
    'Fel_1',
    'Fel_2',
    'D_SL',
    'QQ_C1',
    'QQ_C2',
    'QQ_C3',
    'QQ_C4',
    'QQ_C5',
    'O2_tot',
    'QQ_tot',
    'Sl_tot',
    'O2_C1',
    'O2_C2',
    'O2_C3',
    'O2_C4',
    'O2_C5',
    'AC_level',
    'AC_rbk_open',
    'AC_valve_open',
    'P_H2O',
    'P_O2',
    'P_tot',
    'P_valve',
    'P_tot_2',
    'T_C1',
    'T_C2',
    'T_C3',
    'T_C4',
    'T_C5',
    'T_abg',
    'P_O2_in',
    'FT1_level',
    'FT1_P',
    'FT1_T_in',
    'FT1_T',
    'FT2_level',
    'FT2_P',
    'FT2_T',
    'Cond_Q',
    'Cond_level',
    'Cond_valve',
    'O2_conc_1',
    'O2_conc_2',
    'Fe',
    'Stot',
    'SO4',
    'As',
    'Corg',
    'Ctot',
    'Fe2+',
    'D_S',
    'D_SL_H'
    ]
    df.columns = cols
    #Добавим некоторые новые признаки и уберем данные, за то время когда автоклав не работал.
    #Уберем данные за время простоев автоклава Заполним данные ХА - df.loc[:, ['Fe']].interpolate(method='time')
    df = df.assign(work = np.where((df['Fel_1'] + df['Fel_2'] >= FELUWA_MIN_N), 1, 0))
    #Создадим новый массив данных, убираем простои автоклава.
    df_work = df.drop(df[df['work'] == 0].index).drop('work', axis=1)
    #Заполняем недостающие строки по ХА.
    int_cols = [
    'Fe',
    'Stot',
    'SO4',
    'As',
    'Corg',
    'Ctot',
    'D_S',
    'D_SL_H']
    df_work[int_cols] = df_work.loc[:, int_cols].interpolate(method='time')
   
    file_save(df_work, WORK_DIR, 'df_work_ds')
    
       
dataPrep(dataPath, savePath)

