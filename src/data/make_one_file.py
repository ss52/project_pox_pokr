import pandas as pd
from dotenv import load_dotenv
import os
import click
from pathlib import Path

# load_dotenv(override=True)
#
# WORK_DIR = os.getenv("RAW_DIR")
# FILE_NAME = os.getenv("RAW_DF_NAME")
#
# out_file_path = "df_total_all.zip"


@click.command()
@click.argument('work_dir', type=click.Path())
@click.argument('file_name', type=click.STRING)
@click.argument('out_file', type=click.Path())
def make_one_file(work_dir: str, file_name: str, out_file: str) -> None:
    """
    Makes one file from 4 autoclaves. New names for columns

    Args:
        work_dir: Working directory to find files
        file_name: Names of files without AC number and extension
        out_file: Out full file name

    Returns: None. Save new DF in csv format in work folder

    """
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

    p = Path(work_dir).glob(f'{file_name}*.zip')
    files = [x for x in p if x.is_file()]

    df_new = pd.DataFrame()

    for i, file in enumerate(files):
        df_temp = pd.read_csv(file, index_col=0, parse_dates=True)
        df_temp.columns = cols
        df_temp = df_temp.assign(ac=f"10{i + 1}")
        df_new = pd.concat((df_new, df_temp))
        print(f'Concat AC 10{i + 1}')

    comp = {
        'method': 'zip',
        'archive_name': 'out.csv'
    }

    save_path = out_file

    df_new.to_csv(save_path, compression=comp)


if __name__ == "__main__":
    make_one_file()
