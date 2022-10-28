import os
from src import clean_df
from src import make_one_file
from src import make_features
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(override=True)

raw_dir = os.getenv("RAW_DIR")
clean_dir = os.getenv("CLEAN_DIR")
data_dir = os.getenv("DATA_DIR")

raw_names = os.getenv("RAW_DF_NAME")
raw_out = "df_raw_test.zip"
clean_name = "df_clean_test.zip"
work_name = "df_work_test.zip"

make_one_file(Path(raw_dir), raw_names, Path(raw_dir, raw_out))
print("One file made")
clean_df(in_file=Path(raw_dir, raw_out), out_file=Path(clean_dir, clean_name))
print("Data clean done")
make_features(in_file=Path(clean_dir, clean_name), out_file=Path(data_dir, work_name))
print("Make features done")


