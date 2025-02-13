import polars as pl
import pandas as pd
import torch

class DataframeConfigs:

    def __init__(self):
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 1000)

        # pl.Config.set_tbl_cols(1000)
        # pl.Config.set_tbl_width_chars(230)
        # pl.Config.set_fmt_str_lengths(700)
        # pl.Config.set_tbl_rows(15)
        # pl.Config.set_fmt_table_cell_list_len(700)
        # pl.Config.set_tbl_column_data_type_inline(True)

        pl.Config.set_tbl_cols(1000)
        pl.Config.set_tbl_width_chars(500)
        pl.Config.set_fmt_str_lengths(1500)
        pl.Config.set_tbl_rows(40)
        pl.Config.set_fmt_table_cell_list_len(100)
        pl.Config.set_tbl_column_data_type_inline(True)