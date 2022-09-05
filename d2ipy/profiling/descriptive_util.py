"""
This module is for explaining descriptive stats of a dataframe.
"""
import sys
from typing import Dict, Any, Union, List

import pandas as pd
from numpy import ndarray
from pandas import Series


class MetaData:
    """
    To Create metadata of a dataframe which will provide various details like
    number of records, number of columns, memory size, fill rate etc.

    Attributes:
        self._df: pd.DataFrame
            The dataFrame which will be used.
        self._meta_dict: dict
            The results will be provided in self._meta_dict
        self._n_records: int
            Number of records in this dataframe
        self._n_columns: int
            Number of columns in this dataframe
        self._fill_rate: int
            Percentage of records which have values

    Methods:
        get_meta(): dict
            return meta_dict which contains the metadata
    """

    # TODO: Add Error class, if a dataframe contains less than 3 records.

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializer for the class

        Args:
            df: pd.DataFrame
        """
        self._df = df
        self.meta_dict = None
        self._n_records = None
        self._n_columns = None
        self._fill_rate = None
        self._memory_size = None

    def get_meta(self) -> dict:
        """
        Returns the dictionary which contains all the metadata
        => self.meta_dict
        """
        self._n_records = self._df.shape[0]
        self._n_columns = self._df.shape[1]
        self._fill_rate = (
            (self._df.notna().sum().sum()) * 100 / (self._n_records * self._n_columns)
        )
        self._memory_size = sys.getsizeof(self._df)
        self.meta_dict = {
            "Num_Records": self._n_records,
            "Num_Columns": self._n_columns,
            "Fill_Rate": self._fill_rate,
            "Memory Size": self._memory_size,
        }
        return self.meta_dict


class DescriptiveDetails:
    """
    To create various descriptive statistics around the dataframe.

    Attributes:
        self._df: pd.DataFrame
            The dataframe which will be used
        self._col_meta: pd.DataFrame
            More meta details around the dataframe
        self._num_cols: list
            The list of column which are of numeric datatype.
        self._num_col_df: pd.DataFrame
            The sliced dataframe which contains only numeric columns
        self._descriptive_dict: dict
            The dictionary which return descriptive details like min,
            max, range, percentiles, IQR, and average
        self._eligible_cols: dict
            The list of columns which don't have all nan or a constant
        self._quantile_dict: dict
            The dictionary which contains quantile information like,
            standard deviation, median absolute deviation, skewness,
            variance and monotonicity.
        self._obj_cols_dict: dict
            Contains categorical column details like value counts,
            memory size etc.
        self._date_col_details: dict
            Contains some details around column of data type date like
            min, max, range, year wise and month wise distribution etc.

    Methods:
        get_col_meta(): pd.DataFrame
            Return the metadata in column level
        get_eligible_cols(): None
            Modify the dataframe to pick eligible columns by dropping the
            columns having all null values or constant.
        get_descriptive_stats(): dict
            Returns the self._descriptive_dict
        get_quantile_stat(): dict
            Returns the _quantile_dict
        get_category_details(): dict
            Returns the _obj_cols_dict
        get_date_details(): dict
            Returns the _date_col_details

    """

    # TODO: Add Error class, if a dataframe has no record left after
    #  modification.
    # TODO: Add Error class when all the records are null.

    def __init__(self, df: pd.DataFrame) -> None:
        """Initializer of the class DescriptiveDetails"""
        self._df = None
        self._obj_cols = None
        self._dt_cols = None
        self._df_full = df
        self._col_meta = pd.DataFrame()
        self._num_col_df = None
        self._descriptive_dict = None
        self._quantile_dict = None
        self._num_cols = None
        self._eligible_cols = None
        self._obj_cols_dict = None
        self._date_col_details = None

        self.init_process()

    def init_process(self):
        self.get_col_meta()
        self.get_eligible_cols()
        self.get_col_type()

    def get_col_meta(self) -> pd.DataFrame:
        """Return the self._col_meta"""
        self._col_meta["columns"] = self._df_full.columns.tolist()
        self._col_meta = self._col_meta.set_index('columns')
        self._col_meta["present_datatype"] = self._df_full.dtypes.tolist()
        self._col_meta["non_null_count"] = self._df_full.notna().sum()
        self._col_meta["fill_rate"] = (self._df_full.notna().sum()) * 100 / len(self._df_full)
        self._col_meta["num_unique"] = self._df_full.fillna(0).nunique()
        self._col_meta["unique_rate"] = \
            (self._df_full.fillna(0).nunique()) * 100 / len(self._df_full)

        eligible_cond = (self._col_meta['non_null_count'] != 0) & \
                        (self._col_meta['num_unique'] != 1) & \
                        (self._col_meta['num_unique'] != self._col_meta['non_null_count'])
        self._col_meta['is_eligible'] = False
        self._col_meta.loc[eligible_cond, 'is_eligible'] = True
        self._col_meta = self._col_meta.reset_index()
        return self._col_meta

    def get_eligible_cols(self):
        self._eligible_cols = self._col_meta.loc[self._col_meta['is_eligible'] == True, 'columns'].tolist()
        self._df = self._df_full[self._eligible_cols]

    def get_col_type(self):
        """ Store the values as numeric, categorical or date """
        self._num_cols = self._col_meta.loc[(self._col_meta['is_eligible'] == True) &
                                            ((self._col_meta["present_datatype"] == 'int64')
                                             | (self._col_meta["present_datatype"] == 'float64')),
                                            "columns"].tolist()

        self._obj_cols = self._col_meta.loc[
            (self._col_meta["present_datatype"] == "object") & (self._col_meta["is_eligible"] == True), "columns"
        ].tolist()

        self._dt_cols = self._col_meta.loc[
            self._col_meta["present_datatype"] == "datetime64[ns]", "columns"
        ].tolist()

        if len(self._num_cols) >= 1:
            self._num_col_df = self._df_full[self._num_cols]
        else:
            self._num_col_df = pd.DataFrame()

    def get_df(self):
        return self._df

    def get_descriptive_stat(self) -> dict:
        """Return the self._descriptive_dict"""

        if len(self._num_col_df) > 0:
            minimum = self._num_col_df.min(skipna=True)
            maximum = self._num_col_df.max(skipna=True)
            range_ = (maximum - minimum).to_dict()
            minimum = minimum.to_dict()
            maximum = maximum.to_dict()
            percentile_05 = self._num_col_df.quantile(0.05).to_dict()
            percentile_95 = self._num_col_df.quantile(0.95).to_dict()
            percentile_25 = self._num_col_df.quantile(0.25)
            median = self._num_col_df.quantile(0.5).to_dict()
            mean_ = self._num_col_df.mean().to_dict()
            percentile_75 = self._num_col_df.quantile(0.75)
            iqr = (percentile_75 - percentile_25).to_dict()
            percentile_25 = percentile_25.to_dict()
            percentile_75 = percentile_75.to_dict()
            self._descriptive_dict = {
                "min": minimum,
                "max": maximum,
                "range": range_,
                "5th_percentile": percentile_05,
                "95th_percentile": percentile_95,
                "25th_percentile": percentile_25,
                "75th_percentile": percentile_75,
                "IQR": iqr,
                "median": median,
                "mean": mean_,
            }
            return self._descriptive_dict
        else:
            return {}

    def get_quantile_stat(self) -> dict:
        """Returns the self._quantile_dict"""
        if len(self._num_col_df) > 0:
            std_dev = self._num_col_df.std(numeric_only=True, skipna=True)
            mad = self._num_col_df.mad(skipna=True)
            skewness = self._num_col_df.skew(numeric_only=True, skipna=True)
            sum_ = self._num_col_df.sum(numeric_only=True, skipna=True)
            variance_ = self._num_col_df.var(numeric_only=True, skipna=True)
            #             monotonic = self._num_col_df.is_monotonic
            self._quantile_dict = {
                "Standard Deviation": std_dev,
                "Median absolute deviation": mad,
                "skewness": skewness,
                "sum": sum_,
                "variance": variance_
                #                 "monotonicity": monotonic,
            }
            return self._quantile_dict
        else:
            return {}

    def get_category_details(self) -> dict:
        """Returns the self._obj_cols_dict"""

        obj_df = self._df[self._obj_cols]
        self._obj_cols_dict = {}
        for col_name in obj_df:
            val_counts = obj_df[col_name].value_counts()
            category_distribution = val_counts * 100 / len(obj_df)
            top_5_category = obj_df[col_name].value_counts()[:5]
            memory_size = sys.getsizeof(obj_df[col_name])
            self._obj_cols_dict[col_name] = {
                "value_counts": val_counts,
                "category_distribution": category_distribution,
                "top_5_category": top_5_category,
                "memory_size": memory_size,
            }
        return self._obj_cols_dict

    def get_date_details(self) -> List[Dict[str, Any]]:
        """Return the self._date_col_details"""
        res_ls = []
        for col in self._dt_cols:
            min_date = self._df[col].min()
            max_date = self._df[col].max()
            range_date = max_date - min_date
            date_dist = self._df[col].value_counts()
            top_5_date = self._df[col].value_counts()[:5]
            self._date_col_details = {
                "column": col,
                "min_date": min_date,
                "max_date": max_date,
                "date_range": range_date,
                "date_dist": date_dist,
                "top_5_date": top_5_date,
            }
            res_ls.append(self._date_col_details)
        return res_ls
