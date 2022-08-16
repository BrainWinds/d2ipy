"""
This module applies various features to the dataframe object
"""
import pandas as pd
from typing import List, Dict, Any, Tuple

from d2ipy.profiling import descriptive_util
from d2ipy.profiling.descriptive_util import DescriptiveDetails
from d2ipy.analysis.analysis_util import Analyzer


class ProfileModel:
    """
    This is the mid-level class which communicates between business logic
    and analysis. This class ensemble the required class and pass the object of
    various classes like MetaData, DescriptiveDetails to the analysis with the
    attributes and methods.

    Attributes:
    -----------
        _df: pd.DataFrame
            The  dataframe which will be used in the interim classes
        _meta_data_obj: MetaData
            The MetaData object from profiling.descriptive_util
        meta_data: dict
            The metadata of the dataframe
        _desc_obj: DescriptiveDetails
            The DescriptiveDetails class
        _desc_stat: dict
            The descriptive statistics dictionary
        _quantile_stat: dict
            The quantile stat dictionary
        _cat_details: dict
            The category column details
        _date_details: dict
            The date column details

    Methods:
    --------
        get_pandas_df(): pd.DataFrame
            returns the dataframe objects
        get_column_metadata(): dict
            returns the metadata dictionary
        get_descriptive_stat(): dict
            returns the descriptive dictionary
        get_quantile_stat(): dict
            returns the quantile stats dictionary
        get_category_details(): dict
            returns the categorical column details
        get_date_details(): dict
            returns date column details
    """
    _desc_obj: DescriptiveDetails

    def __init__(self, df):
        """
        The initializer for daqua details. It takes only one argument
        i,e the dataframe.

        Args:
            df: pd.DataFrame
                The dataframe which will be used in the class
        Returns:
            None
        """
        self._df = df
        self._meta_data_obj = descriptive_util.MetaData(self._df)
        self._desc_obj = descriptive_util.DescriptiveDetails(self._df)

        self.meta_data = None
        self._col_meta = None
        self._desc_stat = None
        self._quantile_stat = None
        self._cat_details = None
        self._date_details = None

        self.init_d2ipy()

    def init_d2ipy(self):
        self._desc_obj.get_eligible_cols()
        self.set_descriptive_stat()
        self.set_quantile_stat()
        self.set_category_details()
        self.set_date_details()

    @property
    def get_pandas_df(self):
        """ Return Pandas DataFrame version of the object """
        return self._df

    @property
    def get_metadata(self):
        """ Return the metadata of the dataframe """
        self.meta_data = self._meta_data_obj.get_meta()
        return self.meta_data

    @property
    def get_df_info(self):
        """ Return the column level metadata """
        self._col_meta = self._desc_obj.get_col_meta
        return self._col_meta

    def set_descriptive_stat(self):
        """ set the value  of _desc_stat """
        self._desc_stat = self._desc_obj.get_descriptive_stat()

    def set_quantile_stat(self):
        """ set the value of _quantile_stat """
        self._quantile_stat = self._desc_obj.get_quantile_stat()

    def set_category_details(self):
        self._cat_details = self._desc_obj.get_category_details()

    def set_date_details(self):
        self._date_details = self._desc_obj.get_date_details()

    def describe(self, column_type: str) -> dict:
        """ Initialize all the descriptive values and set the attributes """
        self.init_d2ipy()
        res = {}
        if column_type == 'all':
            res['quantile'] = self._quantile_stat
            res['descriptive'] = self._desc_stat
            res['categorical_column'] = self._cat_details
            res['date_column'] = self._date_details

        elif column_type == 'numeric':
            res['quantile'] = self._quantile_stat
            res['descriptive'] = self._desc_stat

        elif column_type == 'categorical':
            res['categorical_column'] = self._cat_details

        elif column_type == 'date':
            res['date_column'] = self._date_details
        else:
            raise

        return res


class AnalysisModel:
    def __init__(self, df):
        self._df = df
        self.filtered_df = None
        self._analyzer_obj = None
        self._corr_matrix = None
        self._cov_matrix = None
        self.date_df = None
        self.date_dist = None

        self.init_analyzer()

    def init_analyzer(self) -> None:
        """ Initialize the Analyzer class """
        self._analyzer_obj = Analyzer(self._df)
        self.filtered_df = self._analyzer_obj.filtered_df

    def set_correlation(self) -> None:
        """ Set the correlation values """
        self._corr_matrix = self._analyzer_obj.get_correlation()

    def get_correlation(self) -> dict:
        """ Return the correlation matrix """
        self.set_correlation()
        return self._corr_matrix

    def set_covariance(self) -> None:
        """ Set the covariance matrix """
        self._cov_matrix = self._analyzer_obj.get_covariance()

    def get_covariance(self) -> pd.DataFrame:
        """ Return the covariance matrix """
        self.set_covariance()
        return self._cov_matrix

    def set_date_distribution(self) -> None:
        """ Set the date distribution of the dataframe """
        self.date_dist, self.date_df = self._analyzer_obj.date_distribution()

    def get_date_distribution(self) -> List[Dict[str, Any]:
        """Get the date distribution"""
        self.set_date_distribution()
        return self.date_dist


