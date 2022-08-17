"""
This module is for data analysis
"""
from typing import List, Dict, Any, Tuple

from d2ipy.profiling import descriptive_util
import pandas as pd


class Analyzer:
    """
    This class is written for data analysis. This class will have different analysis
    on multivariate front.

    Attributes:
        _df: pd.DataFrame
            The dataframe being used in the analysis.
        _desc_obj: descriptive_util.DescriptiveDetails
            The descriptive analysis class.
        _meta_data: pd.DataFrame
            The metadata dataframe which contains various details like datatype,
            fill rate etc.
        filtered_df: pd.DataFrame
            the dataframe which is contains only eligible columns.
        corr_matrix: dict
            The correlation matrix for  numerical data.
        cov_matrix: pd.DataFrame
            The covariance matrix for numerical data.
        date_category_analyzer: List
            List of dataframes which contains categorical analysis on date.
        date_numeric_analyzer: List
            List of analysis on numeric data after grouping by date
        cat_num_analyzer: List
            List of analysis on numeric columns after grouping by categorical
            columns.
        cat_cat_analyzer: List
            List of analysis on categorical columns after grouping by another
            categorical column.
        self.cat_analysis: df.DataFrame
            Dataframe of categorical analysis
        self.num_analysis: pd.DataFrame
            Dataframe of numerical analysis which contains numeric column as
            categorical by deciling it.

    Methods:
        get_correlation():
        get_covariance():
        date_distribution():
        analyze_category():
        get_category_details():
        check_numeric_analysis():
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """ Class initializer """
        self._df = df
        self._desc_obj = descriptive_util.DescriptiveDetails(df)
        self._meta_data = self._desc_obj.get_col_meta()
        self.filtered_df = self._desc_obj.get_df()
        self.date_cols = None
        self.corr_matrix = None
        self.cov_matrix = None
        self.cat_analyzer = None
        self.date_category_analyzer = None
        self.date_numeric_analyzer = None
        self.cat_num_analyzer = None
        self.cat_cat_analyzer = None
        self.cat_analysis = None

    def get_correlation(self):
        """ Get various kind of correlation values """
        pearson = self.filtered_df.corr(method='pearson')
        spearman = self.filtered_df.corr(method='spearman')
        kendall = self.filtered_df.corr(method='kendall')
        self.corr_matrix = {'pearson': pearson, 'spearman': spearman,
                            'kendall': kendall}

    def get_covariance(self):
        """ Return the covariance in the dataframe """
        self.cov_matrix = self.filtered_df.cov()

    def date_distribution(self) -> Tuple[List[Dict[str, Any]], Any]:
        """ Get the date wise analysis here """
        self.date_cols = self._meta_data.loc[self._meta_data
                                             ['present_datatype'] == "datetime",
                                             "columns"].tolist()
        filtered_copy = self.filtered_df.copy()
        res = []
        for col in self.date_cols:
            filtered_copy.loc[:, col + "_yr"] = filtered_copy[col].dt.year.astype(str)
            filtered_copy.loc[:, col + "_month"] = filtered_copy[col].dt.year.astype(str)
            filtered_copy.loc[:, col + "_date"] = filtered_copy[col].dt.date.astype(str)
            filtered_copy.loc[:, col + "_mon_yr"] = \
                filtered_copy.loc[:, col + "_month"] + "/" + \
                filtered_copy.loc[:, col + "_yr"]
            self.date_cols = self.date_cols + [col + "_yr", col + "_month", col + "_date", col + "_mon_yr"]

            yrs = self.filtered_df[col].dt.year.value_counts()
            months = self.filtered_df[col].dt.month.value_counts()
            days = self.filtered_df[col].dt.month.value_counts()
            mon_yr = (
                self.filtered_df[col].dt.month.astype(str)
                + "-"
                + self.filtered_df[col].dt.year.astype(str)
            )
            mon_yr_dist = mon_yr.value_counts()
            tmp_date_dict = {"yrs": yrs, "months": months,
                             "days": days, "month_yr": mon_yr_dist}
            res.append(tmp_date_dict)
        return res, filtered_copy

    def analyze_category(self, col_name: str) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Get the analysis on categorical columns

        Args:
            col_name: str
                The categorical column, on which basis the analysis will be built
        Returns:
            None
        """
        _, filtered_df = self.date_distribution()
        self.cat_num_analyzer = []
        self.cat_cat_analyzer = []
        self.date_numeric_analyzer = []
        for col in filtered_df:
            if col == col_name:
                continue
            elif filtered_df[col].dtype == 'O' and col not in self.date_cols:
                tmp_analysis = pd.crosstab(filtered_df[col_name], filtered_df[col])
                self.cat_analyzer.append({col_name + " : " + col: tmp_analysis})
            elif filtered_df[col].dtype == 'int64':
                numeric_analysis = filtered_df.groupby(col_name).agg(['sum', 'mean',
                                                                      'median', 'count',
                                                                      'var', 'std'])
                self.cat_num_analyzer.append(numeric_analysis)
            elif col in self.date_cols:
                if filtered_df[col].dtype == 'O':
                    tmp_cat_analysis = pd.crosstab(filtered_df[col_name], filtered_df[col])
                    self.date_category_analyzer.append(tmp_cat_analysis)
                elif filtered_df[col].dtype == 'int':
                    tmp_num_analysis = filtered_df.groupby(col_name).agg(['sum', 'mean',
                                                                          'median', 'count',
                                                                          'var', 'std'])
                    self.date_numeric_analyzer.append(tmp_num_analysis)
        return self.cat_num_analyzer, self.cat_cat_analyzer, self.date_numeric_analyzer

    def get_category_details(self, cols: List[str]) -> None:
        """
        Group by multiple categorical values and show the aggregation values on numeric columns

        Args:
            cols: List[str]
                The column names on which basis the group wise analysis is going to happen
        Returns:
            None
        """
        self.cat_analysis = self.filtered_df.groupby(cols).agg('count')
        return self.cat_analysis

    def decile_numeric_analysis(self, col: str) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Decile the numeric values and get the analysis by treating it as a category

        Args:
             col: str
                The columns which will be used for analysis
        Returns:
             None
        """
        filtered_copy = self.filtered_df.copy()
        filtered_copy.loc[:, col + "_decile"] = pd.cut(filtered_copy.loc[:, col], bins=10, labels=False)
        cat_num_analyzer, cat_cat_analyzer, date_numeric_analyzer = self.analyze_category(col + "_decile")
        return cat_num_analyzer, cat_cat_analyzer, date_numeric_analyzer
