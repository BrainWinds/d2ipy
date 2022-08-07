from d2ipy.data_source_connection import read_flat_file
from d2ipy.model.d2i_model import ModelD2I


class Profiling:
    def __init__(self):
        pass

    def read_csv(self, csv_path):
        csv_reader = read_flat_file.ReadCSV(csv_path)
        df_shape = csv_reader.get_df_size()
        _df = csv_reader.get_df()
        dq_obj = ModelD2I(_df)
        return dq_obj

    def read_excel(self, excel_path):
        excel_reader = read_flat_file.ReadExcel(excel_path)
        excel_file_obj = excel_reader.read_excel_file()
        sheet_names = excel_reader.get_sheet_names()
        res_dict = {}
        for sheet in sheet_names:
            _df = excel_reader.read_sheet(sheet)
            res_dict[sheet] = ModelD2I(_df)
        return res_dict
