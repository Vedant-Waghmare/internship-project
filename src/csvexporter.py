import pandas as pd
import os

class CSVExporter:

    def __init__(self, filepath):
        self.filepath = filepath
        self.create_folder()

    def create_folder(self):
        folder = os.path.dirname(self.filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def save(self, data_list, unique_column=None):
        df_new = pd.DataFrame(data_list)

        if os.path.exists(self.filepath):
            try:
                df_old = pd.read_csv(self.filepath)
            except Exception:
                df_old = pd.DataFrame()

            df_all = pd.concat([df_old, df_new], ignore_index=True)
            if unique_column and unique_column in df_all.columns:
                df_all.drop_duplicates(subset=[unique_column], inplace=True)
            df_all.to_csv(self.filepath, index=False)
        else:
            df_new.to_csv(self.filepath, index=False)


