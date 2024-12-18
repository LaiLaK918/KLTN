import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def merge_and_stat_label(folder_path, to_exclude: list[str] = [], to_include: list[str] = []):
    all_dfs = []
    file_paths = os.listdir(folder_path)
    if to_exclude:
        file_paths = [file for file in file_paths if any(
            att in file for att in to_exclude)]
    if to_include:
        file_paths = [file for file in file_paths if any(
            att in file for att in to_include)]
    for filename in tqdm(file_paths):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)
    merged_df = pd.concat(all_dfs, ignore_index=True)
    label_stats = merged_df['Label'].value_counts()
    return merged_df, label_stats


def filter_by_min_count(df: pd.DataFrame, label_column: str, min_count: int):
    label_counts = df[label_column].value_counts()
    valid_labels = label_counts[label_counts >= min_count].index
    return df[df[label_column].isin(valid_labels)]


def scale_df_with_MinMaxScaler(df: pd.DataFrame, scalers: list[MinMaxScaler]) -> pd.DataFrame:
    scaled_df = df.copy()
    for i, col in enumerate(df.columns):
        scaled_df[col] = scalers[i].fit_transform(
            df[col].values.reshape(-1, 1))
    return scaled_df
