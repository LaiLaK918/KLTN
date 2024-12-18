import pandas as pd
import numpy as np


def scale_df_with_scalar_multiplication(df: pd.DataFrame, target_min: int, target_max: int):
    """
    Scale each column of a DataFrame by multiplying with a scalar, mapping to an integer range [target_min, target_max].

    Parameters:
        df (pd.DataFrame): Input DataFrame with numeric columns to scale.
        target_min (int): Minimum value of the target integer range.
        target_max (int): Maximum value of the target integer range.

    Returns:
        pd.DataFrame: DataFrame with scaled columns as integers.
    """
    scaled_df = df.copy()  # Avoid modifying the original DataFrame
    scalars = [] # Store the scaling factors for each column
    for col in scaled_df.columns:
        if pd.api.types.is_numeric_dtype(scaled_df[col]):
            col_min = scaled_df[col].min()
            col_max = scaled_df[col].max()

            # Calculate the scaling factor
            scalar = target_max / col_max if col_max != 0 else 1
            scalars.append(scalar)

            # Scale values by multiplication and round to integer
            scaled_df[col] = (scaled_df[col] * scalar).round().astype(int)
    return scaled_df, scalars
