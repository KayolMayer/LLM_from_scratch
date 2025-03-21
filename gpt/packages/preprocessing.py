"""
Created on Thu Mar 20 16:37:44 2025.

@author: kayol
"""

def random_split_df(df, train_frac, validation_frac):
    """
    Randomly split a DataFrame into training, validation, and test sets.

    This function shuffles the input DataFrame and then partitions it into
    training, validation, and test subsets based on the specified fractions.

    Parameters
    ----------
        df (pandas.DataFrame): The input DataFrame to be split.
        train_frac (float): The fraction of data to be allocated to the
                            training set (0 to 1).
        validation_frac (float): The fraction of data to be allocated to the
                                 validation set (0 to 1).

    Returns
    -------
        tuple: A tuple containing three DataFrames:
            - train_df (pandas.DataFrame): The training dataset.
            - validation_df (pandas.DataFrame): The validation dataset.
            - test_df (pandas.DataFrame): The test dataset (remaining data
                                                            after train and
                                                            validation splits).

    Raises
    ------
        ValueError: If 'train_frac + validation_frac' exceeds 1.
    """
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df
