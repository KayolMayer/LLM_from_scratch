"""
Created on Thu Mar 20 15:58:40 2025.

@author: kayol
"""

from os import sep, getcwd
from pandas import read_csv, concat
from packages.preprocessing import random_split_df


def create_balanced_dataset_spams(df):
    """
    Create a balanced dataset by undersampling the 'ham' class.

    This function ensures that the dataset has an equal number of 'spam' and
    'ham' messages by randomly sampling 'ham' instances to match the count of
    'spam' instances.

    Parameters
    ----------
        df (pandas.DataFrame): A DataFrame containing a column labeled "Label"
                               with values 'spam' and 'ham'.

    Returns
    -------
        pandas.DataFrame: A new DataFrame with an equal number of 'spam' and
                         'ham' messages.
    """
    # Count the instances of 'spam'
    num_spam = df[df['Label'] == 'spam'].shape[0]

    # Randomly sample 'ham' instances to match the number of 'spam' instances
    ham_subset = df[df['Label'] == 'ham'].sample(num_spam, random_state=123)

    # Combine ham \"subset\" with \"spam\"\n",
    balanced_df = concat([ham_subset, df[df['Label'] == 'spam']])

    return balanced_df


# Path and file name to the dataset
data_path = data_file_path = getcwd() + sep + 'datasets' + sep
data_file_path = data_path + 'SMSSpamCollection.tsv'

# dataframe with the dataset
df = read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])

# Count the number of spams and not spams (ham)
print(df['Label'].value_counts())

# Undersample the smaller class to balace the dataset. It is preferred here
# to simplify the training with a lower number of samples (education purpose).
balanced_df = create_balanced_dataset_spams(df)

# Count the number of spams and not spams (ham)
print(balanced_df["Label"].value_counts())

# Similar to the process of converting text to token ids, we will replace
# 'ham' by 0 and 'spam' by 1
balanced_df['Label'] = balanced_df['Label'].map({'ham': 0, 'spam': 1})

# Split dataset into training (70%), validation (10%), and test (rest 20%)
train_df, validation_df, test_df = random_split_df(balanced_df, 0.7, 0.1)

# Save files to csv to reuse later
train_df.to_csv(data_path + 'train_spam.csv', index=None)
validation_df.to_csv(data_path + 'validation_spam.csv', index=None)
test_df.to_csv(data_path + 'test_spam.csv', index=None)
