"""
Created on Fri Mar 14 11:35:05 2025.

@author: kayol
"""

from torch.utils.data import Dataset, DataLoader
from torch import tensor, long
from tiktoken import get_encoding
from pandas import read_csv


class gpt_dataset_v1(Dataset):
    """
    A custom PyTorch Dataset for tokenized text data.

    It is designed for training GPT models. This dataset tokenizes the input
    text and splits it into overlapping sequences using a sliding window
    approach.

    Attributes
    ----------
        input_ids (list of torch.Tensor): List of tokenized input sequences.
        target_ids (list of torch.Tensor): List of target sequences for
                                           next-token prediction.

    Parameters
    ----------
        txt (str): The input text to be tokenized and processed.
        tokenizer (Tokenizer): A tokenizer with an `encode` method for
                               converting text into token IDs.
        max_length (int): The maximum sequence length for each chunk.
        stride (int): The step size for the sliding window, controlling overlap
                      between sequences.

    Methods
    -------
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the input and target tensor at the specified
                          index.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences
        # of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(tensor(input_chunk))
            self.target_ids.append(tensor(target_chunk))

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieve the input and target tensors for the given index.

        Parameters
        ----------
            idx (int): The index of the sample to retrieve.

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor]: A pair of tensors (input_ids,
                                                                  target_ids).
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    Create a PyTorch DataLoader using tokenized text data.

    This function tokenizes the given text using a GPT-style tokenizer,
    constructs a dataset using a sliding window approach, and then loads it
    into a DataLoader for efficient batching and training.

    Parameters
    ----------
        txt (str): The input text to be tokenized and processed.
        batch_size (int, optional): The number of batch samples (default: 4).
        max_length (int, optional): The maximum sequence length for each sample
                                    (default: 256).
        stride (int, optional): The step size for the sliding window,
                                controlling overlap (default: 128).
        shuffle (bool, optional): Whether to shuffle the dataset at every epoch
                                 (default: True).
        drop_last (bool, optional): Whether to drop the last batch if it is
                                    incomplete (default: True).
        num_workers (int, optional): Number of worker processes for data
                                     loading (default: 0).

    Returns
    -------
        DataLoader: A PyTorch DataLoader for batching and iterating over
                    tokenized sequences.
    """
    # Initialize the tokenizer
    # "p50k_base" is commonly used for gpt3 models
    # "cl100k_base" is used for GPT-4 and later versions
    tokenizer = get_encoding("gpt2")

    # Create dataset
    dataset = gpt_dataset_v1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


class dataset_spam(Dataset):
    """
    A PyTorch Dataset for spam detection.

    This dataset loads text messages from a CSV file, tokenizes them using
    a provided tokenizer, and applies padding/truncation to ensure all
    sequences have the same length.

    Attributes
    ----------
        data (pandas.DataFrame): The loaded dataset containing text messages.
        encoded_texts (list of list of int): List of tokenized and padded text
                                             messages.
        max_length (int): The maximum length of tokenized sequences.

    Parameters
    ----------
        csv_file (str): Path to the CSV file containing the dataset.
        tokenizer: A tokenizer object with an `encode` method for tokenizing.
        max_length (int, optional): The fixed length for tokenized sequences.
                                    If None, the longest sequence determines
                                    the length. (default: None)
        pad_token_id (int, optional): The token ID used for padding shorter
                                      sequences.
                                      Default 50256 (GPT-2 end-of-text token).

    Methods
    -------
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Retrieves the tokenized sequence and corresponding label at the
            specified index.
    """

    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):

        self.data = read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length-len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """
        Retrieve the tokenized sequence and label at the specified index.

        Parameters
        ----------
            index (int): The index of the sample to retrieve.

        Returns
        -------
            tuple: A tuple containing:
                - torch.Tensor: The tokenized input tensor (max_length,).
                - torch.Tensor: The label tensor (e.g., 0 for ham, 1 for spam).
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            tensor(encoded, dtype=long),
            tensor(label, dtype=long)
        )

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
            int: The number of rows in the dataset.
        """
        return len(self.data)

    def _longest_encoded_length(self):
        """
        Compute and returns the length of the longest tokenized sequence.

        Returns
        -------
            int: The length of the longest encoded text sequence.
        """
        max_length = max([len(enc_text) for enc_text in self.encoded_texts])

        return max_length
