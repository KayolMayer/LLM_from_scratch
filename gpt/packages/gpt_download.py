"""
Created on Tue Mar 18 22:31:42 2025.

@author: kayol
"""

from os import path, makedirs
from requests import get, exceptions
from json import load as load_json
from numpy import squeeze, split
from tensorflow import train
from torch import nn, tensor
from tqdm import tqdm


def load_gpt2(model_dir):
    """
    Load GPT-2 model settings and parameters from a TensorFlow checkpoint.

    This function:
    - Locate the latest TensorFlow checkpoint in the specified model directory.
    - Loads model hyperparameters from `hparams.json`.
    - Extracts model parameters from the checkpoint.

    Parameters
    ----------
        model_dir (str): Path to the directory containing the GPT-2 files.

    Returns
    -------
        tuple: A tuple containing:
            - settings (dict): Model hyperparameter loaded from 'hparams.json'.
            - params (dict): Model parameters extracted from the TensorFlow
                             checkpoint.
    """
    tf_ckpt_path = train.latest_checkpoint(model_dir)
    settings = load_json(open(path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_gpt2(model_size, models_dir):
    """
    Download a pre-trained GPT-2 model from OpenAI's public storage.

    This function downloads the necessary files for a specified GPT-2 model
    size, saves them to the provided directory.

    Parameters
    ----------
        model_size (str): The size of the GPT-2 model to download. Must be one
                          of: "124M", "355M", "774M", "1558M".
        models_dir (str): The directory where the model files should be saved.

    Raises
    ------
        ValueError: If 'model_size' is not in the allowed sizes.
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = path.join(base_url, model_size, filename)
        file_path = path.join(model_dir, filename)
        download_file(file_url, file_path)


def download_file(url, destination):
    """
    Download a file from a given URL and saves it to the specified destination.

    This function:
    - Sends an HTTP GET request to download the file.
    - Verifies if the file already exists and is up-to-date before downloading.
    - Displays a progress bar using `tqdm` while downloading.
    - Handles errors gracefully if the request fails.

    Parameters
    ----------
        url (str): The URL of the file to be downloaded.
        destination (str): The local path where the downloaded file should be
                           saved.

    Returns
    -------
        None: The function prints messages and saves the file but does not
              return anything.

    Raises
    ------
        requests.exceptions.RequestException: If an error occurs during the
                                              download process.
    """
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if path.exists(destination):
            file_size_local = path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        # Extract filename from URL
        progress_bar_description = url.split("/")[-1]
        with tqdm(total=file_size, unit="iB", unit_scale=True,
                  desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    Load GPT-2 model parameters from a TensorFlow checkpoint.

    This function extracts parameters from a TensorFlow checkpoint file
    and organizes them into a nested dictionary structure compatible with
    PyTorch model loading.

    Parameters
    ----------
        ckpt_path (str): Path to the TensorFlow checkpoint directory.
        settings (dict): Dictionary containing model hyperparameters,
                         including number of transformer layers ('n_layer').

    Returns
    -------
        dict: A dictionary where:
              - The top-level key `"blocks"` contains a list of dictionaries,
                each representing a transformer layer.
              - Other parameters are stored under their respective keys,
                following the GPT-2 checkpoint structure.
    """
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = squeeze(train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign_param(left, right):
    """
    Assign values from 'right' to 'left' while ensuring shape compatibility.

    This function checks if 'left' and 'right' tensors have the same shape.
    If they do, it returns a new 'torch.nn.Parameter' initialized with 'right'.
    Otherwise, it raises a `ValueError`.

    Parameters
    ----------
        left (torch.Tensor): The target tensor for assignment.
        right (torch.Tensor): The source tensor whose values will be assigned.

    Returns
    -------
        torch.nn.Parameter: A new parameter containing the values of 'right'.

    Raises
    ------
        ValueError: If 'left' and 'right' have different shapes.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, \
                         Right: {right.shape}")

    return nn.Parameter(tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    Load pre-trained weights into a GPT model from a parameter dictionary.

    This function assigns the pre-trained weight tensors from 'params' to the
    corresponding layers in the 'gpt' model. It maps:
    - Positional and token embeddings
    - Multi-head attention weights and biases
    - Feedforward network weights and biases
    - Layer normalization parameters
    - Output projection weights

    Parameters
    ----------
        gpt (torch.nn.Module): The GPT model instance to load weights into.
        params (dict): A dictionary containing pre-trained model parameters.
                       The expected keys follow the GPT-2 checkpoint format.

    Returns
    -------
        None: The function modifies the 'gpt' model in place.
    """
    # Assign the token embedding weights
    gpt.tok_emb.weight = assign_param(gpt.tok_emb.weight, params['wte'])
    # Assign the position embedding weights
    gpt.pos_emb.weight = assign_param(gpt.pos_emb.weight, params['wpe'])

    for b in range(len(params["blocks"])):
        # Split query, key, and values that are kept in only one matrix
        q_w, k_w, v_w = split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign_param(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign_param(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign_param(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # Split bias of query, key, and values that are kept in only one vector
        q_b, k_b, v_b = split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign_param(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign_param(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign_param(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # Assign the synaptic weights of the projection layer after attention
        gpt.trf_blocks[b].att.out_proj.weight = assign_param(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        # Assign the bias of the projection layer after attention
        gpt.trf_blocks[b].att.out_proj.bias = assign_param(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # Assign the synaptic weights of the expansion layer of the MLP
        gpt.trf_blocks[b].ff.layers[0].weight = assign_param(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        # Assign the bias of the expansion layer of the MLP
        gpt.trf_blocks[b].ff.layers[0].bias = assign_param(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # Assign the synaptic weights of the projection layer of the MLP
        gpt.trf_blocks[b].ff.layers[2].weight = assign_param(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        # Assign the bias of the projection layer of the MLP
        gpt.trf_blocks[b].ff.layers[2].bias = assign_param(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # Assign the first normalization layer parameters of scale (*)
        gpt.trf_blocks[b].norm1.scale = assign_param(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        # Assign the first normalization layer parameters of shift (+)
        gpt.trf_blocks[b].norm1.shift = assign_param(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        # Assign the second normalization layer parameters of scale (*)
        gpt.trf_blocks[b].norm2.scale = assign_param(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        # Assign the second normalization layer parameters of shift (+)
        gpt.trf_blocks[b].norm2.shift = assign_param(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # Assign the last normalization layer of scale (*)
    gpt.final_norm.scale = assign_param(gpt.final_norm.scale, params["g"])
    # Assign the last normalization layer of shift (+)
    gpt.final_norm.shift = assign_param(gpt.final_norm.shift, params["b"])
    # Assign the output head weights. To reduce the number of parameters, it is
    # equal to the token embedding weights at the input of the architecture
    gpt.out_head.weight = assign_param(gpt.out_head.weight, params["wte"])
