# Header to Game Stats

A Python neural network project that predicts game genres based on header images. This repository contains both a custom dense network implementation and PyTorch models, including a convolutional network.

## Overview

The goal of this project is to build a pipeline for reading game header images, extracting pixel data, and predicting game genres from that visual input. It is motivated as an introductory learning experience. Game developers can also use this to assess perceived game genres based on their artworks. The repository includes:

- `main.py` — interactive CLI interface to create, train, test, save, and load models
- `dense.py` — a custom dense neural network implementation
- `pytorch_model.py` — a PyTorch-based neural network implementation with dense and 2D convolutional variants
- `training.py` — training loop for networks over the dataset
- `testing.py` — evaluation loop for test batches
- `dataloader.py` — image loading, batch discovery, and genre label transformation

## Project Structure

- `data/` — training image batches and metadata
- `test/` — test batches used by `testing.py`
- `scraping/` — scraper tools used to collect raw header images and game details from Steam API
- `prepare.py` — conversion tool to package raw scraped data for training and testing
- `common_types.py` — abstract network interface definitions
- `README.md` — project documentation

## Requirements

This project requires Python 3.11+ and the following packages:

- `numpy`
- `Pillow`
- `torch` (PyTorch)

Install dependencies with:

```bash
pip install numpy pillow torch
```

Install PyTorch with instructions from the [official website](https://pytorch.org/get-started/locally/) to get the most out of the PyTorch models (e.g. hardware acceleration).

## Quick Start

### 1. Run the interactive CLI

Start the main program:

```bash
python main.py
```

### 2. Create a network

Create a custom-built dense model:

```bash
create <model_name>
```

Create a dense PyTorch model:

```bash
create_pytorch <model_name>
```

Create a convolutional PyTorch model:

```bash
create_pytorch_conv <model_name>
```

To specify hidden layer sizes, overload them as arguments after <model_name>.

`create light 512 256` makes a model named "light" with hidden layers: input->512->256->output

`create heavy 1024 512 128` makes a model named "heavy" with hidden layers: input->1024->512->128->output

### 3. Evaluate a single image

See model output for a single image:

```bash
run <model_name> <image_file_path>
```

Note that this command will auto-resize the input image to match the size specified in `dataloader.py`

To inspect the raw model output, append a `debug` argument at the end:<br/>
`run <model_name> <image_file_path> debug`


### 4. Train the network

Train a model for a number of epochs:

```bash
train <model_name> <epochs>
```

For example, `train my_dense_network 10`.

No training data is included in the repository source code. Refer to the [Scraping & Preparation](#scraping--preparation) section to obtain data.

### 5. Test the network

Run evaluation on the test dataset:

```bash
test <model_name>
```

This will evaluate on channels of the output vector which have corresponding genre keys in `test/genres.json`.

To evaluate on all channels (currently there are 100 channels), append the `placeholder` flag:<br/>
`test <model_name> placeholder`

Similar to training data, no testing data is provided in the repository. Refer to the [Scraping & Preparation](#scraping--preparation) section.

### 6. Save and load models

Save a network to disk:

```bash
save <model_name> <save_file_path>
```

Load a saved model into memory:

```bash
load <save_file_path> <network_type> <model_name>
```

`<network_type>` is either <br/>
`dense` — custom dense model <br/>
`pytorch` — PyTorch dense model <br/>
`pytorch-conv` — PyTorch convolution model

### 7. Manage networks

Show networks currently in memory:

```bash
show
```

Show details of a specific network in memory:

```bash
show <model_name>
```

Stop the CLI:

```bash
stop
```

## Notes

- Images are expected to be `128 x 64` pixels with RGB channels. To adjust this, change the global constants in the Python modules.
- `dataloader.py` provides `load_image_2d` for convolutional inputs and `load_image` for flattened inputs.

## Scraping & Preparation

This repository does not include training or test data by default. The dataset is created from Steam metadata and header images using the scripts in `scraping/`, then packaged into model-ready batches by `prepare.py`.

### Data scraping workflow

1. `scraping/scraper_appids.py`
   - Uses the Steam partner API to collect app ID batches. See [Steam's documentation](https://partner.steamgames.com/doc/webapi_overview/auth) on obtaining a publisher key.
   - Reads the publisher key from `../secrets.json` and saves results in `scraping/appids/appids<start_id>.json`.
   - These files provide the app IDs used for fetching Steam app details in the next step.

2. `scraping/scraper_appdetails.py`
   - Reads each app ID batch and requests app details from Steam’s public `appdetails` API.
   - Keeps only entries where the Steam app is a game and extracts `name`, `header_image`, `genres`, `categories`, and `release_date`.
   - Saves each detail batch to `scraping/appdetails/appdetails<first_appid>.json`.

3. `scraping/scraper_headers.py`
   - Downloads header images from `header_image` URLs found in the app details.
   - Stores images under `scraping/headers/headers<first_appid>/header<appid>.jpg`.
   - The script includes a short delay between requests to avoid overloading the Steam servers.

### Data preparation workflow

1. `prepare.py`
   - Verifies that `scraping/appdetails/`, `scraping/headers/`, and an empty `data/` directory exist.
   - Runs `populate_appdetails()` to convert scraped metadata into batches of up to 1000 apps (adjustable in the script).
   - Writes batch metadata as `data/batch<suffix>/appdetails.json`.
   - Builds `data/genres.json` containing genre ID-to-description mappings.

2. Header images
   - `prepare.py` also resizes and copies scraped header images into the prepared batch folders.
   - Images are resized to `128 x 64` using bilinear interpolation to match the model input expectations. This can be adjusted; see important notes below.
   - Header files are placed into the same batch partition as their corresponding app metadata.

### Important notes

- `prepare.py` requires `data/` to be empty before running, and it will refuse to continue if the folder already contains files or directories.
- If you change the image shape in `prepare.py`, update the model input handling in `dataloader.py`, `dense.py`, and `pytorch_model.py` to match.
- The scraping scripts assume a working Steam partner API key is stored in `secrets.json` as
```json
{
    "steam-publisher-key": "<your_key>"
}
```

## Results

*COMING SOON*
