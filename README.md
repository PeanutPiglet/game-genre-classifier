# Header to Game Stats

A Python neural network project that predicts game genres based on header images. This repository contains both a custom dense network implementation and PyTorch models, including a convolutional network.

## Overview

The goal of this project is to build a pipeline for reading game header images, extracting pixel data, and predicting game genres from that visual input. It is motivated as an introductory learning experience. The repository includes:

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

Install PyTorch with instructions from the official website to get the most out of the PyTorch models (e.g. hardware acceleration).

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

`create light 512 256` makes hidden layers: input->512->256->output

`create heavy 1024 512 128` makes hidden layers: input->1024->512->128->output

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

### 5. Test the network

Run evaluation on the test dataset:

```bash
test <model_name>
```

### 6. Save and load models

Save a network to disk:

```bash
save <model_name> <save_file_path>
```

Load a saved model into memory:

```bash
load <save_file_path> <network_type> <model_name>
```

`<network_type>` is either <br>
`dense` — custom dense model
`pytorch` — PyTorch dense model
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

## Results

*COMING SOON*
