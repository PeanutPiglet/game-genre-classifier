# Game Genre Classifer 🖼️

A Python **neural network** project that predicts **game genres based on header images**. This repository contains both a *custom dense network* implementation and PyTorch models, including a convolutional network.

[✅ Quick Start](#quick-start-)

## Overview 💡

The goal of this project is to build a pipeline for reading game header images, extracting pixel data, and predicting game genres from that visual input. It is motivated as an introductory learning experience. Game developers can also use this to assess perceived game genres based on their artworks. The repository includes:

- `main.py` — **interactive CLI** interface to create, train, test, save, and load models
- `dense.py` — a custom dense neural network implementation
- `pytorch_model.py` — a PyTorch-based neural network implementation with dense and 2D convolutional variants
- `training.py` — training loop for networks over the dataset
- `testing.py` — evaluation loop for test batches
- `dataloader.py` — image loading, batch discovery, and genre label transformation

## Project Structure 🗂️

- `data/` — training image batches and metadata
- `test/` — test batches used by `testing.py`
- `scraping/` — scraper tools used to collect raw header images and game details from Steam API
- `prepare.py` — conversion tool to package raw scraped data for training and testing
- `common_types.py` — abstract network interface definitions
- `README.md` — project documentation

## Requirements 🔒

This project requires Python 3.11+ and the following packages:

- `numpy`
- `Pillow`
- `torch` (PyTorch)

Install dependencies with:

```bash
pip install numpy pillow torch
```

Install PyTorch with instructions from the [official website](https://pytorch.org/get-started/locally/) to get the most out of the PyTorch models (e.g. hardware acceleration).

## Quick Start ✅

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
load <save_file_path> <network_type> <model_name> <hidden_layers>
```

`<network_type>` is either <br/>
`dense` — custom dense model <br/>
`pytorch` — PyTorch dense model <br/>
`pytorch-conv` — PyTorch convolution model

`<hidden_layers>` is the sizes of the hidden layers of the model in order and delimited by space.<br/>
For example, `load <save_file_path> <network_type> <model_name> 1024 512`.

The layer sizes must match the saved model's layer sizes exactly.

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

## Notes 📄

- Images are expected to be `128 x 64` pixels with RGB channels. To adjust this, change the global constants in the Python modules.
- `dataloader.py` provides `load_image_2d` for convolutional inputs and `load_image` for flattened inputs.

## Scraping & Preparation 📁⇄📁

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

### Testing data workflow

Testing data are used for the `test` command in the main program and stored in the `test/` folder.

The folder must contain sub-folders of images (of appropriate size for the model) with genre metadata. Most conveniently, move some of the batches (folders) of images in the prepared `data/` folder into the `test` folder. They share the same structure. 

Ensure a `genres.json` dictionary of genres by their number code exists in `test/`. This can also be copied from the genereted `data/genres.json` after running `prepare.py` on scraped data.

## Results 🤩😅

Simple exploratory tests were done on different models. The purpose for this is mostly a sanity check on model performance and correctness. With consideration for statistical validity, we advise against interpreting the results as representative data.

Nevertheless, we observe significantly faster training times with PyTorch models, given that they use optimized tensors for ML and runs on CUDA cores. All models appear to be stuck at around 90% accuracy. Particularly, models seem to converge well towards their optimums with just one epoch. One outlier is the heavy PyTorch dense network, which had a much lower accuracy than the others.

To examine whether the models were stuck because of over-fitting, we ran additional tests on the highest-epoch saves of each model, marked with an asterisk. Specifically, we sampled from the training data set and hypothesized a higher accuracy if over-fitting is indeed happening. However, no clear differences were observed. Now, a plausible explanation would be that the models are all simply stuck in a local optimum in the optimization landscape. Future training could use different optimizers and hyper-parameters to escape the possible locality. 

Also notice that the false negative (FN) and false positive (FP) rates comparative proportions vary across models. The custom dense models appears to usual have higher false negative rates; during testing, this trend was observed as well, though unrecorded. Whether they are statistically significant, and whether there is a causal relation somewhere, is unknown at this time.

### Training Specs:

The models are trained on 500,000 header images and metadata scraped from the public Steam api, following the procedure in [Scraping & Preparation](#scraping--preparation-). The testing images are 4,000 additional headers scraped similarly.

* Intel Core i7-9700 CPU @ 3.00GHz
* 32GB DDR4 RAM
* 512GB RAID SSD
* NVIDIA GeForce RTX 2060 with 6GB VRAM (for PyTorch CUDA)

### Raw Data
The following table summarizes the results.

| Model                                   | Epoch | Accuracy Mean | Accuracy Std. | FP Rate | FN Rate | Cost Mean | Cost Std. | Training Time |
|-----------------------------------------|-------|----------|---------|---------|---------|-----------|----------|---------------|
| Dense [1024, 512] SGD                   | 0     | 0.6266   | 0.0692  | 0.3209  | 0.0525  | 0.9970    | 0.0018   | 0 s           |
|                                         | 1     | 0.9222   | 0.0419  | 0.0326  | 0.0452  | 0.2033    | 0.0640   | 219 s         |
|                                         | 2     | 0.9256   | 0.0393  | 0.0151  | 0.0593  | 0.2000    | 0.0559   | 441 s         |
|                                         | 5     | 0.9260   | 0.0417  | 0.0198  | 0.0543  | 0.2152    | 0.0470   | 1097 s        |
|                                         | 10    | 0.9260   | 0.0396  | 0.0125  | 0.0616  | 0.2109    | 0.0461   | 2188 s        |
|                                         | 10*   | 0.9296   | 0.0371  | 0.0150  | 0.0554  | 0.2054    | 0.0435   | 2188 s        |
|                                         |       |          |         |         |         |           |          |               |
| Dense [1024, 512] Momentum              | 0     | 0.4775   | 0.0614  | 0.4749  | 0.0476  | 1.0017    | 0.0015   | 0 s           |
|                                         | 1     | 0.9248   | 0.0378  | 0.0086  | 0.0666  | 0.1527    | 0.0770   | 347 s         |
|                                         | 2     | 0.9117   | 0.0403  | 0.0000  | 0.0883  | 0.1788    | 0.0816   | 691 s         |
|                                         | 5     | 0.9248   | 0.0378  | 0.0086  | 0.0666  | 0.1539    | 0.0745   | 1723 s        |
|                                         | 10    | 0.9244   | 0.0418  | 0.0234  | 0.0523  | 0.1965    | 0.0534   | 3427 s        |
|                                         | 10*   | 0.9263   | 0.0405  | 0.0273  | 0.0463  | 0.1942    | 0.0508   | 3427 s        |
|                                         |       |          |         |         |         |           |          |               |
| PyTorch [1024, 512] SGD                 | 0     | 0.5684   | 0.0638  | 0.3697  | 0.0620  | 0.9980    | 0.0025   | 0 s           |
|                                         | 1     | 0.8581   | 0.0371  | 0.1026  | 0.0394  | 0.2839    | 0.0743   | 25 s          |
|                                         | 2     | 0.9176   | 0.0430  | 0.0425  | 0.0399  | 0.1647    | 0.0860   | 50 s          |
|                                         | 5     | 0.9001   | 0.0473  | 0.0664  | 0.0335  | 0.1997    | 0.0946   | 126 s         |
|                                         | 10    | 0.9209   | 0.0409  | 0.0257  | 0.0534  | 0.1583    | 0.0819   | 253 s         |
|                                         | 100   | 0.9072   | 0.0439  | 0.0174  | 0.0754  | 0.1857    | 0.0878   | 2583 s        |
|                                         | 100*  | 0.9082   | 0.0407  | 0.0169  | 0.0749  | 0.1854    | 0.0874   | 2583 s        |
|                                         |       |          |         |         |         |           |          |               |
| PyTorch [1024, 512] Momentum            | 0     | 0.4924   | 0.0579  | 0.4700  | 0.0376  | 0.9983    | 0.0024   | 0 s           |
|                                         | 1     | 0.9047   | 0.0370  | 0.0489  | 0.0464  | 0.1906    | 0.0740   | 25 s          |
|                                         | 2     | 0.8942   | 0.0405  | 0.0845  | 0.0213  | 0.2116    | 0.0810   | 50 s          |
|                                         | 5     | 0.8987   | 0.0421  | 0.0671  | 0.0342  | 0.2026    | 0.0842   | 123 s         |
|                                         | 10    | 0.9051   | 0.0440  | 0.0336  | 0.0613  | 0.1899    | 0.0879   | 247 s         |
|                                         | 100   | 0.8870   | 0.0455  | 0.0578  | 0.0552  | 0.2260    | 0.0910   | 2473 s        |
|                                         | 100*  | 0.8891   | 0.0446  | 0.0563  | 0.0546  | 0.2211    | 0.0895   | 2473 s        |
|                                         |       |          |         |         |         |           |          |               |
| Conv [1024, 512] SGD                    | 0     | 0.5392   | 0.0408  | 0.4091  | 0.0517  | 0.9988    | 0.0013   | 0 s           |
|                                         | 1     | 0.9221   | 0.0401  | 0.0251  | 0.0528  | 0.1557    | 0.0803   | 27 s          |
|                                         | 2     | 0.9051   | 0.0440  | 0.0336  | 0.0613  | 0.1899    | 0.0879   | 55 s          |
|                                         | 5     | 0.9164   | 0.0417  | 0.0431  | 0.0405  | 0.1673    | 0.0834   | 136 s         |
|                                         | 10    | 0.8806   | 0.0448  | 0.0913  | 0.0281  | 0.2387    | 0.0897   | 273 s         |
|                                         | 100   | 0.9006   | 0.0410  | 0.0510  | 0.0484  | 0.1989    | 0.0821   | 2743 s        |
|                                         | 100*  | 0.9013   | 0.0403  | 0.0555  | 0.0432  | 0.1974    | 0.0805   | 2743 s        |
|                                         |       |          |         |         |         |           |          |               |
| Conv [1024, 512] Momentum               | 0     | 0.6236   | 0.0412  | 0.3392  | 0.0372  | 0.9958    | 0.0010   | 0 s           |
|                                         | 1     | 0.9117   | 0.0403  | 0.0000  | 0.0883  | 0.1767    | 0.0806   | 28 s          |
|                                         | 2     | 0.9221   | 0.0401  | 0.0251  | 0.0528  | 0.1557    | 0.0803   | 54 s          |
|                                         | 5     | 0.9221   | 0.0401  | 0.0251  | 0.0528  | 0.1557    | 0.0803   | 138 s         |
|                                         | 10    | 0.9209   | 0.0409  | 0.0257  | 0.0534  | 0.1583    | 0.0819   | 281 s         |
|                                         | 100   | 0.9209   | 0.0409  | 0.0257  | 0.0534  | 0.1583    | 0.0819   | 2721 s        |
|                                         | 100*  | 0.9240   | 0.0407  | 0.0290  | 0.0470  | 0.1520    | 0.0815   | 2721 s        |
|                                         |       |          |         |         |         |           |          |               |
| PyTorch [8192, 2048, 512] Momentum      | 0     | 0.5012   | 0.0567  | 0.4542  | 0.0447  | 1.0021    | 0.0012   | 0 s           |
|                                         | 1     | 0.8856   | 0.0401  | 0.0585  | 0.0559  | 0.2288    | 0.0802   | 63 s          |
|                                         | 2     | 0.8856   | 0.0401  | 0.0585  | 0.0559  | 0.2288    | 0.0802   | 128 s         |
|                                         | 5     | 0.8540   | 0.0409  | 0.0894  | 0.0566  | 0.2920    | 0.0817   | 317 s         |
|                                         | 10    | 0.8895   | 0.0380  | 0.0414  | 0.0691  | 0.2210    | 0.0760   | 637 s         |
|                                         | 100   | 0.8396   | 0.0395  | 0.1118  | 0.0486  | 0.3208    | 0.0790   | 6316 s        |
|                                         | 100*  | 0.8451   | 0.0305  | 0.1099  | 0.0450  | 0.3202    | 0.0791   | 6316 s        |
|                                         |       |          |         |         |         |           |          |               |
| Conv [8192, 2048, 512] Momentum         | 0     | 0.4869   | 0.0405  | 0.4551  | 0.0580  | 1.0008    | 0.0012   | 0 s           |
|                                         | 1     | 0.8831   | 0.0375  | 0.0749  | 0.0420  | 0.2338    | 0.0750   | 32 s          |
|                                         | 2     | 0.9007   | 0.0410  | 0.0661  | 0.0332  | 0.1985    | 0.0819   | 62 s          |
|                                         | 5     | 0.8871   | 0.0399  | 0.0578  | 0.0552  | 0.2259    | 0.0797   | 157 s         |
|                                         | 10    | 0.8409   | 0.0367  | 0.1263  | 0.0328  | 0.3183    | 0.0734   | 316 s         |
|                                         | 100   | 0.9006   | 0.0410  | 0.0510  | 0.0484  | 0.1989    | 0.0821   | 3159 s        |
|                                         | 100*  | 0.9013   | 0.0403  | 0.0555  | 0.0432  | 0.1974    | 0.0805   | 3159 s        |

*testing on training set sample

