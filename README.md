[![PyPI Downloads](https://static.pepy.tech/badge/biomed-multi-view)](https://pepy.tech/projects/biomed-multi-view)
[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/mosheraboh/5dfef582f7add187650c22224a07a52d/raw/clone.json&logo=github)](https://github.com/MShawon/github-clone-count-badge)


# Introduction

This repository contains the implementation of the Multi-view Molecular Embedding with Late Fusion (MMELON) architecture presented in our preprint [Multi-view biomedical foundation models for molecule-target and property prediction](https://arxiv.org/abs/2410.19704). MMELON is a flexible approach to aggregate multiple views (sequence, image, graph) of molecules in a foundation model setting. While models based on single view representation typically performs well on some downstream tasks and not others, the multi-view model performs robustly across a wide range of property prediction tasks encompassing ligand-protein binding, molecular solubility, metabolism and toxicity. It has been applied to screen compounds against a large (> 100 targets) set of G Protein-Coupled receptors (GPCRs) to identify strong binders for 33 targets related to Alzheimer’s disease, which are validated through structure-based modeling and identification of key binding motifs.

Our model integrates:

* Image Representation: Captures the 2D visual depiction of molecular structures, highlighting features like symmetry, bond angles, and functional groups. Molecular images are generated using RDKit and undergo data augmentation during training to enhance robustness.
* Graph Representation: Encodes molecules as undirected graphs where nodes represent atoms and edges represent bonds. Atom-specific properties (e.g., atomic number, chirality) and bond-specific properties (e.g., bond type, stereochemistry) are embedded using categorical embedding techniques.
* Text Representation: Utilizes SMILES strings to represent chemical structures, tokenized with a custom tokenizer. The sequences are embedded using a transformer-based architecture to capture the sequential nature of the chemical information.

The embeddings from these single-view pre-trained encoders are combined using an attention-based aggregator module. This module learns to weight each view appropriately, producing a unified multi-view embedding. This approach leverages the strengths of each representation to improve performance on downstream predictive tasks.

![MultiView diagram](docs/overview.png)

_Figure: Schematic of multi-view architecture. Embeddings from three single view pre-trained encoders (Image, Graph and Text) are combined by the aggregator module into a combined embedding. The network is finetunable for downstream predictive tasks._

Our pre-training dataset comprises 200 million molecules sourced from the PubChem and ZINC22 chemical databases, ensuring diversity and relevance to downstream tasks. Each view encoder is pre-trained with self-supervised tasks tailored to capture the unique features of its representation.

MMELON’s extensible architecture allows for seamless integration of additional views, making it a versatile tool for molecular representation learning. For further details, see [here](https://arxiv.org/abs/2410.19704).

# Getting started with `biomed-multi-view`

## Links

1. [Demo Notebook for inference](notebooks/smmv_api_demo.ipynb)
2. [Contributing code](CONTRIBUTING.md)

## Installation
Follow these steps to set up the `biomed-multi-view` codebase on your system.

### Prerequisites
* Operating System: Linux or macOS
* Python Version: Python 3.11
* Conda: Anaconda or Miniconda installed
* Git: Version control to clone the repository


### Step 1: Set up the project directory
Choose a root directory where you want to install biomed.multi-view. For example:

```bash
export ROOT_DIR=~/biomed-multiview
mkdir -p $ROOT_DIR
```

#### Step 2: Create and activate a Conda environment
```bash
conda create -y python=3.11 --prefix $ROOT_DIR/envs/biomed-multiview
```
Activate the environment:
```bash
conda activate $ROOT_DIR/envs/biomed-multiview
```

#### Step 3: Clone the repository
Navigate to the project directory and clone the repository:
```bash
mkdir -p $ROOT_DIR/code
cd $ROOT_DIR/code

# Clone the repository using HTTPS
git clone https://github.com/BiomedSciAI/biomed-multi-view.git

# Navigate into the cloned repository
cd biomed-multi-view
```
Note: If you prefer using SSH, ensure that your SSH keys are set up with GitHub and use the following command:
```bash
git clone git@github.com:BiomedSciAI/biomed-multi-view.git
```

#### Step 4: Install package dependencies
If you are installing in a Mac, skip this step and proceed to next step. Else, install the package in editable mode along with development dependencies:
``` bash
pip install -e .['dev']
```
Install additional requirements:
``` bash
pip install -r requirements.txt
```

#### Step 5: macOS-Specific instructions (Apple Silicon)
If you are using a Mac with Apple Silicon (M1/M2/M3) and the zsh shell, you may need to disable globbing for the installation command:

``` bash
noglob pip install -e .[dev]
```
Install macOS-specific requirements optimized for Apple’s Metal Performance Shaders (MPS):
```bash
pip install -r requirements-mps.txt
```

#### Step 6:  Installation verification (optional)
Verify that the installation was successful by running unit tests

```bash
python -m unittest bmfm_sm.tests.all_tests
```

#### Summary of commands
Here’s a consolidated list of commands for quick reference:
```bash
# Set up the root directory
export ROOT_DIR=~/biomed-multiview
mkdir -p $ROOT_DIR

# Create and activate the Conda environment
conda create -y python=3.11 --prefix $ROOT_DIR/envs/biomed-multiview
conda activate $ROOT_DIR/envs/biomed-multiview

# Clone the repository
mkdir -p $ROOT_DIR/code
cd $ROOT_DIR/code
git clone https://github.com/BiomedSciAI/biomed-multi-view.git
cd biomed-multi-view

# Install dependencies (non-macOS)
pip install -e .[dev]
pip install -r requirements.txt

# For macOS with Apple Silicon
noglob pip install -e .[dev]
pip install -r requirements-mps.txt

# Verify installation
python -m unittest bmfm_sm.tests.all_tests
```

### Explore the pretrained and finetuned checkpoints from HuggingFace

To explore the pretrained and finetuned models, please refer to the [demo notebook](notebooks/smmv_api_demo.ipynb). This notebook provides detailed examples and explanations on how to use the models effectively. You can launch the notebook from running the following command from a new terminal window.

```bash
cd $ROOT_DIR/code/biomed-multi-view
jupyter lab
```
Copy the URL displayed on the console and paste it on a browser tab. The URL should look something similar to the below
```bash
http://localhost:8888/lab?token=67f8a92c257a82010b4f82b219f7c1ad675ee329e730321e
```
Navigate to the notebooks directory in the sidepanel in the Jupyter Lab GUI to locate the notebook named `smmv_api_demo.ipynb`.

#### Get embeddings from the pretrained model

You can generate embeddings for a given molecule using the pretrained model with the following code. You can excute from cell 1 to 5 in the notebook run the same.

``` python
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel

# Example SMILES string for a molecule
example_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

# Obtain embeddings from the pretrained model
example_emb = SmallMoleculeMultiViewModel.get_embeddings(
    smiles=example_smiles,
    model_path="ibm/biomed.sm.mv-te-84m",
    huggingface=True,
)
print(example_emb)
```

This will output the embedding vector for the given molecule.

#### Inference using finetuned model
You can use the finetuned models to make predictions on new data. You can execute through cell 6 to 8 in the notebook to run the same.

``` python
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.api.dataset_registry import DatasetRegistry

# Initialize the dataset registry
dataset_registry = DatasetRegistry()

# Example SMILES string
example_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

# Get dataset information for BACE (classification task)
bace_ds = dataset_registry.get_dataset_info('BACE')

# Load the finetuned model for BACE
finetuned_model_ds = SmallMoleculeMultiViewModel.from_finetuned(
    bace_ds,
    model_path="ibm/biomed.sm.mv-te-84m-MoleculeNet-ligand_scaffold-BACE-101",
    inference_mode=True,
    huggingface=True
)

# Get predictions
bace_prediction = SmallMoleculeMultiViewModel.get_predictions(
    example_smiles, bace_ds, finetuned_model=finetuned_model_bace
)

print("BACE Prediction:", bace_prediction)
```

##### Output:
```bash
BACE Prediction: {'prediction': tensor(0, dtype=torch.int32)}
```
Note: The outputs are illustrative. Actual predictions may vary depending on the model and data. For more detailed examples and explanations, please refer to the [demo notebook](notebooks/smmv_api_demo.ipynb) cell 8.

## Data preparation
To evaluate and train the models, you need to set up the necessary data, splits, configuration files, and model checkpoints. This section will guide you through the process.

#### Step 1: Set up the `$data_root` directory
First, create a directory to serve as your root directory for all the data. This directory will house all datasets, splits, configuration files, and model checkpoints.
```bash
mkdir -p $ROOT_DIR/data_root
```

#### Step 2: Set the environment variable
Set the `$BMFM_HOME` environment variable to point to your data root directory. This helps the scripts locate your data.
```bash
export BMFM_HOME=$ROOT_DIR/data_root
```
Optionally, add the export command to your shell configuration file (e.g., $HOME/.bashrc for bash):
```bash
echo 'export BMFM_HOME=$ROOT_DIR/data_root' >> $HOME/.bashrc
```

#### Step 3: Download the data splits, configuration files and checkpoints
We provide all the necessary data splits, configuration files, and model checkpoints in a single archive to simplify the setup process.

* Download `data_root_os_v1.tar.gz`: from [this location](https://ad-prod-biomed.s3.us-east-1.amazonaws.com/biomed.multi-view/data_root_os_v1.tar.gz).
* Extract the Archive: 	Uncompress the tar file into your data root directory
  ```bash
  tar -xzvf data_root_os_v1.tar.gz -C $BMFM_HOME
  ```
This will populate `$BMFM_HOME` with the required files and directories.

#### Step 4. Download MoleculeNet datasets
We provide a script to download the MoleculeNet datasets automatically that you can run:
```bash
run-download-moleculenet
```
This script will download the MoleculeNet datasets into `$BMFM_HOME/datasets/raw_data/MoleculeNet/`. **Note**: The `run-download-moleculenet` command launches a Python script that can be executed using `bmfm_sm.python -m bmfm_sm.launch.download_molecule_net_data` from `$ROOT_DIR/code/biomed-multi-view` directory as well.

#### Step 5. Verify the directory structure
After completing the above steps, your `$BMFM_HOME` directory should have the following structure:
```bash
$BMFM_HOME/
├── bmfm_model_dir
│   ├── finetuned
│   │   └── MULTIVIEW_MODEL
│   │       └── MoleculeNet
│   │           └── ligand_scaffold
│   └── pretrained
│       └── MULTIVIEW_MODEL
│           ├── biomed-smmv-base.pth
│           └── biomed-smmv-with-coeff-agg.pth
├── configs_finetune
│   └── MULTIVIEW_MODEL
│       └── MoleculeNet
│           └── ligand_scaffold
│               ├── BACE
│               ├── BBBP
│               ├── CLINTOX
│               ├── ESOL
│               ├── FREESOLV
│               ├── HIV
│               ├── LIPOPHILICITY
│               ├── MUV
│               ├── QM7
│               ├── SIDER
│               ├── TOX21
│               └── TOXCAST
└── datasets
    ├── raw_data
    │   └── MoleculeNet
    │       ├── bace.csv
    │       ├── bbbp.csv
    │       ├── clintox.csv
    │       ├── esol.csv
    │       ├── freesolv.csv
    │       ├── hiv.csv
    │       ├── lipophilicity.csv
    │       ├── muv.csv
    │       ├── qm7.csv
    │       ├── qm9.csv
    │       ├── sider.csv
    │       ├── tox21.csv
    │       └── toxcast.csv
    └── splits
        └── MoleculeNet
            └── ligand_scaffold
                ├── bace_split.json
                ├── bbbp_split.json
                ├── clintox_split.json
                ├── esol_split.json
                ├── freesolv_split.json
                ├── hiv_split.json
                ├── lipophilicity_split.json
                ├── muv_split.json
                ├── qm7_split.json
                ├── sider_split.json
                ├── tox21_split.json
                └── toxcast_split.json
```

After successfully completing the installation and data preparation steps, you are now ready to:

* Usage: Learn how to obtain embeddings from the pretrained model.
* Evaluation: Assess the model’s performance on benchmark datasets using our pretrained checkpoints.
* Training: Understand how to finetune the pretrained model using the provided configuration files.
* Inference: Run the model on sample data in inference mode to make predictions.


### Evaluation
We provide the `run-finetune` command for running finetuning and evaluation processes. You can use this command to evaluate the performance of the finetuned models on various datasets. **Note**: The `run-finetune` command launches a Python script that can be executed using `bmfm_sm.python -m launch.download_molecule_net_data` from `$ROOT_DIR/code/biomed-multi-view` directory as well.

To see the usage options for the script, run:
```bash
run-finetune --help
```

This will display:
``` bash
Usage: run-finetune [OPTIONS]

Options:
  --model TEXT           Model name
  --dataset-group TEXT   Dataset group name
  --split-strategy TEXT  Split strategy name
  --dataset TEXT         Dataset name
  --fit                  Run training (fit)
  --test                 Run testing
  -o, --override TEXT    Override parameters in key=value format (e.g.,
                         trainer.max_epochs=10)
  --help                 Show this message and exit.
```

To evaluate a finetuned checkpoint on a specific dataset, use the --test option along with the --dataset parameter. For example, to evaluate on the BBBP dataset:

``` bash
python run-finetune --test --dataset BBBP
```
If you omit the `--dataset` option, the script will prompt you to choose a dataset:
```bash
python run-finetune --test
Please choose a dataset (FREESOLV, BBBP, CLINTOX, MUV, TOXCAST, QM9, BACE, LIPOPHILICITY, ESOL, HIV, TOX21, SIDER, QM7): BBBP
```
This command will evaluate the finetuned checkpoint corresponding to the BBBP dataset using the test set of the `ligand_scaffold` split.


### Training (finetuning)
To finetune the pretrained model on a specific dataset, use the --fit option:

``` bash
python run-finetune --fit --dataset BBBP
```
Again, if you omit the --dataset option, the script will prompt you to select one:

```bash
python run-finetune --fit
Please choose a dataset (FREESOLV, BBBP, CLINTOX, MUV, TOXCAST, QM9, BACE, LIPOPHILICITY, ESOL, HIV, TOX21, SIDER, QM7): BBBP
```
This command will start the finetuning process for the BBBP dataset using the configuration files provided in the `configs_finetune` directory.

Note: You can override default parameters using the `-o` or `--override` option. For example:

```bash
python run-finetune --fit --dataset BBBP -o trainer.max_epochs=10
```

**Note**: If you run into out of memory errors, you can reduce the batch size using the following syntax

```bash
python run-finetune --fit --dataset BBBP -o data.init_args.batch_size=4
```

# Citations

```
@misc{suryanarayanan2024multiviewbiomedicalfoundationmodels,
      title={Multi-view biomedical foundation models for molecule-target and property prediction},
      author={Parthasarathy Suryanarayanan and Yunguang Qiu and Shreyans Sethi and Diwakar Mahajan and Hongyang Li and Yuxin Yang and Elif Eyigoz and Aldo Guzman Saenz and Daniel E. Platt and Timothy H. Rumbell and Kenney Ng and Sanjoy Dey and Myson Burch and Bum Chul Kwon and Pablo Meyer and Feixiong Cheng and Jianying Hu and Joseph A. Morrone},
      year={2024},
      eprint={2410.19704},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2410.19704},
}
```
