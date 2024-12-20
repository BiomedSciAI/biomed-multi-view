# -----------------------------------------------------------------------------
# Parts of this file incorporate code from the following open-source project:
#   Source: https://github.com/HongxinXiang/ImageMol/blob/master/utils/splitter.py
#   License: MIT License
# -----------------------------------------------------------------------------

import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm import log_execution_time


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold


def scaffold_to_smiles(
    mols: list[str] | list[Chem.Mol], use_indices: bool = False
) -> dict[str, set[str] | set[int]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        if Chem.MolFromSmiles(mol) != None:
            scaffold = generate_scaffold(mol)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)

    return scaffolds


# splitting functions


@log_execution_time
def split_train_val_test_idx(
    idx,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    sort=False,
    seed=42,
    shuffle=True,
):
    random.seed(seed)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    total = len(idx)

    train_idx, valid_idx = train_test_split(
        idx, test_size=frac_valid, shuffle=shuffle, random_state=seed
    )
    train_idx, test_idx = train_test_split(
        train_idx,
        test_size=frac_test * total / (total - len(valid_idx)),
        shuffle=shuffle,
        random_state=seed,
    )

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == total

    if sort:
        train_idx = sorted(train_idx)
        valid_idx = sorted(valid_idx)
        test_idx = sorted(test_idx)

    return train_idx, valid_idx, test_idx


def split_train_val_test_idx_stratified(
    idx, y, frac_train=0.8, frac_valid=0.1, frac_test=0.1, sort=False, seed=42
):
    random.seed(seed)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    total = len(idx)

    train_idx, valid_idx, y_train, _ = train_test_split(
        idx, y, test_size=frac_valid, shuffle=True, stratify=y, random_state=seed
    )
    train_idx, test_idx = train_test_split(
        train_idx,
        test_size=frac_test * total / (total - len(valid_idx)),
        shuffle=True,
        stratify=y_train,
        random_state=seed,
    )

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == total

    if sort:
        train_idx = sorted(train_idx)
        valid_idx = sorted(valid_idx)
        test_idx = sorted(test_idx)

    return train_idx, valid_idx, test_idx


def scaffold_split_train_val_test(
    index, smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, sort=False
):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    index = np.array(index)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_index, val_index, test_index = (
        index[train_idx],
        index[valid_idx],
        index[test_idx],
    )

    if sort:
        train_index = sorted(train_index)
        val_index = sorted(val_index)
        test_index = sorted(test_index)

    return train_index, val_index, test_index


def random_scaffold_split_train_val_test(
    index,
    smiles_list,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    sort=False,
    seed=42,
):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    index = np.array(index)

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(index)))
    n_total_test = int(np.floor(frac_test * len(index)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_index, val_index, test_index = (
        index[train_idx],
        index[valid_idx],
        index[test_idx],
    )

    if sort:
        train_index = sorted(train_index)
        val_index = sorted(val_index)
        test_index = sorted(test_index)

    return train_index, val_index, test_index


def scaffold_split_balanced_train_val_test(
    index,
    smiles_list,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    balanced: bool = False,
    seed: int = 42,
):
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    index = np.array(index)

    # Split
    train_size, val_size, test_size = (
        frac_train * len(smiles_list),
        frac_valid * len(smiles_list),
        frac_test * len(smiles_list),
    )
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the smiles
    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)

    if (
        balanced
    ):  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(
            scaffold_to_indices.values(),
            key=lambda index_set: len(index_set),
            reverse=True,
        )

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1
    print(
        f"Total scaffolds = {len(scaffold_to_indices)} | train scaffolds = {train_scaffold_count} | val scaffolds = {val_scaffold_count} | test scaffolds = {test_scaffold_count}"
    )

    train_idx = index[train]
    val_idx = index[val]
    test_idx = index[test]

    return train_idx, val_idx, test_idx


# Used to create Split files that are compatible with the new BaseDataModule; Called by submit_jobs_finetune


def create_splits(source_file, frac_val, frac_test, frac_train=1.0, strategy="random"):
    """
    Splits data into training, validation, and test sets according to the specified strategy.

    Args:
    ----
        source_file (str): Path to the CSV file containing the data.
        frac_val (float): Fraction of data to be used for validation.
        frac_test (float): Fraction of data to be used for testing.
        frac_train (float, optional): Fraction of the available training data to actually use.
                                      Defaults to 1.0, meaning all available training data is used.
                                      This differs from frac_val and frac_test, which determine the
                                      fraction of the total dataset allocated to validation and test sets.
        strategy (str, optional): The strategy used to split the data. Options include:
                                  'random', 'sequential', 'stratified', 'ligand_scaffold',
                                  'ligand_random_scaffold', 'ligand_scaffold_balanced'.
                                  Defaults to 'random'.

    Returns:
    -------
        dict: A dictionary containing the train, validation, and test indices, as well as the source file.

    """
    assert 0 <= frac_train <= 1, "frac_train should be between 0 and 1."
    assert 0 <= frac_val <= 1, "frac_val should be between 0 and 1."
    assert 0 <= frac_test <= 1, "frac_test should be between 0 and 1."
    assert frac_val + frac_test <= 1, "frac_val + frac_test should not exceed 1."

    frac_train_real = frac_train

    frac_train = 1 - (frac_val + frac_test)
    data_csv = pd.read_csv(source_file)
    idxs = list(data_csv[ns.FIELD_INDEX])

    split_strategy = ns.SplitStrategy[strategy.upper()]

    if split_strategy == ns.SplitStrategy.SEQUENTIAL:
        train_idx, val_idx, test_idx = split_train_val_test_idx(
            idxs, frac_train, frac_val, frac_test, shuffle=False
        )
    elif split_strategy == ns.SplitStrategy.RANDOM:
        train_idx, val_idx, test_idx = split_train_val_test_idx(
            idxs, frac_train, frac_val, frac_test, shuffle=True
        )
    elif split_strategy == ns.SplitStrategy.STRATIFIED:
        labels = data_csv[ns.FIELD_LABEL].values.tolist()
        train_idx, val_idx, test_idx = split_train_val_test_idx_stratified(
            idxs, labels, frac_train, frac_val, frac_test
        )
    elif split_strategy == ns.SplitStrategy.LIGAND_SCAFFOLD:
        smiles = data_csv[ns.FIELD_SMILES].values.tolist()
        train_idx, val_idx, test_idx = scaffold_split_train_val_test(
            idxs, smiles, frac_train, frac_val, frac_test
        )
    elif split_strategy == ns.SplitStrategy.LIGAND_RANDOM_SCAFFOLD:
        smiles = data_csv[ns.FIELD_SMILES].values.tolist()
        train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(
            idxs, smiles, frac_train, frac_val, frac_test
        )
    elif split_strategy == ns.SplitStrategy.LIGAND_SCAFFOLD_BALANCED:
        smiles = data_csv[ns.FIELD_SMILES].values.tolist()
        train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(
            idxs, smiles, frac_train, frac_val, frac_test, balanced=True
        )

    if type(train_idx[0] is not int):
        train_idx = [int(idx) for idx in train_idx]
    if type(val_idx[0] is not int):
        val_idx = [int(idx) for idx in val_idx]
    if type(test_idx[0] is not int):
        test_idx = [int(idx) for idx in test_idx]

    if frac_train_real < 1.0:
        train_idx = np.random.choice(
            train_idx, int(len(train_idx) * frac_train_real), replace=False
        ).tolist()

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "source_file": source_file,
    }
    return splits


def calculate_task_weights(train_ids, data_file):
    """
    Returns
    -------
    - weights (np.array): A numpy array of shape (num_tasks, 2), where each row contains the
                          weights for class 0 and class 1 for each task.

    """
    df = pd.read_csv(data_file)
    if "index" not in df.columns:
        raise ValueError("CSV file does not contain 'index' column")
    df_train = df[df["index"].isin(train_ids)]

    labels_train = df_train["label"].apply(
        lambda x: (
            np.array([int(i) for i in str(x).split()])
            if isinstance(x, str)
            else np.array([x])
        )
    )

    labels_train = np.stack(labels_train.values)

    num_tasks = labels_train.shape[1]
    weights = []

    for task_idx in range(num_tasks):
        task_labels = labels_train[:, task_idx]
        valid_task_labels = task_labels[task_labels != -1]

        if len(valid_task_labels) == 0:
            task_weights = [0.5, 0.5]
        else:
            count_labels_train = Counter(valid_task_labels)
            total_valid_labels = len(valid_task_labels)
            imbalance_weight = {
                key: 1 - count_labels_train[key] / total_valid_labels for key in [0, 1]
            }
            task_weights = [imbalance_weight.get(0, 1), imbalance_weight.get(1, 1)]
        weights.append(task_weights)

    weights = np.array(weights, dtype="float")
    return weights


def calculate_label_means(train_ids, data_file):
    """
    Returns
    -------
    - label_means (np.array): A numpy array of shape (num_tasks, 1), where each row contains the
                          label means for each task.

    """
    df = pd.read_csv(data_file)
    if "index" not in df.columns:
        raise ValueError("CSV file does not contain 'index' column")
    df_train = df[df["index"].isin(train_ids)]

    labels_train = df_train["label"].apply(
        lambda x: np.array([float(i) for i in str(x).split()])
        if isinstance(x, str)
        else np.array([x])
    )

    labels_train = np.stack(labels_train.values)

    num_tasks = labels_train.shape[1]
    label_means = []

    for task_idx in range(num_tasks):
        task_labels = labels_train[:, task_idx]
        valid_task_labels = task_labels[task_labels != -1]

        if len(valid_task_labels) == 0:
            means = [0]
        else:
            means = [np.mean(valid_task_labels)]
        label_means.append(means)

    label_means = np.array(label_means, dtype="float")
    return label_means