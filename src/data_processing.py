import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold
import os

# Atom feature definitions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_aromatic': [0, 1]
}

def one_hot_encoding(x, allowable_set):
    """One-hot encodes a value based on a list of allowable values."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    """Generates a feature vector for a given atom."""
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    features += one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs'])
    features += one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
    features += one_hot_encoding(int(atom.GetIsAromatic()), ATOM_FEATURES['is_aromatic'])
    return np.array(features, dtype=np.float32)

def mol_to_graph_data(mol, target, task_type, device):
    """Converts an RDKit molecule to a PyTorch Geometric Data object."""
    if mol is None:
        return None

    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x = np.array(atom_features_list, dtype=np.float32)

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        # Handle single atom case with a self-loop for edge_index
        edge_index = np.array([[0], [0]], dtype=np.int64)
    else:
        edge_index = np.array(edge_index, dtype=np.int64).T

    if task_type == 'classification':
        y = torch.LongTensor([int(target)]).to(device)
    else: # regression
        y = torch.FloatTensor([float(target)]).to(device)

    return Data(
        x=torch.tensor(x, dtype=torch.float).to(device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).to(device),
        y=y
    )

def scaffold_split(data, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """Performs a scaffold split on the dataset."""
    scaffolds = {}
    for smiles in data['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.setdefault(scaffold, []).append(smiles)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_data, val_data, test_data = [], [], []
    train_cutoff = frac_train * len(data)
    val_cutoff = train_cutoff + frac_valid * len(data)

    for scaffold_set in scaffold_sets:
        if len(train_data) + len(scaffold_set) <= train_cutoff:
            train_data.extend(scaffold_set)
        elif len(val_data) + len(scaffold_set) <= val_cutoff:
            val_data.extend(scaffold_set)
        else:
            test_data.extend(scaffold_set)

    train_df = data[data['smiles'].isin(train_data)]
    val_df = data[data['smiles'].isin(val_data)]
    test_df = data[data['smiles'].isin(test_data)]
    
    return train_df, val_df, test_df


def get_dataset(dataset_name, split_type, device):
    """Loads, splits, and processes a dataset for GNN training."""
    data_path = f'data/{dataset_name}.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df = pd.read_csv(data_path)
    # Standardize column names
    if 'smiles' not in df.columns or 'value' not in df.columns:
        # Attempt to find likely columns, similar to notebooks
        smiles_col = next((c for c in df.columns if 'smiles' in c.lower()), df.columns[0])
        value_col = next((c for c in df.columns if 'measured' in c.lower() or 'value' in c.lower()), df.columns[1])
        df = df.rename(columns={smiles_col: 'smiles', value_col: 'value'})
    
    df = df[['smiles', 'value']].dropna()

    # Determine task type
    task_type = 'regression'
    if dataset_name in ['BACE', 'BBBP']:
        task_type = 'classification'

    # Perform the split
    if split_type == 'random':
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    elif split_type == 'stratified':
        if task_type == 'classification':
            train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['value'])
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['value'])
        else:
            raise ValueError("Stratified split is only supported for classification tasks.")
    elif split_type == 'scaffold':
        train_df, val_df, test_df = scaffold_split(df)
    else:
        raise ValueError(f"Unsupported split type: {split_type}")

    # Convert to PyG Data objects
    def create_data_list(dataframe):
        data_list = []
        for _, row in dataframe.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            data_obj = mol_to_graph_data(mol, row['value'], task_type, device)
            if data_obj:
                data_list.append(data_obj)
        return data_list

    train_dataset = create_data_list(train_df)
    val_dataset = create_data_list(val_df)
    test_dataset = create_data_list(test_df)
    
    print(f"Dataset: {dataset_name}, Split: {split_type}, Task: {task_type}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, task_type
