import requests
import random
import numpy as np
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch
from torch_geometric.data import Data

allowed_bond_types    = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
allowed_elements      = ['C', 'N', 'O', 'H', 'Cl', 'P', 'S', 'As', 'Br', 'Ca', 'Se', 'I', 'K', 'Mg', 'Na', 'Ni', 'W', 'F', 'B', 'Hg', 'Al', 'Li', 'Zn', 'Si', 'Co', 'Pb', 'Sn', 'Cu', 'Ba', 'Fe', 'Mn', 'Cr']
allowed_hybridization = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S', 'UNSPECIFIED']
allowed_chirality     = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']

allowed_bond_indexed         = {token: idx for idx, token in enumerate(allowed_bond_types)}
allowed_elements_indexed     = {token: idx for idx, token in enumerate(allowed_elements)}
allowed_hybridization_indexed = {token: idx for idx, token in enumerate(allowed_hybridization)}
allowed_chirality_indexed    = {token: idx for idx, token in enumerate(allowed_chirality)}

def extract_atoms_and_bonds(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    mol = Chem.AddHs(mol)
    atom_list = []
    hybrid_list = []
    aromatic_list = []
    charge_list = []
    chirality_list = []

    for atom in mol.GetAtoms():
        atom_list.append(atom.GetSymbol())
        hybrid_list.append(str(atom.GetHybridization()))
        aromatic_list.append(int(atom.GetIsAromatic()))
        charge_list.append(atom.GetFormalCharge())
        chirality_list.append(str(atom.GetChiralTag()))

    bond_list = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())
        is_conjugated = int(bond.GetIsConjugated())
        bond_list.append((begin_idx, end_idx, bond_type, is_conjugated))

    return atom_list, hybrid_list, aromatic_list, charge_list, chirality_list, bond_list

def one_hot_encode_np(allowed_tokens, input_string):
    one_hot = np.zeros(len(allowed_tokens) + 1, dtype=np.int32)
    if input_string in allowed_tokens:
        one_hot[allowed_tokens[input_string]] = 1
    else:
        one_hot[-1] = 1
    return one_hot

def process_molecule(row):
    name, formula, smiles, logp = row
    try:
        atoms, hybridization, aromatic, charge, chirality, bonds = extract_atoms_and_bonds(smiles)
        if len(atoms) < 3 or len(bonds) < 2:
            return None
    except:
        return None
        
    try:
        logp_flt = float(logp)
    except:
        return None

    encoded_elements  = np.stack([one_hot_encode_np(allowed_elements_indexed, a) for a in atoms])
    encoded_hybrid    = np.stack([one_hot_encode_np(allowed_hybridization_indexed, h) for h in hybridization])
    encoded_chirality = np.stack([one_hot_encode_np(allowed_chirality_indexed, c) for c in chirality])
    encoded_aromatic  = np.array([[a] for a in aromatic], dtype=np.int32)
    encoded_charge    = np.array([[c] for c in charge], dtype=np.int32)

    encoded_atoms = np.concatenate([
        encoded_elements,
        encoded_hybrid,
        encoded_chirality,
        encoded_aromatic,
        encoded_charge,
    ], axis=1)

    indexing = np.array([(bi, bj) for (bi, bj, _t, _c) in bonds], dtype=np.int32).T
    encoded_bonds = np.stack([
        np.concatenate([
            one_hot_encode_np(allowed_bond_indexed, b[2]),
            np.array([b[3]], dtype=np.int32)
        ], axis=0)
        for b in bonds
    ])

    return {
        'atoms':    encoded_atoms,
        'indexing': indexing,
        'bonds':    encoded_bonds,
        'logp':     logp_flt,
        'name':     name,
        'formula':  formula
    }

def convert_to_pyg(V_features, E_indexing, E_features, Y_labels):
    dataset = []
    for v, ei, ef, y in zip(V_features, E_indexing, E_features, Y_labels):
        data = Data(
            x          = torch.tensor(v, dtype=torch.float32),
            edge_index = torch.tensor(ei, dtype=torch.long),
            edge_attr  = torch.tensor(ef, dtype=torch.float32),
            y          = torch.tensor(y, dtype=torch.float32),
        )
        dataset.append(data)
    return dataset

def process_all_data(num_molecules=10000):
    input_data="https://raw.githubusercontent.com/dgront/chem-ml/refs/heads/main/INPUTS/xlogp_JChemEdu/xlogp.tsv"
    req = requests.get(input_data)
    table = []
    for row in req.text.splitlines():
        tokens = row.split("\t")
        if len(tokens) == 4:
            if len(tokens[2]) == 0: continue
            table.append(tokens)
            
    if num_molecules is None:
        rows = table[1:]
    else:
        rows = table[1:num_molecules+1] # Skip header
    print(f"Processing {len(rows)} molecules...")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_molecule, rows),
            total=len(rows),
            desc="Processing molecules"
        ))

    V_features = []
    E_indexing = []
    E_features = []
    Y_labels   = []

    for r in results:
        if r is None:
            continue
        V_features.append(r['atoms'])
        E_indexing.append(r['indexing'])
        E_features.append(r['bonds'])
        Y_labels.append([r['logp']])

    data_len = len(V_features)
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    train_split = int(train_ratio * data_len)
    val_split = int((train_ratio + val_ratio) * data_len)

    indices = list(range(data_len))
    random.shuffle(indices)

    V_train = [V_features[i] for i in indices[:train_split]]
    V_val = [V_features[i] for i in indices[train_split:val_split]]
    V_test = [V_features[i] for i in indices[val_split:]]

    E_i_train = [E_indexing[i] for i in indices[:train_split]]
    E_i_val = [E_indexing[i] for i in indices[train_split:val_split]]
    E_i_test = [E_indexing[i] for i in indices[val_split:]]

    E_f_train = [E_features[i] for i in indices[:train_split]]
    E_f_val = [E_features[i] for i in indices[train_split:val_split]]
    E_f_test = [E_features[i] for i in indices[val_split:]]

    Y_train = [Y_labels[i] for i in indices[:train_split]]
    Y_val = [Y_labels[i] for i in indices[train_split:val_split]]
    Y_test = [Y_labels[i] for i in indices[val_split:]]

    train_dataset = convert_to_pyg(V_train, E_i_train, E_f_train, Y_train)
    val_dataset   = convert_to_pyg(V_val,   E_i_val,   E_f_val,   Y_val)
    test_dataset  = convert_to_pyg(V_test,  E_i_test,  E_f_test,  Y_test)
    
    return train_dataset, val_dataset, test_dataset
