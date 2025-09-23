from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class ScaffoldKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, smiles_list, y=None):
        # Step 1: Compute scaffolds
        scaffold_to_indices = defaultdict(list)
        for idx, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles)
            scaffold_to_indices[scaffold].append(idx)

        # Step 2: Shuffle scaffolds (if required)
        scaffold_groups = list(scaffold_to_indices.values())
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(scaffold_groups)

        # Step 3: Split scaffolds into K folds
        fold_indices = [[] for _ in range(self.n_splits)]
        for i, group in enumerate(scaffold_groups):
            fold_indices[i % self.n_splits].extend(group)

        # Step 4: Yield train/validation indices
        for i in range(self.n_splits):
            train_idx = [idx for j, fold in enumerate(fold_indices) if j != i for idx in fold]
            valid_idx = fold_indices[i]
            yield np.array(train_idx), np.array(valid_idx)


def generate_scaffold(smiles) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{smiles} is not a valid SMILES. Could not generate scaffold. Returning None.")
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold
