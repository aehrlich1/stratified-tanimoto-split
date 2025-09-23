import csv

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_smiles


class PolarisDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        task: str,
        train=True,
        force_reload=True,
    ):
        self.train = train
        self.force_reload = force_reload

        self.target_col = self._admet_target_to_col_mapping(task)
        self.fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=1024)

        super().__init__(root, force_reload=force_reload)
        self.load(self.processed_paths[0] if train else self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ["train_polaris.csv", "test_polaris.csv"]

    @property
    def processed_file_names(self):
        return ["train_polaris.pt", "test_polaris.pt"]

    def process(self):
        self.process_train() if self.train else self.process_test()

    def process_train(self):
        data_list: list[Data] = []
        with open(self.raw_paths[0], "r") as file:
            lines = csv.reader(file)
            next(lines)  # skip header

            for line in lines:
                smiles = line[0]
                label = line[self.target_col]
                if len(label) == 0:
                    continue

                y = torch.tensor(float(label), dtype=torch.float).view(-1, 1)

                # Log Transform data
                y = torch.log10(y)
                if y.isinf():
                    print("Log transform issues.")
                    y = torch.zeros_like(y)

                data = from_smiles(smiles)
                data.y = y

                data.ecfp = self._generate_ecfp(smiles)

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def process_test(self):
        data_list: list[Data] = []
        with open(self.raw_paths[1], "r") as file:
            lines = csv.reader(file)
            next(lines)  # skip header

            for line in lines:
                smiles = line[0]
                label = line[self.target_col]
                if len(label) == 0:
                    continue

                y = torch.tensor(float(label), dtype=torch.float).view(-1, 1)

                data = from_smiles(smiles)
                data.y = y
                data.ecfp = self._generate_ecfp(smiles)
                data_list.append(data)

        self.save(data_list, self.processed_paths[1])

    def _generate_ecfp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        ecfp = self.fpgen.GetFingerprint(mol)

        return torch.tensor(ecfp, dtype=torch.float32)

    @staticmethod
    def _admet_target_to_col_mapping(target_task: str) -> int:
        match target_task:
            case "MLM":
                return 1
            case "HLM":
                return 2
            case "KSOL":
                return 3
            case "LogD":
                return 4
            case "MDR1-MDCKII":
                return 5
            case _:
                raise ValueError(f"Unknown target task: {target_task}")


class ZincDataset(InMemoryDataset):
    def __init__(self, root, force_reload=False):
        self.fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=1024)
        super().__init__(root, force_reload=force_reload)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["zinc.csv"]

    @property
    def processed_file_names(self):
        return ["zinc.pt"]

    def process(self):
        data_list: list[Data] = []
        with open(self.raw_paths[0], "r") as file:
            lines = csv.reader(file)
            next(lines)  # skip header

            for line in lines:
                smiles = line[0]
                label = line[1]
                if len(label) == 0:
                    continue

                y = torch.tensor(float(label), dtype=torch.float).view(-1, 1)

                data = from_smiles(smiles)
                data.y = y

                data.ecfp = self._generate_ecfp(smiles)

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def _generate_ecfp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        ecfp = self.fpgen.GetFingerprint(mol)

        return torch.tensor(ecfp, dtype=torch.float32)


if __name__ == "__main__":
    dataset = ZincDataset(root="dataset/zinc", force_reload=False)
    print(dataset[0])
