"""
data/featurizer.py
3D molecular graph construction from SMILES.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from typing import Optional, Tuple


class Molecule3DFeaturizer:
    """
    Convert SMILES → 3D molecular graph (PyG Data object).
    
    Node features:
      - Atomic number (one-hot, 1-100)
      - Degree (one-hot, 0-5)
      - Formal charge (one-hot, -2 to +2)
      - Hybridization (one-hot, SP, SP2, SP3, SP3D, SP3D2)
      - Aromaticity (binary)
      - Total Hs (one-hot, 0-4)
    
    Edge features:
      - Bond type (single, double, triple, aromatic)
      - Conjugation (binary)
      - Ring membership (binary)
    
    Geometric features:
      - 3D coordinates (x, y, z) from RDKit ETKDG conformer
    """

    # Feature vocabularies
    ATOM_FEATURES = {
        "atomic_num": list(range(1, 101)),
        "degree": [0, 1, 2, 3, 4, 5],
        "formal_charge": [-2, -1, 0, 1, 2],
        "hybridization": [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
        "is_aromatic": [False, True],
        "num_hs": [0, 1, 2, 3, 4],
    }

    def __init__(self, use_3d: bool = True):
        self.use_3d = use_3d

    @staticmethod
    def _one_hot(value, choices):
        """One-hot encoding with unknown handling."""
        encoding = [0] * (len(choices) + 1)
        try:
            idx = choices.index(value)
        except ValueError:
            idx = -1  # Unknown value → last position
        encoding[idx] = 1
        return encoding

    def atom_features(self, atom) -> list:
        """Extract node features from RDKit Atom."""
        feats = []
        feats += self._one_hot(atom.GetAtomicNum(), self.ATOM_FEATURES["atomic_num"])
        feats += self._one_hot(atom.GetDegree(), self.ATOM_FEATURES["degree"])
        feats += self._one_hot(atom.GetFormalCharge(), self.ATOM_FEATURES["formal_charge"])
        feats += self._one_hot(atom.GetHybridization(), self.ATOM_FEATURES["hybridization"])
        feats += self._one_hot(atom.GetIsAromatic(), self.ATOM_FEATURES["is_aromatic"])
        feats += self._one_hot(atom.GetTotalNumHs(), self.ATOM_FEATURES["num_hs"])
        return feats

    @staticmethod
    def bond_features(bond) -> list:
        """Extract edge features from RDKit Bond."""
        bt = bond.GetBondType()
        return [
            float(bt == Chem.rdchem.BondType.SINGLE),
            float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE),
            float(bt == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit Mol with 3D conformer.
        
        Returns None if generation fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        if self.use_3d:
            # Generate 3D conformer using ETKDG
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            success = AllChem.EmbedMolecule(mol, params)
            
            if success == -1:
                # Fallback to 2D if 3D fails
                AllChem.Compute2DCoords(mol)
                return mol
            
            # Optimize geometry with MMFF
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass  # Optimization can fail for certain molecules

        return mol

    def mol_to_graph(
        self,
        mol: Chem.Mol,
        label: Optional[float] = None,
        task_name: Optional[str] = None,
    ) -> Optional[Data]:
        """
        Convert RDKit Mol to PyG Data object.
        
        Parameters
        ----------
        mol       : RDKit Mol
        label     : float (optional) – binary classification label
        task_name : str (optional) – task identifier
        
        Returns
        -------
        torch_geometric.data.Data or None if conversion fails
        """
        if mol is None:
            return None

        # Node features
        atom_feats = [self.atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)

        # 3D positions
        if self.use_3d:
            try:
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)
            except:
                # No conformer available → use zeros
                pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)
        else:
            pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)

        # Edges (undirected graph)
        edge_indices = []
        edge_feats = []

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = self.bond_features(bond)
            
            # Add both directions
            edge_indices.extend([[i, j], [j, i]])
            edge_feats.extend([bf, bf])

        if len(edge_indices) == 0:
            # Molecule with no bonds (e.g., single atom)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            num_nodes=mol.GetNumAtoms(),
        )

        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float)

        if task_name is not None:
            data.task_name = task_name

        return data

    def get_feature_dims(self) -> Tuple[int, int]:
        """
        Returns (node_dim, edge_dim).
        """
        node_dim = sum(len(v) + 1 for v in self.ATOM_FEATURES.values())
        edge_dim = 6
        return node_dim, edge_dim


def featurize_smiles(smiles: str, label: Optional[float] = None) -> Optional[Data]:
    """
    Convenience function: SMILES → PyG Data.
    """
    featurizer = Molecule3DFeaturizer(use_3d=True)
    mol = featurizer.smiles_to_mol(smiles)
    return featurizer.mol_to_graph(mol, label=label)