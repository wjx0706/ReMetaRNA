import os
from abc import ABC, abstractmethod
from typing import Callable
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, Batch
from tqdm.auto import tqdm


class BaseRNADataset(Dataset, ABC):
    def __init__(
        self,
        csv_data_path: str,
        coord_npy_dir: str,
        threshold: Union[int,float],
        use_full_backbone: bool = False,
        edge_data_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.use_full_backbone = use_full_backbone
        self.raw_data = pd.read_csv(csv_data_path)
        self.name_list = self.raw_data["pdb_id"].to_list()
        self.seq_list = self.raw_data["seq"].to_list()

        if edge_data_cache_dir is None:
            edge_data_cache_dir = os.path.join(
                os.path.dirname(csv_data_path), "edge_cache"
            )
        os.makedirs(edge_data_cache_dir, exist_ok=True)
        self.edge_data_cache_dir = edge_data_cache_dir

        self.edge_index_cache_dir = os.path.join(
            edge_data_cache_dir, "edge_index_cache"
        )
        os.makedirs(self.edge_index_cache_dir, exist_ok=True)
        self.edge_attr_cache_dir = os.path.join(edge_data_cache_dir, "edge_attr_cache")
        os.makedirs(self.edge_attr_cache_dir, exist_ok=True)

        self.available_cache_files = self.scan_cache_files()

    def scan_cache_files(self) -> dict:
        """
        Scan the cache directories for existing edge index and attribute files.
        Returns:
            A dictionary mapping PDB IDs to a set of chunk IDs for which cache files exist.
        """
        cache_files = {}

        for file_name in os.listdir(self.edge_index_cache_dir):
            if file_name.endswith(".npy"):
                base_name = file_name.replace(".npy", "")
                pdb_id, chunk_id = base_name.rsplit("_", 1)
                cache_files.setdefault(pdb_id, set()).add(chunk_id)

        # assume edge_attr exists if edge_index exists
        return cache_files

    def tokenize(self, seq: str) -> np.ndarray:
        mapping = {"A": 0, "U": 1, "C": 2, "G": 3}  # [UNK]: 4
        return np.array([mapping.get(char, 4) for char in seq], dtype=int)

    def process_structure_data(
        self, coords: np.ndarray, seq: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seq_len = len(seq)
        if self.use_full_backbone:
            coords = coords.reshape(-1, 3)
            atom_ids = np.tile(np.arange(7), seq_len)
            nucl_ids = np.repeat(np.arange(seq_len), 7)
        else:
            coords = coords[:, [0, 3, 6], :].reshape(-1, 3)  # P, C4', N
            atom_ids = np.tile(np.arange(3), seq_len)
            nucl_ids = np.repeat(np.arange(seq_len), 3)

        # filter nan values
        valid_mask = ~np.isnan(coords).any(axis=1)
        valid_coords = coords[valid_mask]
        # if valid_coords.shape[0] == 1:
        #     return None

        valid_atom_ids = atom_ids[valid_mask]
        valid_nucl_ids = nucl_ids[valid_mask]
        valid_seq_ids, valid_nucl_ids = np.unique(valid_nucl_ids, return_inverse=True)
        tokenized_seq = self.tokenize(seq)[valid_seq_ids]

        return valid_coords, valid_atom_ids, valid_nucl_ids, tokenized_seq

    def get_cache_availability(self, pdb_id: str, chunk_id: Union[str, int]) -> bool:
        """
        Check if the cache files for the given PDB ID and chunk ID are available.
        """
        chunk_id = str(chunk_id)
        if pdb_id in self.available_cache_files:
            return chunk_id in self.available_cache_files[pdb_id]
        return False

    def build_graphs(
        self,
        coords: np.ndarray,
        atom_ids: np.ndarray,
        nucl_ids: np.ndarray,
        seq: np.ndarray,
        pdb_id: str,
        chunk_id: int,
    ) -> Data:
        num_atoms = coords.shape[0]
        edge_index_cache_path = os.path.join(
            self.edge_index_cache_dir, f"{pdb_id}_{chunk_id}.npy"
        )
        edge_attr_cache_path = os.path.join(
            self.edge_attr_cache_dir, f"{pdb_id}_{chunk_id}.npy"
        )

        if self.get_cache_availability(pdb_id, chunk_id):
            edge_index = np.load(edge_index_cache_path)
            edge_attr = np.load(edge_attr_cache_path)
        else:
            # add self-loop
            edge_index = np.tile(np.arange(num_atoms), (2, 1))
            edge_attr = np.zeros(num_atoms)  # self-loop distance is 0.0

            if num_atoms > 1:
                rows, cols = np.triu_indices(num_atoms, k=1)
                # calculate distances between all pairs of atoms
                distances = np.linalg.norm(coords[rows] - coords[cols], axis=1)
                mask = distances <= self.threshold
                filtered_rows = rows[mask]
                filtered_cols = cols[mask]
                filtered_distances = distances[mask]

                src_to_dst = np.stack([filtered_rows, filtered_cols], axis=0)
                dst_to_src = np.stack([filtered_cols, filtered_rows], axis=0)
                bidirectional_edges = np.concatenate([src_to_dst, dst_to_src], axis=1)
                bidirectional_distances = np.concatenate(
                    [filtered_distances, filtered_distances], axis=0
                )
                edge_index = np.concatenate([edge_index, bidirectional_edges], axis=1)
                edge_attr = np.concatenate([edge_attr, bidirectional_distances], axis=0)

            # save edge data to cache
            np.save(edge_index_cache_path, edge_index)
            np.save(edge_attr_cache_path, edge_attr)

        return Data(
            x=torch.tensor(
                coords, dtype=torch.float32
            ),  # atom coordinates, [num_atoms, 3]
            edge_index=torch.tensor(edge_index, dtype=torch.long),  # [2, num_edges]
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),  # [num_edges]
            atom_ids=torch.tensor(atom_ids, dtype=torch.long),  # [num_atoms]
            nucl_ids=torch.tensor(nucl_ids, dtype=torch.long),  # [num_atoms]
            seq=torch.tensor(seq, dtype=torch.long),  # tokenized sequence
            # num_nodes=num_atoms,
            pdb_id=pdb_id,
        )

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Data:
        pass


class RNADataset(BaseRNADataset):
    """
    Dataset for RNA structures from PDB files.
    Loads RNA 3D coordinates and creates graph representations with
        edges between atoms within a distance threshold.
    """

    def __init__(
        self,
        csv_data_path: str,
        coord_npy_dir: str,
        threshold: Union[int,float],
        use_full_backbone: bool = False,
        edge_data_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            csv_data_path,
            coord_npy_dir,
            threshold,
            use_full_backbone,
            edge_data_cache_dir,
        )
        self.data_list = []

        for idx, pdb_id in enumerate(tqdm(self.name_list)):
            # coord: [len(seq), 7, 3]
            coords = np.load(os.path.join(coord_npy_dir, pdb_id + ".npy"))
            seq = self.seq_list[idx]
            processed_data = self.process_structure_data(coords, seq)

            if processed_data is None:
                print(f"{pdb_id = } has only one valid atom, skipping.")
                continue

            valid_coords, valid_atom_ids, valid_nucl_ids, tokenized_seq = processed_data

            # normalize coords (center around 0)
            centroid = np.mean(valid_coords, axis=0)
            normalized_coords = valid_coords - centroid

            self.data_list.append(
                self.build_graphs(
                    normalized_coords,
                    valid_atom_ids,
                    valid_nucl_ids,
                    tokenized_seq,
                    pdb_id=pdb_id,
                    chunk_id=0,  # not used in this dataset
                )
            )

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


class RNADatasetChunked(BaseRNADataset):
    def __init__(
        self,
        csv_data_path: str,
        coord_npy_dir: str,
        threshold: Union[int,float],
        use_full_backbone: bool = False,
        max_nucl_per_chunk: int = 300,
        chunk_overlap: int = 0,
        edge_data_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            csv_data_path,
            coord_npy_dir,
            threshold,
            use_full_backbone,
            edge_data_cache_dir,
        )
        self.max_nucl_per_chunk = max_nucl_per_chunk
        self.chunk_overlap = chunk_overlap
        self.data_list = []

        for idx, pdb_id in enumerate(tqdm(self.name_list, desc="Processing dataset")):
            # coord: [len(seq), 7, 3]
            full_coords = np.load(os.path.join(coord_npy_dir, pdb_id + ".npy"))
            full_seqs = self.seq_list[idx]

            chunk_coords, chunk_seqs = self.chunk_sequence(full_coords, full_seqs)

            for chunk_id, (coords, seq) in enumerate(zip(chunk_coords, chunk_seqs)):
                processed_data = self.process_structure_data(coords, seq)
                if processed_data is None:
                    print(f"{pdb_id = } has only one valid atom, skipping.")
                    continue
                valid_coords, valid_atom_ids, valid_nucl_ids, tokenized_seq = (
                    processed_data
                )

                # normalize coords (center around 0)
                centroid = np.mean(valid_coords, axis=0)
                normalized_coords = valid_coords - centroid

                self.data_list.append(
                    self.build_graphs(
                        normalized_coords,
                        valid_atom_ids,
                        valid_nucl_ids,
                        tokenized_seq,
                        pdb_id=pdb_id,
                        chunk_id=chunk_id,  # use the index of the original structure
                    )
                )

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]

    def chunk_sequence(
        self, full_coords: np.ndarray, full_seqs: np.ndarray
    ) -> tuple[list, list]:
        seq_len = len(full_seqs)

        if seq_len < self.max_nucl_per_chunk:
            chunk_coords = [full_coords]
            chunk_seqs = [full_seqs]
        else:
            chunk_coords = []
            chunk_seqs = []
            step = self.max_nucl_per_chunk - self.chunk_overlap
            start_positions = list(range(0, seq_len, step))

            for start_pos in start_positions:
                end_pos = min(start_pos + self.max_nucl_per_chunk, seq_len)
                segment_coords = full_coords[start_pos:end_pos]
                segment_seqs = full_seqs[start_pos:end_pos]
                if segment_coords.shape[0] < self.max_nucl_per_chunk:
                    # merge with the previous chunk
                    if chunk_coords:  # actually, this should always be true
                        chunk_coords[-1] = np.concatenate(
                            (chunk_coords[-1], segment_coords), axis=0
                        )
                        chunk_seqs[-1] += segment_seqs
                else:
                    chunk_coords.append(segment_coords)
                    chunk_seqs.append(segment_seqs)

        return chunk_coords, chunk_seqs


class RNADatasetInference(BaseRNADataset):
    def __init__(
        self,
        raw_data: pd.DataFrame,
        coord_npy_dir: str,
        threshold: Union[int,float],
        use_full_backbone: bool = False,
        max_nucl_per_chunk: int = 300,
        chunk_overlap: int = 0,
        edge_data_cache_dir: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.use_full_backbone = use_full_backbone
        self.raw_data = raw_data
        self.name_list = self.raw_data["pdb_id"].to_list()
        self.seq_list = self.raw_data["seq"].to_list()

        self.max_nucl_per_chunk = max_nucl_per_chunk
        self.chunk_overlap = chunk_overlap

        # keep original sequences as the main index
        self.data_batches = []

        for idx, pdb_id in enumerate(tqdm(self.name_list, desc="Processing dataset")):
            # coord: [len(seq), 7, 3]
            full_coords = np.load(os.path.join(coord_npy_dir, pdb_id + ".npy"))
            full_seqs = self.seq_list[idx]

            chunk_coords, chunk_seqs = self.chunk_sequence(full_coords, full_seqs)

            data_batch = []
            for chunk_id, (coords, seq) in enumerate(zip(chunk_coords, chunk_seqs)):
                processed_data = self.process_structure_data(coords, seq)
                # if processed_data is None:
                #     print(f"{pdb_id = } has only one valid atom, skipping.")
                #     continue
                valid_coords, valid_atom_ids, valid_nucl_ids, tokenized_seq = (
                    processed_data
                )

                # normalize coords (center around 0)
                centroid = np.mean(valid_coords, axis=0)
                normalized_coords = valid_coords - centroid

                data_batch.append(
                    self.build_graphs(
                        normalized_coords,
                        valid_atom_ids,
                        valid_nucl_ids,
                        tokenized_seq,
                        pdb_id=pdb_id,
                        chunk_id=chunk_id,  # use the index of the original structure
                    )
                )
            self.data_batches.append(Batch.from_data_list(data_batch))

    def __len__(self) -> int:
        return len(self.data_batches)

    def __getitem__(self, idx: int) -> Data:
        return self.data_batches[idx]

    def chunk_sequence(
        self, full_coords: np.ndarray, full_seqs: np.ndarray
    ) -> tuple[list, list]:
        seq_len = len(full_seqs)

        if seq_len < self.max_nucl_per_chunk:
            chunk_coords = [full_coords]
            chunk_seqs = [full_seqs]
        else:
            chunk_coords = []
            chunk_seqs = []
            step = self.max_nucl_per_chunk - self.chunk_overlap
            start_positions = list(range(0, seq_len, step))

            for start_pos in start_positions:
                end_pos = min(start_pos + self.max_nucl_per_chunk, seq_len)
                segment_coords = full_coords[start_pos:end_pos]
                segment_seqs = full_seqs[start_pos:end_pos]
                if segment_coords.shape[0] < self.max_nucl_per_chunk:
                    # merge with the previous chunk
                    if chunk_coords:  # actually, this should always be true
                        chunk_coords[-1] = np.concatenate(
                            (chunk_coords[-1], segment_coords), axis=0
                        )
                        chunk_seqs[-1] += segment_seqs
                else:
                    chunk_coords.append(segment_coords)
                    chunk_seqs.append(segment_seqs)

        return chunk_coords, chunk_seqs
    
    def process_structure_data(
        self, coords: np.ndarray, seq: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seq_len = len(seq)
        if self.use_full_backbone:
            coords = coords.reshape(-1, 3)
            atom_ids = np.tile(np.arange(7), seq_len)
            nucl_ids = np.repeat(np.arange(seq_len), 7)
        else:
            coords = coords[:, [0, 3, 6], :].reshape(-1, 3)  # P, C4', N
            atom_ids = np.tile(np.arange(3), seq_len)
            nucl_ids = np.repeat(np.arange(seq_len), 3)

        # filter nan values
        coords = np.nan_to_num(coords, nan=0.0)
        valid_mask = ~np.isnan(coords).any(axis=1)
        valid_coords = coords[valid_mask]
        # if valid_coords.shape[0] == 1:
        #     return None

        valid_atom_ids = atom_ids[valid_mask]
        valid_nucl_ids = nucl_ids[valid_mask]
        valid_seq_ids, valid_nucl_ids = np.unique(valid_nucl_ids, return_inverse=True)
        tokenized_seq = self.tokenize(seq)[valid_seq_ids]

        return valid_coords, valid_atom_ids, valid_nucl_ids, tokenized_seq
    
    def build_graphs(
        self,
        coords: np.ndarray,
        atom_ids: np.ndarray,
        nucl_ids: np.ndarray,
        seq: np.ndarray,
        pdb_id: str,
        chunk_id: int,
    ) -> Data:
        num_atoms = coords.shape[0]

        # add self-loop
        edge_index = np.tile(np.arange(num_atoms), (2, 1))
        edge_attr = np.zeros(num_atoms)  # self-loop distance is 0.0

        if num_atoms > 1:
            rows, cols = np.triu_indices(num_atoms, k=1)
            # calculate distances between all pairs of atoms
            distances = np.linalg.norm(coords[rows] - coords[cols], axis=1)
            mask = distances <= self.threshold
            filtered_rows = rows[mask]
            filtered_cols = cols[mask]
            filtered_distances = distances[mask]

            src_to_dst = np.stack([filtered_rows, filtered_cols], axis=0)
            dst_to_src = np.stack([filtered_cols, filtered_rows], axis=0)
            bidirectional_edges = np.concatenate([src_to_dst, dst_to_src], axis=1)
            bidirectional_distances = np.concatenate(
                [filtered_distances, filtered_distances], axis=0
            )
            edge_index = np.concatenate([edge_index, bidirectional_edges], axis=1)
            edge_attr = np.concatenate([edge_attr, bidirectional_distances], axis=0)

        return Data(
            x=torch.tensor(
                coords, dtype=torch.float32
            ),  # atom coordinates, [num_atoms, 3]
            edge_index=torch.tensor(edge_index, dtype=torch.long),  # [2, num_edges]
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),  # [num_edges]
            atom_ids=torch.tensor(atom_ids, dtype=torch.long),  # [num_atoms]
            nucl_ids=torch.tensor(nucl_ids, dtype=torch.long),  # [num_atoms]
            seq=torch.tensor(seq, dtype=torch.long),  # tokenized sequence
            # num_nodes=num_atoms,
            pdb_id=pdb_id,
        )

class RandomRotation:
    """Applies a random rotation to the node coordinates."""

    def __call__(self, data: Data) -> Data:
        # assume node features are 3D coordinates
        pos = data.x  # shape: [num_nodes, 3]
        assert pos is not None and pos.size(1) == 3, "Expected 3D coordinates in data.x"

        # generate a random quaternion
        q = torch.rand(4)
        q = q / q.norm()
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        R = torch.tensor(
            [
                [
                    1 - 2 * q2**2 - 2 * q3**2,
                    2 * q1 * q2 - 2 * q0 * q3,
                    2 * q1 * q3 + 2 * q0 * q2,
                ],
                [
                    2 * q1 * q2 + 2 * q0 * q3,
                    1 - 2 * q1**2 - 2 * q3**2,
                    2 * q2 * q3 - 2 * q0 * q1,
                ],
                [
                    2 * q1 * q3 - 2 * q0 * q2,
                    2 * q2 * q3 + 2 * q0 * q1,
                    1 - 2 * q1**2 - 2 * q2**2,
                ],
            ]
        )

        # apply rotation to coordinates
        data.x = pos @ R.T
        return data


class RandomTranslation:
    """Applies a random translation to the node coordinates."""
    def __call__(self, data: Data) -> Data:
        # assume node features are 3D coordinates
        pos = data.x  # shape: [num_nodes, 3]
        assert pos is not None and pos.size(1) == 3, "Expected 3D coordinates in data.x"

        # generate a random translation vector
        translation_vector = (torch.rand(3) - 0.5) * 4  # scale to [-2, 2]
        translation_vector = translation_vector.unsqueeze(0)  # shape: [1, 3]

        # apply translation to coordinates
        data.x = pos + translation_vector
        return data


class GaussianNoise:
    """Applies Gaussian noise to the node coordinates."""

    def __init__(self, std: float = 0.1) -> None:
        self.std = std

    def __call__(self, data: Data) -> Data:
        # assume node features are 3D coordinates
        pos = data.x  # shape: [num_nodes, 3]
        assert pos is not None and pos.size(1) == 3, "Expected 3D coordinates in data.x"

        # generate Gaussian noise
        noise = torch.randn_like(pos) * self.std

        # apply noise to coordinates
        data.x = pos + noise
        return data


class AugmentTransform:
    """Applies a random transformation to the node coordinates."""
    def __init__(self) -> None:
        self.transforms = [RandomRotation(), RandomTranslation(), GaussianNoise()]

    def __call__(self, data: Data) -> Data:
        # Apply a random transformation to the data
        seed = torch.rand(1).item()
        if seed < 0.7:
            transform = self.transforms[0]  # RandomRotation
        elif seed < 0.9:
            transform = self.transforms[1]  # RandomTranslation
        else:
            transform = self.transforms[2]  # GaussianNoise

        return transform(data)
        

class ContrastiveDataset(Dataset):
    """
    Dataset wrapper for contrastive learning with RNA structures.
    Applies transformations to create different views for contrastive learning.
    """

    def __init__(self, base_dataset: BaseRNADataset, transform: Callable) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[Data, Data]:
        data = self.base_dataset[idx]
        assert data.x is not None, "Data.x should not be None"

        aug_data = self.transform(Data(x=data.x.clone()))
        return data, aug_data
