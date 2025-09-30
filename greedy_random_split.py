"""
multiobjective_stratified_split.py
Prototype pipeline to split molecules into Train/Test satisfying:
  1) distribution similarity on numeric properties (Wasserstein per-column),
  2) low inter-set structural similarity (Tanimoto),
  3) high intra-set structural similarity (Tanimoto).
Method: cluster by structure (Butina), then greedy selection of clusters for Test
with multi-objective score. Increase restarts to improve solution.
"""
"""
GreedyKFold for molecular datasets (uses RDKit)

Implements the requested pipeline:
1) Butina clustering on ECFP4
2) Compute bin edges for y (quantiles)
3) Build cluster info from smiles, y, bin edges
4) Compute medoid cluster fingerprints
5) Compute medoid similarity matrix (Tanimoto)
6) Compute weighted avg pairwise / cross similarity helpers
7) Greedy partition clusters into n_splits so that each fold's validation
   set is non-overlapping and ~test_fraction of data. The greedy objective
   minimizes Wasserstein distance of y distributions between train/val,
   minimizes inter-similarity (train vs val), and maximizes intra-similarity
   within train and validation sets.

Usage:
    from greedykfold import GreedyKFold
    gkf = GreedyKFold(n_splits=5, random_state=42)
    splits = gkf.split(smiles, y)  # yields list of (train_idx, val_idx) pairs

Note: RDKit is required.
"""

import math
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.stats import wasserstein_distance
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs import BulkTanimotoSimilarity, FingerprintSimilarity
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator


# ------------------------ utility functions ------------------------

def mol_from_smiles(smi):
    try:
        m = Chem.MolFromSmiles(str(smi).split()[0])  # strip trailing annotations if any
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def fingerprint(smiles, radius=2, nBits=2048):
    m = mol_from_smiles(smiles)
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = gen.GetFingerprint(m)
    return fp



def butina_cluster(fps, cutoff=0.2):
    """Perform Butina clustering on a list of rdkit ExplicitBitVect fingerprints.
    Returns a list of clusters (each cluster a tuple of indices).
    cutoff is the distance cutoff (1 - similarity). Typical similarity cutoff ~0.2-0.4 => distance 0.8-0.6
    We accept cutoff as distance; convert to similarity cutoff (1 - cutoff).
    """
    # compute distance matrix in upper triangular form required by Butina
    n = len(fps)
    if n == 0:
        return []
    dists = []
    for i in range(n):
        fi = fps[i]
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fi, fps[j])
            d = 1.0 - sim
            dists.append(d)
    # Butina expects cutoff as distance threshold
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return clusters


# ------------------------ main class ------------------------


class GreedyKFold:
    def __init__(
        self,
        n_splits=5,
        shuffle=True,
        est_fraction=0.2,
        restarts=3,
        lambda_wass=1.0,
        lambda_inter=1.0,
        lambda_intra=1.0,
        random_state=0,
        butina_cutoff=0.3,
        n_bits=2048,
        fp_radius=2,
        n_bins=10,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.test_fraction = est_fraction
        self.restarts = restarts
        self.lambda_wass = lambda_wass
        self.lambda_inter = lambda_inter
        self.lambda_intra = lambda_intra
        self.random_state = random_state
        self.butina_cutoff = butina_cutoff
        self.n_bits = n_bits
        self.fp_radius = fp_radius
        self.n_bins = n_bins

        np.random.seed(random_state)
        random.seed(random_state)

    # ---------------- preprocessing helpers ----------------

    def compute_bin_edges(self, y, n_bins=None):
        """Compute bin edges for y using quantiles (n_bins default self.n_bins).
        Returns edges (len = n_bins+1).
        """
        if n_bins is None:
            n_bins = self.n_bins
        y = np.asarray(y)
        # use quantiles to ensure roughly balanced bin counts
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(y, quantiles))
        # If edges have duplicates (e.g., many identical y), fall back to small histogram
        if len(edges) <= 2:
            edges = np.histogram_bin_edges(y, bins='auto')
        return edges

    def fingerprint_list(self, smiles):
        fps = []
        for s in smiles:
            try:
                fp = fingerprint(s)
            except Exception as e:
                raise
            fps.append(fp)
        return fps

    def build_cluster_info(self, smiles, y, bin_edges=None):
        """Run butina clustering and produce cluster info dict.
        cluster_info = {cluster_id: { 'indices': [i,...], 'y': [...], 'size': int, 'y_bins': hist }}
        Also returns fingerprints for all molecules and medoid fps.
        """
        fps = self.fingerprint_list(smiles)
        clusters = butina_cluster(fps, cutoff=self.butina_cutoff)

        cluster_info = {}
        for cid, cluster in enumerate(clusters):
            idxs = list(cluster)
            ys = [y[i] for i in idxs]
            cluster_info[cid] = {
                'indices': idxs,
                'size': len(idxs),
                'y': np.array(ys),
            }

        # any molecules not clustered (Butina returns clusters covering all points), but guard anyway
        covered = set(i for c in clusters for i in c)
        others = [i for i in range(len(smiles)) if i not in covered]
        for oi in others:
            cid = len(cluster_info)
            cluster_info[cid] = {'indices': [oi], 'size': 1, 'y': np.array([y[oi]])}

        # bin edges
        if bin_edges is None:
            bin_edges = self.compute_bin_edges(y)

        # compute y_bins histogram per cluster
        for cid, info in cluster_info.items():
            hist, _ = np.histogram(info['y'], bins=bin_edges)
            cluster_info[cid]['y_hist'] = hist

        # compute fingerprints medoid per cluster and intra-cluster avg pairwise Tanimoto
        medoid_fps = {}
        intra_sim = {}
        for cid, info in cluster_info.items():
            idxs = info['indices']
            if len(idxs) == 1:
                medoid_fps[cid] = fps[idxs[0]]
                intra_sim[cid] = 1.0
                continue
            # compute pairwise sims
            sims = np.zeros((len(idxs), len(idxs)))
            for i, ii in enumerate(idxs):
                for j, jj in enumerate(idxs):
                    if j <= i:
                        continue
                    sims[i, j] = DataStructs.TanimotoSimilarity(fps[ii], fps[jj])
                    sims[j, i] = sims[i, j]
            # medoid: maximize sum of sims to others
            sums = sims.sum(axis=1)
            medoid_idx = idxs[int(np.argmax(sums))]
            medoid_fps[cid] = fps[medoid_idx]
            # intra-cluster average pairwise similarity (upper triangle)
            if len(idxs) > 1:
                iu = np.triu_indices(len(idxs), k=1)
                mean_pairwise = sims[iu].mean() if len(iu[0]) > 0 else 1.0
            else:
                mean_pairwise = 1.0
            intra_sim[cid] = mean_pairwise

        return cluster_info, medoid_fps, fps, bin_edges, clusters

    def medoid_similarity_matrix(self, medoid_fps, cluster_ids=None):
        """Return dict-of-dicts similarity matrix between cluster medoids.
        sim[c1][c2] = tanimoto(medoid_fps[c1], medoid_fps[c2])
        """
        if cluster_ids is None:
            cluster_ids = sorted(medoid_fps.keys())
        sim = {c: {} for c in cluster_ids}
        for i, c1 in enumerate(cluster_ids):
            for j, c2 in enumerate(cluster_ids):
                if j < i:
                    sim[c1][c2] = sim[c2][c1]
                    continue
                s = DataStructs.TanimotoSimilarity(medoid_fps[c1], medoid_fps[c2])
                sim[c1][c2] = s
        return sim

    def weighted_cross_similarity(self, sim_matrix, cluster_info, setA, setB):
        """Compute weighted average similarity between clusters in setA and setB.
        We weight by cluster sizes.
        """
        if len(setA) == 0 or len(setB) == 0:
            return 0.0
        total = 0.0
        weight = 0.0
        for a in setA:
            for b in setB:
                s = sim_matrix[a][b]
                w = cluster_info[a]['size'] * cluster_info[b]['size']
                total += s * w
                weight += w
        return total / weight if weight > 0 else 0.0

    def weighted_intra_similarity(self, sim_matrix, cluster_info, setA):
        """Compute weighted intra similarity for setA. Combines within-cluster intra and between-cluster medoid similarities.
        """
        if len(setA) == 0:
            return 0.0
        # within-cluster intra (weighted)
        total_within = 0.0
        weight_within = 0.0
        for a in setA:
            total_within += cluster_info[a].get('size', 1) * cluster_info[a].get('intra_sim', 1.0)
            weight_within += cluster_info[a].get('size', 1)
        avg_within = total_within / weight_within if weight_within > 0 else 0.0
        # between-cluster medoid similarities (weighted)
        total_between = 0.0
        weight_between = 0.0
        for i, a in enumerate(setA):
            for j, b in enumerate(setA):
                if j <= i:
                    continue
                s = sim_matrix[a][b]
                w = cluster_info[a]['size'] * cluster_info[b]['size']
                total_between += s * w
                weight_between += w
        avg_between = total_between / weight_between if weight_between > 0 else 0.0
        # combine (give within-cluster more weight)
        return 0.6 * avg_within + 0.4 * (avg_between if weight_between > 0 else avg_within)

    # ---------------- objective & assignment ----------------

    def _compute_fold_objective(self, cluster_info, sim_matrix, val_cluster_ids):
        """Compute objective for one fold given its validation cluster ids.
        Objective = lambda_wass * wass(y_train, y_val)
                    + lambda_inter * inter_sim(train, val)
                    - lambda_intra * (intra_train + intra_val)
        We try to minimize objective.
        """
        all_cids = set(cluster_info.keys())
        val_set = set(val_cluster_ids)
        train_set = sorted(list(all_cids - val_set))
        val_set = sorted(list(val_set))

        # gather y
        y_val = np.concatenate([cluster_info[c]['y'] for c in val_set]) if len(val_set) > 0 else np.array([])
        y_train = np.concatenate([cluster_info[c]['y'] for c in train_set]) if len(train_set) > 0 else np.array([])

        if len(y_val) == 0 or len(y_train) == 0:
            return 1e6, dict(wass=np.nan, inter=np.nan, intra_train=np.nan, intra_val=np.nan)
        wass = wasserstein_distance(y_train, y_val)

        # inter similarity
        inter_sim = self.weighted_cross_similarity(sim_matrix, cluster_info, val_set, train_set)
        # intra
        intra_val = self.weighted_intra_similarity(sim_matrix, cluster_info, val_set)
        intra_train = self.weighted_intra_similarity(sim_matrix, cluster_info, train_set)

        obj = self.lambda_wass * wass + self.lambda_inter * inter_sim - self.lambda_intra * (intra_train + intra_val)
        return obj, dict(wass=wass, inter=inter_sim, intra_train=intra_train, intra_val=intra_val)

    def split(self, smiles, y):
        """Main entry point.
        Returns list of (train_idx, val_idx) for each fold. The validation sets are non-overlapping.
        """
        # input validation
        if len(smiles) != len(y):
            raise ValueError("smiles and y must have same length")
        n = len(smiles)
        desired_val_size = int(math.ceil(self.test_fraction * n))

        # 1-5 preprocessing
        cluster_info, medoid_fps, fps, bin_edges, clusters = self.build_cluster_info(smiles, y, bin_edges=None)
        # add intra_sim into cluster_info
        for cid in cluster_info:
            # cluster_info already had 'indices' and 'y'; we added intra_sim via return tuple
            # but ensure we have it: compute if missing
            if 'intra_sim' not in cluster_info[cid]:
                cluster_info[cid]['intra_sim'] = 1.0
        # recompute intra_sim and medoid_fps were returned separately; attach intra_sim
        # We need to recompute intra_sim and medoid_fps by reusing build results: but build returned medoid_fps and intra_sim local var
        # To avoid complexity, recompute medoid_fps and intra_sim here using cluster_info + fps
        medoid_fps = {}
        intra_sim_local = {}
        for cid, info in cluster_info.items():
            idxs = info['indices']
            if len(idxs) == 1:
                medoid_fps[cid] = fps[idxs[0]]
                intra_sim_local[cid] = 1.0
            else:
                sims = np.zeros((len(idxs), len(idxs)))
                for i, ii in enumerate(idxs):
                    for j, jj in enumerate(idxs):
                        if j <= i:
                            continue
                        sims[i, j] = DataStructs.TanimotoSimilarity(fps[ii], fps[jj])
                        sims[j, i] = sims[i, j]
                sums = sims.sum(axis=1)
                medoid_idx = idxs[int(np.argmax(sums))]
                medoid_fps[cid] = fps[medoid_idx]
                iu = np.triu_indices(len(idxs), k=1)
                mean_pairwise = sims[iu].mean() if len(iu[0]) > 0 else 1.0
                intra_sim_local[cid] = mean_pairwise
            cluster_info[cid]['intra_sim'] = intra_sim_local[cid]

        # 5 compute medoid similarity matrix
        sim_matrix = self.medoid_similarity_matrix(medoid_fps)

        # prepare list of clusters sorted by size descending
        cids = sorted(cluster_info.keys(), key=lambda c: cluster_info[c]['size'], reverse=True)

        best_partition = None
        best_score = float('inf')

        for restart in range(self.restarts):
            if self.shuffle:
                random.shuffle(cids)

            # initialize empty folds: each fold holds list of cluster ids for validation
            folds = [[] for _ in range(self.n_splits)]
            fold_sizes = [0] * self.n_splits

            # greedy: assign each cluster to the fold that produces the lowest objective after assignment
            for pos,cid in enumerate(cids):
                remaining = cids[pos+1:]  # clusters still to be assigned after this one
                remaining_total = sum(cluster_info[c]['size'] for c in remaining)
                # candidate assignment: choose fold minimizing objective for that fold after adding cid
                best_local_fold = None
                best_local_obj = float('inf')
                for f in range(self.n_splits):
                    candidate_val = folds[f] + [cid]
                    # ensure candidate fold doesn't exceed desired_val_size by too much
                    cand_size = sum(cluster_info[c]['size'] for c in candidate_val)
                    # penalize overfull candidate strongly
                    over_penalty = 0.0
                    if cand_size > desired_val_size * 1.15:  # allow small slack
                        over_penalty = (cand_size - desired_val_size) / desired_val_size * 10.0
                    obj, _ = self._compute_fold_objective(cluster_info, sim_matrix, candidate_val)
                    obj = obj + over_penalty
                    # also prefer folds with smaller current sizes to balance
                    obj = obj + 0.01 * fold_sizes[f]

                    new_fold_sizes = fold_sizes.copy()
                    new_fold_sizes[f] = cand_size
                    deficits = [max(0, desired_val_size - s) for s in new_fold_sizes]
                    total_deficit = sum(deficits)
                    if total_deficit > remaining_total:
                        # infeasible: not enough remaining samples to fulfill all folds' targets
                        # discourage heavily
                        obj += 1e5

                    if obj < best_local_obj:
                        best_local_obj = obj
                        best_local_fold = f
                # assign
                folds[best_local_fold].append(cid)
                fold_sizes[best_local_fold] = sum(cluster_info[c]['size'] for c in folds[best_local_fold])

            # after assignment, compute total score across folds (sum objectives)
            total_score = 0.0
            for f in range(self.n_splits):
                obj, stats = self._compute_fold_objective(cluster_info, sim_matrix, folds[f])
                total_score += obj
            if total_score < best_score:
                best_score = total_score
                best_partition = deepcopy(folds)

        # build train/val indices for each fold and ensure about right fraction
        # splits = []
        for f in range(self.n_splits):
            val_cids = best_partition[f]
            val_idx = [i for c in val_cids for i in cluster_info[c]['indices']]
            train_idx = [i for c in cluster_info.keys() for i in cluster_info[c]['indices'] if c not in val_cids]
            # final checks: if validation fraction deviates too much, try to adjust (simple balancing)
            # final tiny safety: if val_idx empty (shouldn't happen), move smallest cluster from largest fold
            if len(val_idx) == 0:
                donor = max(range(self.n_splits), key=lambda x: sum(cluster_info[c]['size'] for c in best_partition[x]))
                if donor != f and len(best_partition[donor])>0:
                    smallest_c = min(best_partition[donor], key=lambda c: cluster_info[c]['size'])
                    best_partition[donor].remove(smallest_c)
                    best_partition[f].append(smallest_c)
                    val_idx = [i for c in best_partition[f] for i in cluster_info[c]['indices']]
                    train_idx = [i for c in cluster_info.keys() for i in cluster_info[c]['indices'] if c not in best_partition[f]]

    
            yield np.array(train_idx), np.array(val_idx)


if __name__ == '__main__':
    # small test example (user should replace with real data)
    sample_smiles = ['CCO', 'CCN', 'CCC', 'CCCC', 'CCCl', 'c1ccccc1', 'c1ccncc1', 'CC(=O)O', 'C1CCCCC1', 'CCS']
    sample_y = np.random.randn(len(sample_smiles))
    gkf = GreedyKFold(n_splits=5, restarts=2, random_state=42)
    splits = gkf.split(sample_smiles, sample_y)
    for i, (tr, va) in enumerate(splits):
        print(f"Fold {i}: train {len(tr)} val {len(va)} \n train:{tr} \n val: {va}")
