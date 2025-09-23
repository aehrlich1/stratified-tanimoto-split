import itertools
import math

import torch

from STS_utils import fcmdd

def coverage(A, B):
    """
    Percentage of elements in A that are also in B.
    A and B can be any iterables.
    """
    A, B = set(A), set(B)
    if not A:
        return float('nan')  # undefined if A is empty
    return len(A & B) / len(A)
def jaccard_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: (d,) boolean or {0,1}
    a = a.bool()
    b = b.bool()
    inter = (a & b).sum()
    union = (a | b).sum()
    return 1.0 - (inter.float() / (union.float().clamp_min(1)))


def sample_from_cluster_with_impurity_metric(U, target_values, cluster_id, n_samples=5):
    '''
        1) Extract membership of a specific cluster
    '''
    memberships = U[:, cluster_id]
    '''
        2) Sort the membership values to the cluster
    '''
    sorted_values, sorted_indices = torch.sort(memberships, descending=True)

    '''
        3) Extract the quantiles for extracting the distribution
    '''
    probs = (torch.arange(1, n_samples + 1)) / (n_samples + 1)
    quantiles = torch.quantile(target_values, probs).unsqueeze(-1)

    sorted_target_values = target_values[sorted_indices] ## sort by membership
    validation_set_indeces = []
    for quantile in quantiles:
        idx = torch.argmin(torch.abs(sorted_target_values - quantile)) ## argmin return the left most value since it's ordered by membership
        validation_set_indeces.append(sorted_indices[idx].item())
        ## remove the value
        sorted_target_values[idx] = float("inf")

    #val_set_purity = memberships[sorted_indices][validation_set_indeces].mean()
    training_set_indeces = torch.as_tensor(list(set(sorted_indices.tolist()) - set(validation_set_indeces)))
    validation_set_indeces = torch.as_tensor(validation_set_indeces)
    assert training_set_indeces.shape[0] + validation_set_indeces.shape[0] == memberships.shape[0]
    return training_set_indeces, validation_set_indeces

class StratifiedTanimotoSplit:

    def __init__(self, dataset, K: int = 5, split_size: int = 0.2, random_seed: int = 42):

        self.n_val_set = int(split_size * len(dataset))
        self.K = K
        ecfp_dataset = [d.ecfp for d in dataset]
        ecfp_dataset = torch.stack(ecfp_dataset)

        self.D = torch.vmap(torch.vmap(jaccard_distance, in_dims=(0, None)), in_dims=(None, 0))(ecfp_dataset, ecfp_dataset)
        self.fuzzy_labels, self.medoids, self.labels, _ = fcmdd(
            X=None,
            n_clusters=self.K,
            m=1.5,
            max_iter=150,
            tol=1e-12,
            seed=random_seed,
            D=self.D,
            verbose=True,
        )
        '''
            Purity: Percentage of the elements in the validation set that are taken only from the cluster itself, the higher the better
            Intrasimilarity: Average jaccard distance between the elements of the clusters
            Intersimilarity: Average jaccard distance between the centr
        '''


    def split(self, target, smiles = None, return_average_purity=False):
        assert target.shape[0] == self.fuzzy_labels.shape[0]
        train_val_split = []
        average_purity = 0
        purity_list = []
        similarity_list = []
        for i in range(self.K):
            train, val = sample_from_cluster_with_impurity_metric(self.fuzzy_labels, target, cluster_id=i,
                                                                          n_samples=self.n_val_set)

            cluster_idxs = torch.where(self.labels == i)[0].tolist()
            purity = coverage(val.tolist(), cluster_idxs)
            purity_list.append(purity)
            intrasimilarity = torch.triu(1 - self.D[val][:, val]).sum() / ((self.n_val_set * (self.n_val_set - 1))/2)
            similarity_list.append(intrasimilarity)
            train_val_split.append((train, val))
            average_purity += purity




        if return_average_purity:
            intersimilarity = 0
            vals = [val for train, val in train_val_split]
            i = 0
            for val_1, val_2 in itertools.combinations(vals, 2):
                intersimilarity += torch.triu(1 - self.D[val_1][:, val_2]).sum() / ((self.n_val_set * (self.n_val_set - 1))/2)
                i = i + 1
            intersimilarity /= i
            return train_val_split, average_purity / self.K, purity_list, similarity_list, intersimilarity
        return train_val_split