import torch

def jaccard_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Jaccard distance between two binary 1D tensors.
    """
    intersection = torch.sum(x * y).float()
    union = torch.sum((x + y) > 0).float()
    return  torch.where(union == 0, torch.tensor(0.0), 1.0 - intersection / union )

def fuzzy_k_medoids_from_distance(X, k, m=2, max_iter=100, eps=1e-6):
    D = torch.vmap(torch.vmap(jaccard_distance, in_dims=(None, 0)), in_dims=(0, None))(X, X)
    """
    Fuzzy K-medoids clustering from a precomputed distance matrix.

    D: (N, N) distance matrix (symmetric, 0 diag)
    k: number of clusters
    m: fuzziness parameter (>1)
    """
    N = D.shape[0]
    gen = torch.Generator().manual_seed(42)
    # Initialize medoids randomly
    medoid_indices = torch.randperm(N, generator=gen)[:k]

    for _ in range(max_iter):
        # Compute memberships U (N,k)
        d_to_medoids = D[:, medoid_indices]  # (N,k)
        d_to_medoids = d_to_medoids.clamp(min=1e-8)

        ratio = (d_to_medoids.unsqueeze(2) / d_to_medoids.unsqueeze(1)) ** (2 / (m - 1))
        U = 1.0 / ratio.sum(dim=2)

        # Update medoids: pick the point minimizing weighted distance in each cluster
        new_medoids = []
        for j in range(k):
            weights = (U[:, j] ** m)  # (N,)
            weighted_distances = (weights.unsqueeze(0) * D).sum(dim=1)
            new_medoids.append(torch.argmin(weighted_distances).item())
        new_medoids = torch.tensor(new_medoids)

        if torch.equal(new_medoids, medoid_indices):
            break
        medoid_indices = new_medoids

    return U, medoid_indices


def sample_from_cluster_with_impurity_metric(U, target_values, cluster_id, n_samples=5, threshold=0.8, with_resampling=False):
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
        validation_set_indeces.append(idx.item())
        ## remove the value
        sorted_target_values[idx] = float("inf")

    val_set_purity = memberships[sorted_indices][validation_set_indeces].mean()
    training_set_indeces = torch.as_tensor(list(set(sorted_indices.tolist()) - set(validation_set_indeces)))
    validation_set_indeces = torch.as_tensor(validation_set_indeces)
    assert training_set_indeces.shape[0] + validation_set_indeces.shape[0] == memberships.shape[0]
    return training_set_indeces, validation_set_indeces, val_set_purity

class StratifiedTanimotoSplit:

    def __init__(self, dataset, K: int = 5, split_size: int = 0.2):

        self.n_val_set = int(split_size * len(dataset))
        self.K = K
        ecfp_dataset = [d.ecfp for d in dataset]
        ecfp_dataset = torch.stack(ecfp_dataset)
        self.labels, medoid_indices = fuzzy_k_medoids_from_distance(ecfp_dataset, k=K)


    def split(self, target, smiles = None, return_average_purity=False):
        assert target.shape[0] == self.labels.shape[0]
        train_val_split = []
        average_purity = 0
        for i in range(self.K):
            train, val, purity = sample_from_cluster_with_impurity_metric(self.labels, target, cluster_id=i,
                                                                          n_samples=self.n_val_set)

            train_val_split.append((train, val))
            average_purity += purity
        if return_average_purity:
            return train_val_split, average_purity
        return train_val_split