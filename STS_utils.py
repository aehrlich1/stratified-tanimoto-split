import torch
from typing import Callable, Optional, Tuple, Dict

"""
FCMdd (Fuzzy C-Medoids) in PyTorch with a pluggable distance function.

Objective (for fuzzifier m>1):
    J(U, M) = sum_i sum_k u_{ik}^m * d(x_i, x_{m_k})
subject to sum_k u_{ik} = 1, u_{ik} >= 0, and medoids x_{m_k} are data points.

Updates:
- Memberships (for each i with all d_{ik} > 0):
      u_{ik} = 1 / sum_h (d_{ik} / d_{ih})**(1/(m-1))
  If any d_{ik} == 0 for sample i, set u_{ik}=1 for the zero-distance medoid(s) and 0 otherwise.
- Medoids: for each cluster k, pick index j minimizing  sum_i u_{ik}^m * D[i, j].

Complexity: O(N^2 K) per outer iteration if we recompute full medoid search. Suitable for N up to a few thousands.

Supports:
- Custom distance function: distance_fn(X) -> (N,N) matrix of pairwise dissimilarities (symmetric, nonneg., zero diag).
- Or pass a precomputed distance matrix D.
- Arbitrary (even non-metric) dissimilarities.

Returns:
- memberships: (N,K) tensor
- medoid_indices: (K,) long tensor of chosen medoids
- labels: (N,) long tensor via argmax
- history: dict with trajectory and final objective
"""

def _pairwise_distance(
    X: torch.Tensor,
    distance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
) -> torch.Tensor:
    if distance_fn is None:
        # Default: Euclidean distances
        # Efficiently compute squared euclidean, then sqrt
        X2 = (X**2).sum(dim=1, keepdim=True)
        D2 = X2 + X2.T - 2.0 * (X @ X.T)
        D2.clamp_(min=0.0)
        D = torch.sqrt(D2 + 1e-12)
    else:
        D = distance_fn(X)
        if not torch.is_tensor(D):
            raise TypeError("distance_fn must return a torch.Tensor")
    # sanitize
    D = D.float()
    D = 0.5 * (D + D.T)
    D = torch.clamp(D, min=0.0)
    D.fill_diagonal_(0.0)
    return D


def _init_medoids(D: torch.Tensor, K: int, seed: Optional[int] = None) -> torch.Tensor:
    N = D.shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    # k-medoids++ style: pick first at random, then sample proportional to distance to nearest chosen
    medoids = []
    first = torch.randint(0, N, (1,)).item()
    medoids.append(first)
    for _ in range(1, K):
        dist_to_nearest = torch.min(D[:, medoids], dim=1).values
        probs = dist_to_nearest / (dist_to_nearest.sum() + 1e-12)
        idx = torch.multinomial(probs, 1).item()
        # avoid duplicates via fallback
        tries = 0
        while idx in medoids and tries < 10:
            idx = torch.randint(0, N, (1,)).item()
            tries += 1
        medoids.append(idx)
    return torch.tensor(medoids, dtype=torch.long, device=D.device)


def _update_memberships(DKm: torch.Tensor, m: float) -> torch.Tensor:
    """Update memberships given distances to current medoids.
    DKm: (N,K) distances d_{ik} to medoids, m>1.
    """
    eps = 1e-12
    N, K = DKm.shape
    U = torch.zeros(N, K, device=DKm.device, dtype=DKm.dtype)
    # Handle zeros: for each i, find medoids at zero distance
    zero_mask = DKm <= eps
    any_zero = zero_mask.any(dim=1)
    # For rows with zeros, set equal membership among zero-distance clusters
    if any_zero.any():
        idxs = torch.nonzero(any_zero, as_tuple=False).squeeze(1)
        for i in idxs.tolist():
            z = zero_mask[i]
            cnt = z.sum()
            U[i, z] = 1.0 / cnt
    # For the rest, use standard formula
    nonzero_rows = (~any_zero).nonzero(as_tuple=False).squeeze(1)
    if nonzero_rows.numel() > 0:
        Dnz = DKm[nonzero_rows] + eps
        p = 1.0 / (m - 1.0)
        # u_{ik} = 1 / sum_h (d_{ik}/d_{ih})^{p}
        # Implement via: denom_i_k = sum_h (d_{ik}^p / d_{ih}^p) = d_{ik}^p * sum_h (1/d_{ih}^p)
        Dp = Dnz.pow(p)
        inv_sum = (1.0 / Dp).sum(dim=1, keepdim=True)  # (n_i,1)
        U_nz = 1.0 / (Dp * inv_sum)
        # normalize for numerical safety
        U_nz = U_nz / (U_nz.sum(dim=1, keepdim=True) + eps)
        U[nonzero_rows] = U_nz
    return U


def _objective(D: torch.Tensor, U: torch.Tensor, m: float, medoids: torch.Tensor) -> torch.Tensor:
    DKm = D[:, medoids]  # (N,K)
    return (U.pow(m) * DKm).sum()


def fcmdd(
    X: Optional[torch.Tensor] = None,
    n_clusters: Optional[int] = None,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: Optional[int] = None,
    distance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    D: Optional[torch.Tensor] = None,
    medoids_init: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    diversity_weight: float = 0.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    FCMdd with custom distance function or precomputed D.

    Parameters
----------
X : (N,d) tensor, optional if D is provided
n_clusters : int, required if medoids_init is None
m : float, fuzzifier (>1)
max_iter : int
tol : float, relative tolerance on objective or medoid change
seed : int, for reproducible init
distance_fn : callable(X) -> D (N,N), ignored if D provided
D : (N,N) precomputed dissimilarity matrix (symmetric
    """
    assert m > 1.0, "Fuzzifier m must be > 1"

    if D is None:
        assert X is not None, "Provide X or a precomputed D"
        if device is None:
            device = X.device
        D = _pairwise_distance(X.to(device), distance_fn)
    else:
        if device is None:
            device = D.device
        D = 0.5 * (D + D.T)
        D = torch.clamp(D.to(device).float(), min=0.0)
        D.fill_diagonal_(0.0)

    N = D.shape[0]

    if medoids_init is None:
        assert n_clusters is not None and n_clusters >= 2
        medoids = _init_medoids(D, n_clusters, seed=seed)
    else:
        medoids = medoids_init.to(device).long()
        n_clusters = medoids.numel()

    history_loss = []
    prev_obj = None

    for it in range(max_iter):
        DKm = D[:, medoids]  # (N,K)
        # Update memberships
        U = _update_memberships(DKm, m)
        # Update medoids: for each k, choose j minimizing sum_i u_{ik}^m * D[i, j]
        W = U.pow(m)  # (N,K)
        # Precompute cluster weights sums per candidate j for each k via matrix mul
        # cost_k(j) = (w_k)^T * D[:, j]; vectorized: costs = D.T @ w_k  => (N,)
        new_medoids = medoids.clone()
        for k in range(n_clusters):
            wk = W[:, k]  # (N,)
            # Compute costs for all candidates j
            costs = D.T @ wk  # (N,)
            jmin = torch.argmin(costs)
            new_medoids[k] = jmin
        # Compute objective and stopping criteria
        obj = (W * D[:, new_medoids]).sum()
        history_loss.append(obj.detach())

        medoid_change = (new_medoids != medoids).any().item()
        rel_improve = None
        if prev_obj is not None:
            rel_improve = ((prev_obj - obj).abs() / (prev_obj.abs() + 1e-12)).item()
        if verbose and (it % 1 == 0 or it == max_iter - 1):
            print(f"iter {it:4d}  J={obj.item():.6f}  medoid_change={medoid_change}  rel_imp={rel_improve}")

        if not medoid_change and (rel_improve is not None and rel_improve < tol):
            medoids = new_medoids
            break
        medoids = new_medoids
        prev_obj = obj.detach()

    labels = U.argmax(dim=1)
    history = {
        'loss': torch.stack(history_loss),
        'final_objective': history_loss[-1].detach(),
        'iterations': torch.tensor(len(history_loss))
    }
    return U, medoids, labels, history


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Build a toy dataset in 2D
    n1, n2, n3 = 30, 30, 30
    C1 = torch.randn(n1, 2) * 0.3 + torch.tensor([0.0, 0.0])
    C2 = torch.randn(n2, 2) * 0.3 + torch.tensor([3.0, 0.0])
    C3 = torch.randn(n3, 2) * 0.3 + torch.tensor([1.5, 2.6])
    X = torch.cat([C1, C2, C3], dim=0)

    # Example custom distance: cosine dissimilarity (1 - cosine similarity)
    def cosine_dissimilarity(X: torch.Tensor) -> torch.Tensor:
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        S = Xn @ Xn.T
        D = 1.0 - S
        D = torch.clamp(D, min=0.0)
        D.fill_diagonal_(0.0)
        return D

    U, medoids, labels, hist = fcmdd(
        X=X,
        n_clusters=3,
        m=2.0,
        max_iter=200,
        tol=1e-7,
        seed=42,
        distance_fn=cosine_dissimilarity,
        verbose=True,
    )

    print("Medoid indices:", medoids.tolist())
    print("Label counts:", torch.bincount(labels).tolist())
    print("Final objective:", float(hist['final_objective']))
