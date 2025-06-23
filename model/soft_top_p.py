import torch

def hard_top_p(x, p):
    """Hard top-p WTA — selects smallest number of top values summing to ≥ p."""
    assert 0 < p <= 1.0, "p must be in (0, 1]"
    original_shape = x.shape
    x_flat = x.view(-1, x.shape[-1])  # flatten for batch processing

    out = torch.zeros_like(x_flat)
    for i in range(x_flat.size(0)):
        row = x_flat[i]
        sorted_vals, sorted_indices = torch.sort(row, descending=True)
        cumulative = torch.cumsum(sorted_vals, dim=0)

        # Find how many elements needed to reach threshold p
        k = (cumulative >= p).nonzero(as_tuple=False)[0].item() + 1
        selected_indices = sorted_indices[:k]
        out[i].scatter_(0, selected_indices, 1.0)

    return out.view(original_shape)


def soft_top_p(x, p, temperature=1.0, eps=None):
    """Soft top-p WTA — uses sigmoid weighting around the threshold value."""
    if eps is None:
        eps = x.new_tensor(1e-12)
    assert 0 < p <= 1.0, "p must be in (0, 1]"
    
    original_shape = x.shape
    x_flat = x.view(-1, x.shape[-1])  # process in 2D

    out = torch.zeros_like(x_flat)
    for i in range(x_flat.size(0)):
        row = x_flat[i]
        sorted_vals, sorted_indices = torch.sort(row, descending=True)
        cumulative = torch.cumsum(sorted_vals, dim=0)

        # Determine k such that cumulative sum >= p
        k = (cumulative >= p).nonzero(as_tuple=False)[0].item() + 1
        threshold = ((sorted_vals[k-1] + sorted_vals[min(k, len(sorted_vals)-1)]) / 2)

        diff = (row - threshold) / (temperature * (sorted_vals[k-1] - sorted_vals[k]) + eps)
        weights = torch.sigmoid(diff)

        # Normalize to sum to ~k (optional)
        weights = weights * (k / weights.sum())

        out[i] = weights

    return out.view(original_shape)


def hard_top_p_with_soft_gradient(x, p, temperature=1.0, eps=None):
    """Hard top-p WTA with soft gradients (straight-through)."""
    hard = hard_top_p(x, p)
    soft = soft_top_p(x, p, temperature=temperature, eps=eps)
    return hard - soft.detach() + soft
