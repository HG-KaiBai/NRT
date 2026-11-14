"""neural_only's self-imputed integrated Brier formula for all observations.

The distribution covers [0,infinity) via u=t/(t+s); S(infinity)=0.
No reference model, censoring-specific weighting, or true CDF enters selection.
"""
import torch

EPS = 1e-6  # identical to neural_only/mlp.py


def imputed_target(survival, survival_left, survival_right, left, right, grid):
    interval_mass = (survival_left-survival_right).clamp(min=EPS)
    target = (survival-survival_right[:,None])/interval_mass[:,None]
    target = torch.where(grid[None,:] <= left[:,None],1.,target)
    target = torch.where(torch.isfinite(right[:,None]) & (grid[None,:] >= right[:,None]),0.,target)
    point = torch.isfinite(left) & torch.isfinite(right) & torch.isclose(left,right)
    return torch.where(point[:,None],(left[:,None]>grid[None,:]).to(left),target).clamp(0,1)


def expected_brier(survival, target, grid):
    if survival.shape != target.shape or survival.ndim != 2:
        raise ValueError("survival and target must be matching matrices")
    if grid.ndim != 1 or len(grid) != survival.shape[1] or len(grid) < 2:
        raise ValueError("invalid Brier integration grid")
    if not bool(torch.isfinite(grid).all()) or not bool((grid[1:] > grid[:-1]).all()):
        raise ValueError("Brier grid must be finite and strictly increasing")
    return torch.trapezoid(target-2*target*survival+survival.square(),grid.to(survival),dim=-1).mean()


@torch.no_grad()
def score_brier(model, observed, grid, batch_size=128):
    """Mean per-subject integrals; retains neural_only's EPS/isclose behavior."""
    model.eval()
    x,left,right = (v.to(model.time_scale) for v in observed)
    grid = grid.to(model.time_scale)
    transformed = grid/(grid+model.time_scale)
    total = model.time_scale.new_zeros(())
    for start in range(0,len(x),batch_size):
        stop = start+batch_size
        state = model(x[start:stop])
        survival = model.tree.survival(state,transformed).clamp(0,1)
        L,R = left[start:stop],right[start:stop]
        safe_R = torch.where(torch.isfinite(R),R,0.)
        bounds = torch.stack((L/(L+model.time_scale),
            torch.where(torch.isfinite(R),safe_R/(safe_R+model.time_scale),1.)),dim=1)
        fractions = ((state.upper[:,None,:]-bounds[:,:,None])/state.width[:,None,:]).clamp(0,1)
        survival_bounds = (fractions*state.log_mass.exp()[:,None,:]).sum(-1).clamp(0,1)
        target = imputed_target(survival,*survival_bounds.unbind(1),L,R,grid)
        total += expected_brier(survival,target,grid)*len(L)
    return float(total/len(x))
