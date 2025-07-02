import numpy as np
import torch
from scipy.interpolate import BSpline

def batch_bspline(ctrl_pts, t_values=None, degree=3):
    """
    Perform B-spline interpolation in batch.

    Args:
        ctrl_pts (torch.Tensor): [B, N, D] control points
        num_points (int): number of interpolated points
        degree (int): spline degree

    Returns:
        torch.Tensor: interpolated trajectory [B, num_points, D]
    """
    B, N, D = ctrl_pts.shape
    device = ctrl_pts.device

    out = []
    for b in range(B):
        cp = ctrl_pts[b].detach().cpu().numpy()
        n = len(cp)
        k = min(degree, n - 1)
        knots = np.concatenate((
            np.zeros(k),
            np.linspace(0, 1, n - k + 1),
            np.ones(k)
        ))
        spline = BSpline(knots, cp, k)
        interp = torch.tensor(spline(t_values.cpu().numpy()), device=device).float()
        out.append(interp)
    return torch.stack(out, dim=0)