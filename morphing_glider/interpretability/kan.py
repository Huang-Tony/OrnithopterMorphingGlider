"""Kolmogorov-Arnold Networks (KAN): BSplineBasis, KANLayer, KANPolicyNetwork."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from morphing_glider.config import DX_RANGE, DY_RANGE, DZ_RANGE
from morphing_glider.environment.observation import OBS_IDX, OBS_DIM


class BSplineBasis(torch.nn.Module):
    """B-spline basis functions for KAN layers.

    Computes B-spline basis values using the Cox-de Boor recursion.
    Uses a uniform extended knot vector so that all basis functions
    have support within x_range.

    Args:
        n_bases: Number of basis functions.
        degree: B-spline degree (default 3 = cubic).
        x_range: Input domain (min, max).

    References:
        [LIU_2024] KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.
        [DE_BOOR_1978] A Practical Guide to Splines.
    """
    def __init__(self, n_bases: int = 8, degree: int = 3,
                 x_range: Tuple[float, float] = (-3.0, 3.0)):
        super().__init__()
        self.degree = degree
        self.n_bases = n_bases
        h = (x_range[1] - x_range[0]) / max(n_bases - degree, 1)
        knots = torch.linspace(
            float(x_range[0]) - degree * h,
            float(x_range[1]) + degree * h,
            n_bases + degree + 1)
        self.register_buffer('knots', knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all B-spline basis functions at x.

        Args:
            x: Input tensor of shape (...,).

        Returns:
            Tensor of shape (..., n_bases) with basis values.
        """
        x = x.unsqueeze(-1)
        t = self.knots

        bases = ((x >= t[:-1]) & (x < t[1:])).float()

        for d in range(1, self.degree + 1):
            n_new = bases.shape[-1] - 1

            d1 = t[d:d + n_new] - t[:n_new]
            d2 = t[d + 1:d + 1 + n_new] - t[1:1 + n_new]

            a1 = (x - t[:n_new]) / (d1 + 1e-10)
            a2 = (t[d + 1:d + 1 + n_new] - x) / (d2 + 1e-10)

            bases = a1 * bases[..., :-1] + a2 * bases[..., 1:]

        return bases


class KANLayer(torch.nn.Module):
    """Single Kolmogorov-Arnold Network layer.

    Each edge (i,j) has a learnable univariate function parameterized
    as a linear combination of B-spline basis functions, plus a residual
    SiLU connection for out-of-support robustness:

        output_j = sum_i [ c_{ij} . B(x_i) + w_{ij} * silu(x_i) ] + bias_j

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        n_bases: Number of B-spline basis functions per edge.
        degree: B-spline polynomial degree.

    References:
        [LIU_2024] KAN: Kolmogorov-Arnold Networks.
        [KOLMOGOROV_1957] On the representation of continuous functions.
    """
    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 8, degree: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_bases = n_bases

        self.basis = BSplineBasis(n_bases=n_bases, degree=degree, x_range=(-3.0, 3.0))

        self.coeff = torch.nn.Parameter(
            torch.randn(out_dim, in_dim, n_bases) * (1.0 / math.sqrt(in_dim * n_bases)))

        self.residual_weight = torch.nn.Parameter(
            torch.randn(out_dim, in_dim) * (1.0 / math.sqrt(in_dim)))

        self.bias = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, in_dim).

        Returns:
            Output tensor (batch, out_dim).
        """
        basis_vals = self.basis(x)

        spline_out = torch.einsum('bin,oin->bo', basis_vals, self.coeff)

        silu_x = torch.nn.functional.silu(x)
        residual_out = torch.nn.functional.linear(silu_x, self.residual_weight)

        return spline_out + residual_out + self.bias

    def get_symbolic_approximation(self, input_names: Optional[List[str]] = None,
                                     polynomial_degree: int = 3) -> List[str]:
        """Extract approximate symbolic expressions for each output.

        For each output dimension j, fits a polynomial to each learned
        univariate function f_{ij}(x_i) and reports the dominant terms.

        Args:
            input_names: Names for input features.
            polynomial_degree: Degree for polynomial fit.

        Returns:
            List of symbolic expression strings (one per output dim).
        """
        if input_names is None:
            input_names = [f"x{i}" for i in range(self.in_dim)]

        expressions = []
        device = self.coeff.device
        x_test = torch.linspace(-2.0, 2.0, 100, device=device)
        basis_test = self.basis(x_test)

        for j in range(self.out_dim):
            terms = []
            for i in range(min(self.in_dim, len(input_names))):
                coeff_ij = self.coeff[j, i, :]
                y_spline = (basis_test * coeff_ij.unsqueeze(0)).sum(-1)

                w_ij = float(self.residual_weight[j, i].item())
                y_residual = w_ij * torch.nn.functional.silu(x_test)
                y_total = (y_spline + y_residual).detach().cpu().numpy()
                x_np = x_test.detach().cpu().numpy()

                poly_coeffs = np.polyfit(x_np, y_total, polynomial_degree)

                name = input_names[i]
                sig_terms = []
                for deg_idx, c in enumerate(poly_coeffs):
                    power = polynomial_degree - deg_idx
                    if abs(c) < 1e-4:
                        continue
                    if power == 0:
                        sig_terms.append(f"{c:+.4f}")
                    elif power == 1:
                        sig_terms.append(f"{c:+.4f}*{name}")
                    else:
                        sig_terms.append(f"{c:+.4f}*{name}^{power}")

                if sig_terms:
                    terms.append("(" + " ".join(sig_terms) + ")")

            expr = f"y{j} = {' + '.join(terms[:10]) if terms else '0'}"
            if len(terms) > 10:
                expr += f" + ... ({len(terms)} total input terms)"
            if abs(self.bias[j].item()) > 1e-4:
                expr += f" {self.bias[j].item():+.4f}"
            expressions.append(expr)

        return expressions


class KANPolicyNetwork(torch.nn.Module):
    """Kolmogorov-Arnold Network as an interpretable RL policy.

    Architecture: obs -> LayerNorm -> KAN1 -> KAN2 -> tanh -> action

    After training via DAgger or behavioral cloning, call
    get_symbolic_equations() to extract human-readable control laws.

    Args:
        obs_dim: Observation dimension (41).
        action_dim: Action dimension (6).
        hidden_dim: Hidden layer width.
        n_bases: B-spline basis count per edge.

    References:
        [LIU_2024] KAN: Kolmogorov-Arnold Networks.
    """
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = 6,
                 hidden_dim: int = 32, n_bases: int = 8):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.input_norm = torch.nn.LayerNorm(obs_dim)
        self.kan1 = KANLayer(obs_dim, hidden_dim, n_bases=n_bases)
        self.kan2 = KANLayer(hidden_dim, action_dim, n_bases=n_bases)
        self._action_scale = np.array(
            [DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1],
             DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1]], dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: obs -> normalized -> KAN1 -> KAN2 -> tanh.

        Args:
            x: Observation tensor (batch, obs_dim).

        Returns:
            Action tensor (batch, action_dim) in [-1, 1].
        """
        x = self.input_norm(x)
        h = self.kan1(x)
        return torch.tanh(self.kan2(h))

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """SB3-compatible predict interface.

        Args:
            obs: Observation array (obs_dim,).

        Returns:
            (action, state) tuple with action in physical units.
        """
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32).reshape(1, -1),
            device=next(self.parameters()).device)
        with torch.no_grad():
            action_norm = self.forward(obs_t).cpu().numpy().flatten()
        action = action_norm * self._action_scale
        return action.astype(np.float32), state

    def reset(self):
        pass

    def get_symbolic_equations(self) -> Dict[str, List[str]]:
        """Extract symbolic equations from both KAN layers.

        Prints and returns polynomial approximations of the learned
        univariate functions, providing a human-readable control law.

        Returns:
            Dict with 'layer1' and 'layer2' symbolic expressions,
            plus 'action_equations' mapping action names to expressions.
        """
        idx_to_name = {v: k for k, v in OBS_IDX.items()}
        input_names = [idx_to_name.get(i, f"obs_{i}") for i in range(self.obs_dim)]

        layer1_exprs = self.kan1.get_symbolic_approximation(
            input_names=input_names, polynomial_degree=3)

        hidden_names = [f"h{i}" for i in range(self.kan1.out_dim)]
        layer2_exprs = self.kan2.get_symbolic_approximation(
            input_names=hidden_names, polynomial_degree=3)

        action_names = ["p3R_dx", "p3R_dy", "p3R_dz", "p3L_dx", "p3L_dy", "p3L_dz"]

        print(f"\n{'='*80}")
        print("[KAN] Extracted Symbolic Equations")
        print(f"{'='*80}")
        print(f"\nLayer 1 (obs -> hidden[{self.kan1.out_dim}]):")
        for i, expr in enumerate(layer1_exprs[:5]):
            print(f"  {expr[:150]}")
        if len(layer1_exprs) > 5:
            print(f"  ... ({len(layer1_exprs)} total hidden units)")

        print(f"\nLayer 2 (hidden -> action[{self.action_dim}]):")
        for i, expr in enumerate(layer2_exprs):
            an = action_names[i] if i < len(action_names) else f"action_{i}"
            content = expr[len(f"y{i} = "):] if expr.startswith(f"y{i} = ") else expr
            print(f"  {an} = {content[:150]}")

        return {"layer1": layer1_exprs, "layer2": layer2_exprs,
                "action_names": action_names}
