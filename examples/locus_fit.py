"""Existing study visualization fit; no changes to model predictions."""
import hashlib
import numpy as np
from scipy.optimize import least_squares
FIT_CACHE = {}

def yld2000_equivalent(
    stress: np.ndarray, alpha: np.ndarray, exponent: int = 6
) -> np.ndarray:
    stress = np.atleast_2d(np.asarray(stress, dtype=float))
    alpha = np.asarray(alpha, dtype=float)
    if stress.shape[1] != 3 or alpha.shape != (8,):
        raise ValueError("Yld2000-2d expects [sigma11,sigma22,sigma12] and 8 alpha values")
    s11, s22, s12 = stress.T
    a1, a2, a3, a4, a5, a6, a7, a8 = alpha
    l11p, l12p = 2.0 * a1 / 3.0, -a1 / 3.0
    l21p, l22p = -a2 / 3.0, 2.0 * a2 / 3.0
    l66p = a7
    l11pp = (-2.0 * a3 + 2.0 * a4 + 8.0 * a5 - 2.0 * a6) / 9.0
    l12pp = (a3 - 4.0 * a4 - 4.0 * a5 + 4.0 * a6) / 9.0
    l21pp = (4.0 * a3 - 4.0 * a4 - 4.0 * a5 + a6) / 9.0
    l22pp = (-2.0 * a3 + 8.0 * a4 + 2.0 * a5 - 2.0 * a6) / 9.0
    l66pp = a8

    def principal_values(
        x11: np.ndarray, x22: np.ndarray, x12: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        center = 0.5 * (x11 + x22)
        half_span = np.sqrt((0.5 * (x11 - x22)) ** 2 + x12**2)
        return center + half_span, center - half_span

    x1p, x2p = principal_values(
        l11p * s11 + l12p * s22,
        l21p * s11 + l22p * s22,
        l66p * s12,
    )
    x1pp, x2pp = principal_values(
        l11pp * s11 + l12pp * s22,
        l21pp * s11 + l22pp * s22,
        l66pp * s12,
    )
    phi = 0.5 * (
        np.abs(x1p - x2p) ** exponent
        + np.abs(2.0 * x1pp + x2pp) ** exponent
        + np.abs(x1pp + 2.0 * x2pp) ** exponent
    )
    return np.maximum(phi, 1.0e-30) ** (1.0 / exponent)

def fit_yld2000(points: np.ndarray, exponent: int = 6) -> dict[str, object]:
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    key = hashlib.sha256(np.round(points, 10).tobytes()).hexdigest()
    if key in FIT_CACHE:
        return FIT_CACHE[key]
    radii = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(radii) & (radii > 1.0e-9)
    if int(valid.sum()) < 8:
        raise ValueError("At least eight finite stress points are required for Yld2000-2d")
    directions = points[valid] / radii[valid, None]
    radii = radii[valid]
    sigma0 = float(np.median(radii))

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float]:
        alpha = np.ones(8, dtype=float)
        alpha[1:6] = np.exp(parameters[:5])
        return alpha, float(np.exp(parameters[5]))

    def predicted_radius(parameters: np.ndarray) -> np.ndarray:
        alpha, sigma_ref = unpack(parameters)
        equivalent = yld2000_equivalent(directions, alpha, exponent=exponent)
        return sigma_ref / np.maximum(equivalent, 1.0e-30)

    lower = np.r_[np.log(np.full(5, 0.10)), np.log(40.0)]
    upper = np.r_[np.log(np.full(5, 4.00)), np.log(800.0)]
    initial = np.r_[np.zeros(5), np.log(sigma0)]
    digest = hashlib.sha256(points.tobytes()).digest()
    random = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    starts = [initial]
    for _ in range(7):
        start = initial.copy()
        start[:5] += random.normal(0.0, 0.32, size=5)
        start[5] += random.normal(0.0, 0.08)
        starts.append(np.clip(start, lower + 1.0e-8, upper - 1.0e-8))

    candidates = []
    for start in starts:
        result = least_squares(
            lambda parameters: np.log(predicted_radius(parameters) / radii),
            x0=start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.025,
            max_nfev=4000,
        )
        fitted = predicted_radius(result.x)
        score = float(np.sqrt(np.mean(np.log(fitted / radii) ** 2)))
        candidates.append((score, result, fitted))
    _, result, fitted = min(candidates, key=lambda item: item[0])
    alpha, sigma_ref = unpack(result.x)
    fit = {
        "model": "Yld2000-2d",
        "exponent": int(exponent),
        "alpha": alpha,
        "sigma_ref": sigma_ref,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "radial_rmse_pct": float(
            100.0 * np.sqrt(np.mean(((fitted - radii) / radii) ** 2))
        ),
        "radial_mae_mpa": float(np.mean(np.abs(fitted - radii))),
    }
    FIT_CACHE[key] = fit
    return fit

def yld2000_curve(
    fit: dict[str, object], n_points: int = 721
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    directions = np.column_stack(
        [np.cos(theta), np.sin(theta), np.zeros_like(theta)]
    )
    radius = float(fit["sigma_ref"]) / yld2000_equivalent(
        directions,
        np.asarray(fit["alpha"], dtype=float),
        exponent=int(fit["exponent"]),
    )
    return theta, radius * directions[:, 0], radius * directions[:, 1]

