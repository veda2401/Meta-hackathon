"""
env/dc_power_flow.py
--------------------
DC Power Flow Solver using the B-matrix (susceptance matrix) approach.

Physics
-------
DC approximation assumptions:
  1. |V| = 1.0 pu at all buses (flat voltage profile)
  2. Line resistance r ≪ reactance x  →  use susceptance b = 1/x only
  3. sin(θ_i - θ_j) ≈ θ_i - θ_j  (small angle)

Network equations:
  P_i = Σ_j  B_ij · (θ_i - θ_j)      (nodal power balance)
  →  P_bus = B_bus · θ               (in matrix form, pu)

Reduced system (remove slack row/col, θ_slack = 0):
  B_red · θ_red = P_inj_red / MVA_BASE
  →  θ_red = B_red⁻¹ · P_inj_red

Line flow (positive = from→to direction):
  f_ij = b_ij · (θ_i - θ_j) · MVA_BASE   [MW]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from env import ieee14


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class DCResult:
    angles_rad: np.ndarray          # shape (N_BUSES,), θ_slack = 0
    angles_deg: np.ndarray          # degrees
    line_flows_mw: np.ndarray       # shape (N_LINES,), signed MW (+ = from→to)
    converged: bool                 # False if B_red is singular (island)
    slack_injection_mw: float       # MW picked up / shed by slack bus
    max_angle_diff_deg: float       # worst bus-to-bus angle difference (deg)

    @property
    def overloaded_lines(self) -> List[int]:
        """0-indexed list of lines where |flow| > thermal rating."""
        return [
            i for i, (f, r) in enumerate(
                zip(self.line_flows_mw, ieee14.LINE_RATINGS_MW)
            )
            if abs(f) > r
        ]

    @property
    def loading_fractions(self) -> np.ndarray:
        """Per-line |flow| / rating (0-indexed)."""
        ratings = np.array(ieee14.LINE_RATINGS_MW, dtype=float)
        return np.abs(self.line_flows_mw) / ratings


# ── B-matrix builder ─────────────────────────────────────────────────────────

def build_b_matrix(line_status: List[bool]) -> np.ndarray:
    """
    Construct the full N_BUSES × N_BUSES nodal susceptance matrix.

    Parameters
    ----------
    line_status : list[bool]
        True = line online; False = tripped/disconnected.

    Returns
    -------
    B : ndarray, shape (N_BUSES, N_BUSES)
    """
    n = ieee14.N_BUSES
    B = np.zeros((n, n), dtype=float)

    for k, (online, x) in enumerate(zip(line_status, ieee14.LINE_X_PU)):
        if not online or abs(x) < 1e-12:
            continue
        b = 1.0 / x
        fi = ieee14.LINE_FROM[k]
        ti = ieee14.LINE_TO[k]
        B[fi, fi] += b
        B[ti, ti] += b
        B[fi, ti] -= b
        B[ti, fi] -= b

    return B


# ── DC solver ────────────────────────────────────────────────────────────────

def solve_dc_power_flow(
    p_injection_mw: np.ndarray,
    line_status: List[bool],
) -> DCResult:
    """
    Run one DC power-flow iteration.

    Parameters
    ----------
    p_injection_mw : ndarray, shape (N_BUSES,)
        Net injection at each bus = P_gen - P_load in MW.
        Slack bus value is ignored; it floats to balance the network.
    line_status : list[bool]
        Per-line online/tripped flags.

    Returns
    -------
    DCResult
    """
    n     = ieee14.N_BUSES
    slack = ieee14.SLACK_IDX

    # Build full B matrix
    B = build_b_matrix(line_status)

    # Non-slack bus indices
    pq_idx = [i for i in range(n) if i != slack]

    # Reduced system: B_red · θ_red = P_red / MVA_BASE
    B_red = B[np.ix_(pq_idx, pq_idx)]
    P_red = p_injection_mw[pq_idx] / ieee14.MVA_BASE

    angles = np.zeros(n, dtype=float)
    converged = True

    try:
        # Dense direct solve (N=13 — very fast)
        theta_red = np.linalg.solve(B_red, P_red)
        for k, idx in enumerate(pq_idx):
            angles[idx] = theta_red[k]
    except np.linalg.LinAlgError:
        # Singular B (island / disconnected network)
        converged = False

    # Compute line flows
    flows = np.zeros(ieee14.N_LINES, dtype=float)
    for k, (online, x) in enumerate(zip(line_status, ieee14.LINE_X_PU)):
        if not online or abs(x) < 1e-12:
            flows[k] = 0.0
            continue
        fi = ieee14.LINE_FROM[k]
        ti = ieee14.LINE_TO[k]
        b  = 1.0 / x
        flows[k] = b * (angles[fi] - angles[ti]) * ieee14.MVA_BASE

    # Slack bus picks up the residual to enforce global balance
    slack_inj = float(p_injection_mw[slack] - (
        -sum(flows[k] for k in range(ieee14.N_LINES)
             if ieee14.LINE_TO[k] == slack and line_status[k]) +
        sum(flows[k] for k in range(ieee14.N_LINES)
            if ieee14.LINE_FROM[k] == slack and line_status[k])
    ))

    angles_deg = np.degrees(angles)
    max_diff = float(np.max(np.abs(
        angles_deg[list(ieee14.LINE_FROM)] - angles_deg[list(ieee14.LINE_TO)]
    ))) if any(line_status) else 0.0

    return DCResult(
        angles_rad=angles,
        angles_deg=angles_deg,
        line_flows_mw=flows,
        converged=converged,
        slack_injection_mw=slack_inj,
        max_angle_diff_deg=max_diff,
    )
