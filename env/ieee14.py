"""
env/ieee14.py
-------------
IEEE 14-Bus System – Topology Constants (100 MVA base).

Source: UW Power Systems Test Case Archive
        https://labs.ece.uw.edu/pstca/pf14/ieee14.zip

All reactances in per-unit.  Ratings in MW.
Bus numbering: 1-indexed throughout; internally converted to 0-indexed.
"""

from dataclasses import dataclass
from typing import Tuple

MVA_BASE:   float = 100.0   # System base (MVA)
N_BUSES:    int   = 14
N_LINES:    int   = 20
SLACK_BUS:  int   = 1       # 1-indexed slack / reference bus
SLACK_IDX:  int   = 0       # 0-indexed

# ── Line data ────────────────────────────────────────────────────────────────
# (from_bus, to_bus, r_pu, x_pu, b_pu, thermal_rating_mw)
LINE_DATA: Tuple = (
    (1,  2,  0.01938, 0.05917, 0.0528, 160),
    (1,  5,  0.05403, 0.22304, 0.0492, 130),
    (2,  3,  0.04699, 0.19797, 0.0438, 130),
    (2,  4,  0.05811, 0.17632, 0.0340,  65),
    (2,  5,  0.05695, 0.17388, 0.0346, 130),
    (3,  4,  0.06701, 0.17103, 0.0128,  65),
    (4,  5,  0.01335, 0.04211, 0.0000,  90),
    (4,  7,  0.00000, 0.20912, 0.0000,  70),   # transformer
    (4,  9,  0.00000, 0.55618, 0.0000,  32),   # transformer
    (5,  6,  0.00000, 0.25202, 0.0000,  65),   # transformer
    (6,  11, 0.09498, 0.19890, 0.0000,  45),
    (6,  12, 0.12291, 0.25581, 0.0000,  32),
    (6,  13, 0.06615, 0.13027, 0.0000,  65),
    (7,  8,  0.00000, 0.17615, 0.0000,  32),
    (7,  9,  0.00000, 0.11001, 0.0000,  32),
    (9,  10, 0.03181, 0.08450, 0.0000,  32),
    (9,  14, 0.12711, 0.27038, 0.0000,  32),
    (10, 11, 0.08205, 0.19207, 0.0000,  32),
    (12, 13, 0.22092, 0.19988, 0.0000,  16),
    (13, 14, 0.17093, 0.34802, 0.0000,  16),
)

LINE_RATINGS_MW = tuple(row[5] for row in LINE_DATA)   # thermal limits
LINE_FROM       = tuple(row[0] - 1 for row in LINE_DATA)  # 0-indexed from-bus
LINE_TO         = tuple(row[1] - 1 for row in LINE_DATA)  # 0-indexed to-bus
LINE_X_PU       = tuple(row[3] for row in LINE_DATA)      # reactances (pu)

# ── Generator data ───────────────────────────────────────────────────────────
# (gen_id, bus_1idx, fuel, p_min_mw, p_max_mw, cost_a $/MW²h, cost_b $/MWh, ramp_mw_step)
#   6 generators across 5 fuel types: Coal×1, Gas×2, Wind×1, Hydro×1, Solar×1
GENERATOR_DATA: Tuple = (
    (0,  1, "coal",    20.0, 100.0, 0.0200, 20.0, 30.0),
    (1,  2, "gas",     10.0,  80.0, 0.0150, 25.0, 40.0),
    (2,  3, "gas",      5.0,  50.0, 0.0175, 28.0, 45.0),
    (3,  6, "wind",     0.0,  50.0, 0.0000,  3.0, 50.0),   # intermittent
    (4,  8, "hydro",    5.0,  40.0, 0.0010, 10.0, 35.0),
    (5, 12, "solar",    0.0,  30.0, 0.0000,  3.0, 30.0),   # intermittent
)

N_GENS = len(GENERATOR_DATA)

# Derived quick-access arrays (0-indexed bus)
GEN_BUS_IDX   = tuple(row[1] - 1 for row in GENERATOR_DATA)
GEN_FUEL      = tuple(row[2]     for row in GENERATOR_DATA)
GEN_P_MIN     = tuple(row[3]     for row in GENERATOR_DATA)
GEN_P_MAX     = tuple(row[4]     for row in GENERATOR_DATA)
GEN_COST_A    = tuple(row[5]     for row in GENERATOR_DATA)
GEN_COST_B    = tuple(row[6]     for row in GENERATOR_DATA)
GEN_RAMP      = tuple(row[7]     for row in GENERATOR_DATA)
RENEWABLE_IDX = tuple(i for i, f in enumerate(GEN_FUEL) if f in ("wind", "solar"))

# ── Nominal bus loads (MW) ────────────────────────────────────────────────────
# Buses without load are 0.  Total = 259.0 MW.
BASE_LOAD_MW = (
    0.0,    # Bus 1  – generation only
    21.7,   # Bus 2
    94.2,   # Bus 3
    47.8,   # Bus 4
    7.6,    # Bus 5
    11.2,   # Bus 6
    0.0,    # Bus 7  – transit
    0.0,    # Bus 8  – generation only
    29.5,   # Bus 9
    9.0,    # Bus 10
    3.5,    # Bus 11
    6.1,    # Bus 12
    13.5,   # Bus 13
    14.9,   # Bus 14
)
TOTAL_BASE_LOAD_MW: float = sum(BASE_LOAD_MW)  # 259.0 MW

# ── Bus XY coordinates (normalised [0,1]) for plotly visualisation ──────────
BUS_XY = (
    (0.22, 0.97),  # Bus 1
    (0.49, 0.97),  # Bus 2
    (0.88, 0.85),  # Bus 3
    (0.61, 0.82),  # Bus 4
    (0.36, 0.82),  # Bus 5
    (0.88, 0.62),  # Bus 6
    (0.74, 0.62),  # Bus 7
    (0.88, 0.45),  # Bus 8
    (0.74, 0.45),  # Bus 9
    (0.74, 0.28),  # Bus 10
    (0.88, 0.28),  # Bus 11
    (1.00, 0.15),  # Bus 12
    (0.88, 0.10),  # Bus 13
    (0.74, 0.10),  # Bus 14
)

# Maximum quadratic cost (all gens at P_max) – used to normalise cost reward
_MAX_COST: float = sum(
    GEN_COST_A[i] * GEN_P_MAX[i] ** 2 + GEN_COST_B[i] * GEN_P_MAX[i]
    for i in range(N_GENS)
)
MAX_COST: float = _MAX_COST if _MAX_COST > 0 else 1.0
