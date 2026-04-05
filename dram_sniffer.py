#!/usr/bin/env python3
“””
DRAM-SNIFFER v3.0 - DDR Noise, Irregularities and Failure Finder
with External metrics using Reinforcement learning

Automated Generation of Correctable and Uncorrectable Error-Inducing
Test Cases on Real DDR Hardware Using Reinforcement Learning

This implementation covers:

- Phase 0: Simulation-based development with probabilistic error model
- Phase 1: Real hardware training via GSAT/stressapptest + EDAC
- Phase 2: Production validation across vendors

Key Features:

- GSAT-aligned action space mapping directly to stressapptest commands
- EDAC/BMC-inspired state representation
- Dual-path error discovery (CE escalation + sudden UE)
- PPO training with robust hyperparameters from SSD SNIFFER
- Comprehensive evaluation and TC library generation
  “””

from **future** import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
Any,
Callable,
Dict,
Iterator,
List,
Optional,
Sequence,
Set,
Tuple,
Union,
)

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Optional imports with graceful fallback

try:
import matplotlib
matplotlib.use(“Agg”)
import matplotlib.pyplot as plt
MATPLOTLIB_AVAILABLE = True
except ImportError:
MATPLOTLIB_AVAILABLE = False
plt = None

try:
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
SB3_AVAILABLE = True
except ImportError as e:
SB3_AVAILABLE = False
SB3_IMPORT_ERROR = str(e)
PPO = None
BaseCallback = object
EvalCallback = None
DummyVecEnv = None
SubprocVecEnv = None
Monitor = None

# Configure logging

logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s | %(levelname)-8s | %(name)s | %(message)s”,
datefmt=”%Y-%m-%d %H:%M:%S”,
)
logger = logging.getLogger(“DRAM-SNIFFER”)

class NumpyJSONEncoder(json.JSONEncoder):
“”“JSON encoder that handles numpy types.”””

```
def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return super().default(obj)
```

# =============================================================================

# SECTION 1: CONFIGURATION & CONSTANTS

# =============================================================================

class ErrorType(Enum):
“”“Types of DRAM errors.”””
NONE = auto()
CE = auto()      # Correctable Error
UE = auto()      # Uncorrectable Error
CE_UE = auto()   # Both CE and UE in same step

class BackendType(Enum):
“”“Execution backend types.”””
SYNTHETIC_FAST = “synthetic_fast”       # Pure simulation, no delays
SYNTHETIC_TIMED = “synthetic_timed”     # Simulation with timing
GSAT_WRAPPER = “gsat_wrapper”           # Real stressapptest execution

class ActionGranularity(Enum):
“”“Action space granularity levels.”””
COARSE = “coarse”   # ~672 actions (pattern × region × concurrency × access × duration)
MEDIUM = “medium”   # ~2016 actions (adds block_size)
FULL = “full”       # ~6048 actions (full Cartesian)

@dataclass
class DRAMSnifferConfig:
“”“Complete configuration for DRAM-SNIFFER framework.”””

```
# =========================================================================
# GSAT Action Space Parameters (from Framework Proposal Table 1)
# =========================================================================
patterns: Tuple[str, ...] = (
    "walking_1_0",
    "checkerboard",
    "random",
    "march",
    "butterfly",
    "solid",
    "inversion",
)
regions: Tuple[int, ...] = tuple(range(8))  # 8 discretized address segments
concurrencies: Tuple[str, ...] = ("low", "medium", "high")
access_patterns: Tuple[str, ...] = ("sequential", "random", "rowhammer_like")
block_sizes: Tuple[str, ...] = ("cacheline_64B", "page_4KB", "large_2MB")
durations_s: Tuple[int, ...] = (30, 60, 120, 300)

# GSAT executor parameters
memory_sizes_mb: Tuple[int, ...] = (256, 1024, 4096, 0)  # 0 = max
copy_threads: Tuple[int, ...] = (2, 4, 8, 16)
cpu_threads: Tuple[int, ...] = (0, 4, 8)
warm_copy_options: Tuple[bool, ...] = (False, True)
max_errors_options: Tuple[int, ...] = (1, 10, -1)  # -1 = unlimited

# =========================================================================
# MDP Parameters (from Framework Proposal Section 3.B)
# =========================================================================
max_steps_per_episode: int = 64
gamma: float = 0.99  # Discount factor

# Reward coefficients: r_t = α·ΔCE + β·ΔUE + γ·coverage + exploration - penalty
alpha_ce: float = 1.0           # CE reward weight
beta_ue: float = 100.0          # UE reward weight (β >> α per proposal)
gamma_coverage: float = 2.0     # Coverage bonus weight
exploration_bonus: float = 0.5  # New action exploration bonus
time_penalty_coef: float = 0.001  # Per-second time penalty

# =========================================================================
# Probabilistic Error Model (calibrated from Google/field data)
# =========================================================================
# Base probabilities per step (calibrated from ~3.7% CE/DIMM/month)
base_ce_prob: float = 0.008
base_ue_prob: float = 0.0002
base_ue_sudden_prob: float = 0.0001

# CE escalation parameters (from Google finding: CE→UE correlation 9-400×)
ce_escalation_factor: float = 5.0
ce_ue_correlation_min: float = 9.0
ce_ue_correlation_max: float = 100.0

# Region vulnerability distribution
weak_region_fraction: float = 0.25
weak_region_vulnerability_min: float = 2.5
weak_region_vulnerability_max: float = 5.0
normal_region_vulnerability_min: float = 0.8
normal_region_vulnerability_max: float = 1.2

# Temperature effects
base_temperature_c: float = 40.0
temperature_min_c: float = 35.0
temperature_max_c: float = 85.0
temperature_error_threshold_c: float = 55.0
temperature_error_multiplier: float = 1.15

# =========================================================================
# PPO Hyperparameters (from SSD SNIFFER Table 2)
# =========================================================================
learning_rate: float = 3e-4
n_steps: int = 2048
batch_size: int = 64
n_epochs: int = 10
gae_lambda: float = 0.95
clip_range: float = 0.2
ent_coef: float = 0.01
vf_coef: float = 0.5
max_grad_norm: float = 0.5
net_arch: Tuple[int, ...] = (256, 256)

# Training settings
total_timesteps: int = 200_000
eval_freq: int = 10_000
n_eval_episodes: int = 20
save_freq: int = 25_000

# =========================================================================
# Backend Settings
# =========================================================================
backend: str = "synthetic_fast"
action_granularity: str = "coarse"

# Synthetic backend
sleep_scale: float = 0.0  # For synthetic_timed: sleep = duration_s * sleep_scale
fixed_weak_map: bool = False  # Use same vulnerability map across episodes

# GSAT backend
gsat_binary: str = "stressapptest"
use_numactl: bool = False
gsat_timeout_s: int = 600
require_edac: bool = False
edac_poll_interval_s: float = 1.0

# =========================================================================
# Evaluation Settings
# =========================================================================
eval_episodes: int = 100
deterministic_eval: bool = True

# =========================================================================
# Output Settings
# =========================================================================
output_dir: str = "runs/dram_sniffer"
save_tc_library: bool = True
save_action_history: bool = True
generate_plots: bool = True
verbose: int = 1

def to_dict(self) -> Dict[str, Any]:
    """Convert config to dictionary."""
    result = {}
    for k, v in asdict(self).items():
        if isinstance(v, tuple):
            result[k] = list(v)
        else:
            result[k] = v
    return result

@classmethod
def from_dict(cls, d: Dict[str, Any]) -> "DRAMSnifferConfig":
    """Create config from dictionary."""
    # Convert lists back to tuples for immutable fields
    converted = {}
    for k, v in d.items():
        if isinstance(v, list) and k in {
            "patterns", "regions", "concurrencies", "access_patterns",
            "block_sizes", "durations_s", "memory_sizes_mb", "copy_threads",
            "cpu_threads", "warm_copy_options", "max_errors_options", "net_arch"
        }:
            converted[k] = tuple(v)
        else:
            converted[k] = v
    return cls(**converted)

def save(self, path: Union[str, Path]) -> None:
    """Save config to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(self.to_dict(), f, indent=2)

@classmethod
def load(cls, path: Union[str, Path]) -> "DRAMSnifferConfig":
    """Load config from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return cls.from_dict(json.load(f))
```

# =============================================================================

# SECTION 2: ACTION SPACE DEFINITION

# =============================================================================

# Stress factor lookup tables (calibrated for error probability scaling)

PATTERN_STRESS = {
“walking_1_0”: 1.10,
“checkerboard”: 1.20,
“random”: 1.00,
“march”: 1.15,
“butterfly”: 1.30,  # Highest stress - targets cell coupling
“solid”: 1.05,
“inversion”: 1.18,
}

CONCURRENCY_STRESS = {
“low”: 1.0,
“medium”: 1.4,
“high”: 1.9,
}

ACCESS_STRESS = {
“sequential”: 1.0,
“random”: 1.2,
“rowhammer_like”: 1.6,  # Highest stress - targets row activation
}

BLOCK_STRESS = {
“cacheline_64B”: 1.25,
“page_4KB”: 1.0,
“large_2MB”: 0.9,
}

DURATION_STRESS = {
30: 1.0,
60: 1.2,
120: 1.5,
300: 2.0,
}

MEMORY_STRESS = {
256: 0.85,
1024: 1.0,
4096: 1.2,
0: 1.4,  # Full memory
}

COPY_THREADS_STRESS = {
2: 0.9,
4: 1.0,
8: 1.2,
16: 1.4,
}

CPU_THREADS_STRESS = {
0: 1.0,
4: 1.1,
8: 1.25,
}

MAX_ERRORS_STRESS = {
1: 1.1,   # Stop early - good for UE hunting
10: 1.0,
-1: 0.95,  # Run full duration
}

@dataclass(frozen=True)
class ActionSpec:
“”“Immutable specification for a single GSAT action.”””
action_id: int
pattern: str
region: int
concurrency: str
access_pattern: str
block_size: str
duration_s: int
memory_mb: int
copy_threads: int
cpu_threads: int
warm_copy: bool
max_errors: int
stress_score: float

```
def to_gsat_command(
    self,
    gsat_binary: str = "stressapptest",
    use_numactl: bool = False,
    log_path: Optional[str] = None,
) -> str:
    """Generate stressapptest command string."""
    parts = []
    
    # Optional NUMA binding
    if use_numactl:
        numa_node = self.region % 2
        parts.extend(["numactl", f"--membind={numa_node}"])
    
    # Base command
    parts.append(gsat_binary)
    parts.extend(["-s", str(self.duration_s)])
    parts.extend(["-M", str(self.memory_mb)])
    parts.extend(["-m", str(self.copy_threads)])
    
    # Optional CPU threads
    if self.cpu_threads > 0:
        parts.extend(["-C", str(self.cpu_threads)])
    
    # Warm copy
    if self.warm_copy:
        parts.append("-W")
    
    # Max errors
    if self.max_errors != -1:
        parts.extend(["--max_errors", str(self.max_errors)])
    
    # Log file
    log = log_path or f"/tmp/gsat_region_{self.region}.log"
    parts.extend(["-l", log])
    
    return " ".join(parts)

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary."""
    return asdict(self)
```

class ActionLibrary:
“””
Manages the action space for DRAM-SNIFFER.

```
Maps high-level semantic actions (pattern, region, concurrency, etc.)
to concrete GSAT configurations with computed stress scores.
"""

def __init__(self, cfg: DRAMSnifferConfig):
    self.cfg = cfg
    self.actions: List[ActionSpec] = []
    self._build_action_library()
    
    # Build lookup indices
    self._by_pattern: Dict[str, List[int]] = defaultdict(list)
    self._by_region: Dict[int, List[int]] = defaultdict(list)
    self._by_concurrency: Dict[str, List[int]] = defaultdict(list)
    self._by_access: Dict[str, List[int]] = defaultdict(list)
    
    for action in self.actions:
        self._by_pattern[action.pattern].append(action.action_id)
        self._by_region[action.region].append(action.action_id)
        self._by_concurrency[action.concurrency].append(action.action_id)
        self._by_access[action.access_pattern].append(action.action_id)

def _compute_stress_score(
    self,
    pattern: str,
    concurrency: str,
    access_pattern: str,
    block_size: str,
    duration_s: int,
    memory_mb: int,
    copy_threads: int,
    cpu_threads: int,
    warm_copy: bool,
    max_errors: int,
) -> float:
    """Compute composite stress score for action parameters."""
    score = (
        PATTERN_STRESS[pattern]
        * CONCURRENCY_STRESS[concurrency]
        * ACCESS_STRESS[access_pattern]
        * BLOCK_STRESS[block_size]
        * DURATION_STRESS[duration_s]
        * MEMORY_STRESS[memory_mb]
        * COPY_THREADS_STRESS[copy_threads]
        * CPU_THREADS_STRESS[cpu_threads]
        * (1.1 if warm_copy else 1.0)
        * MAX_ERRORS_STRESS[max_errors]
    )
    return score

def _map_executor_params(
    self,
    concurrency: str,
    access_pattern: str,
    duration_s: int,
    block_size: str,
) -> Tuple[int, int, int, bool, int]:
    """
    Map semantic parameters to GSAT executor settings.
    
    Returns: (memory_mb, copy_threads, cpu_threads, warm_copy, max_errors)
    """
    # Base mapping from concurrency level
    if concurrency == "low":
        copy_threads = 4
        cpu_threads = 0
        memory_mb = 256 if duration_s <= 30 else 1024
        warm_copy = access_pattern != "sequential"
    elif concurrency == "medium":
        copy_threads = 8
        cpu_threads = 4
        memory_mb = 1024 if duration_s <= 60 else 4096
        warm_copy = True
    else:  # high
        copy_threads = 16
        cpu_threads = 8
        memory_mb = 4096 if duration_s <= 120 else 0
        warm_copy = True
    
    # Adjust for access pattern
    if access_pattern == "rowhammer_like":
        max_errors = 1  # Stop early on UE
        warm_copy = True
    else:
        max_errors = 10 if access_pattern == "random" else -1
    
    # Adjust for block size
    if block_size == "cacheline_64B":
        # Smaller blocks = more row activations
        if access_pattern == "rowhammer_like":
            max_errors = 1
    elif block_size == "large_2MB":
        # Large blocks = retention stress
        if duration_s >= 120 and access_pattern == "sequential":
            memory_mb = 0
    
    return memory_mb, copy_threads, cpu_threads, warm_copy, max_errors

def _build_action_library(self) -> None:
    """Build the complete action library based on granularity setting."""
    granularity = ActionGranularity(self.cfg.action_granularity)
    action_id = 0
    
    # Determine block sizes based on granularity
    if granularity == ActionGranularity.COARSE:
        block_sizes = ("page_4KB",)
    elif granularity == ActionGranularity.MEDIUM:
        block_sizes = self.cfg.block_sizes
    else:  # FULL
        block_sizes = self.cfg.block_sizes
    
    for pattern in self.cfg.patterns:
        for region in self.cfg.regions:
            for concurrency in self.cfg.concurrencies:
                for access_pattern in self.cfg.access_patterns:
                    for block_size in block_sizes:
                        for duration_s in self.cfg.durations_s:
                            # Get executor parameters
                            (memory_mb, copy_threads, cpu_threads,
                             warm_copy, max_errors) = self._map_executor_params(
                                concurrency, access_pattern, duration_s, block_size
                            )
                            
                            # Compute stress score
                            stress_score = self._compute_stress_score(
                                pattern, concurrency, access_pattern, block_size,
                                duration_s, memory_mb, copy_threads, cpu_threads,
                                warm_copy, max_errors
                            )
                            
                            self.actions.append(ActionSpec(
                                action_id=action_id,
                                pattern=pattern,
                                region=region,
                                concurrency=concurrency,
                                access_pattern=access_pattern,
                                block_size=block_size,
                                duration_s=duration_s,
                                memory_mb=memory_mb,
                                copy_threads=copy_threads,
                                cpu_threads=cpu_threads,
                                warm_copy=warm_copy,
                                max_errors=max_errors,
                                stress_score=stress_score,
                            ))
                            action_id += 1
    
    logger.info(f"Built action library with {len(self.actions)} actions "
               f"(granularity={granularity.value})")

def __len__(self) -> int:
    return len(self.actions)

def __getitem__(self, idx: int) -> ActionSpec:
    return self.actions[idx]

def get_actions_by_pattern(self, pattern: str) -> List[ActionSpec]:
    """Get all actions with specified pattern."""
    return [self.actions[i] for i in self._by_pattern[pattern]]

def get_actions_by_region(self, region: int) -> List[ActionSpec]:
    """Get all actions targeting specified region."""
    return [self.actions[i] for i in self._by_region[region]]

def get_high_stress_actions(self, threshold: float = 3.0) -> List[ActionSpec]:
    """Get actions with stress score above threshold."""
    return [a for a in self.actions if a.stress_score >= threshold]

def to_dataframe(self) -> pd.DataFrame:
    """Convert action library to DataFrame."""
    return pd.DataFrame([a.to_dict() for a in self.actions])

def save(self, path: Union[str, Path]) -> None:
    """Save action library to CSV."""
    self.to_dataframe().to_csv(path, index=False)
```

# =============================================================================

# SECTION 3: ERROR DETECTION & TELEMETRY

# =============================================================================

@dataclass
class TelemetrySnapshot:
“”“Point-in-time telemetry snapshot from EDAC/BMC.”””
timestamp: float
ce_count: int
ue_count: int
ce_per_region: np.ndarray
ue_per_region: np.ndarray
temperature_c: float
ecc_syndrome: int
error_locations: List[Tuple[int, int, int, int]]  # (bank, rank, row, col)

```
def delta(self, other: "TelemetrySnapshot") -> "TelemetryDelta":
    """Compute delta from another snapshot."""
    return TelemetryDelta(
        delta_ce=self.ce_count - other.ce_count,
        delta_ue=self.ue_count - other.ue_count,
        delta_ce_per_region=self.ce_per_region - other.ce_per_region,
        delta_ue_per_region=self.ue_per_region - other.ue_per_region,
        temperature_c=self.temperature_c,
        elapsed_s=self.timestamp - other.timestamp,
        new_error_locations=[
            loc for loc in self.error_locations
            if loc not in other.error_locations
        ],
    )
```

@dataclass
class TelemetryDelta:
“”“Change in telemetry between two snapshots.”””
delta_ce: int
delta_ue: int
delta_ce_per_region: np.ndarray
delta_ue_per_region: np.ndarray
temperature_c: float
elapsed_s: float
new_error_locations: List[Tuple[int, int, int, int]]

```
@property
def has_ce(self) -> bool:
    return self.delta_ce > 0

@property
def has_ue(self) -> bool:
    return self.delta_ue > 0

@property
def error_type(self) -> ErrorType:
    if self.has_ce and self.has_ue:
        return ErrorType.CE_UE
    elif self.has_ue:
        return ErrorType.UE
    elif self.has_ce:
        return ErrorType.CE
    return ErrorType.NONE
```

class EDACReader:
“””
Reads error counts from Linux EDAC subsystem.

```
EDAC exposes error counts via sysfs at:
/sys/devices/system/edac/mc/mc*/
    ce_count, ue_count - total counts
    csrow*/ce_count, ue_count - per chip-select row
"""

EDAC_ROOT = Path("/sys/devices/system/edac/mc")

def __init__(self, n_regions: int = 8):
    self.n_regions = n_regions
    self._available = self.EDAC_ROOT.exists()
    if self._available:
        logger.info("EDAC subsystem detected")
    else:
        logger.warning("EDAC subsystem not available")

@property
def available(self) -> bool:
    return self._available

def read_counts(self) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Read CE/UE counts from EDAC.
    
    Returns: (total_ce, total_ue, ce_per_region, ue_per_region)
    """
    total_ce = 0
    total_ue = 0
    ce_per_region = np.zeros(self.n_regions, dtype=np.int64)
    ue_per_region = np.zeros(self.n_regions, dtype=np.int64)
    
    if not self._available:
        return total_ce, total_ue, ce_per_region, ue_per_region
    
    try:
        # Read all memory controllers
        for mc_path in self.EDAC_ROOT.iterdir():
            if not mc_path.is_dir() or not mc_path.name.startswith("mc"):
                continue
            
            # Total counts
            ce_path = mc_path / "ce_count"
            ue_path = mc_path / "ue_count"
            
            if ce_path.exists():
                total_ce += int(ce_path.read_text().strip())
            if ue_path.exists():
                total_ue += int(ue_path.read_text().strip())
            
            # Per-CSROW counts (map to regions)
            region_idx = 0
            for csrow_path in sorted(mc_path.glob("csrow*")):
                if region_idx >= self.n_regions:
                    break
                
                csrow_ce = csrow_path / "ce_count"
                csrow_ue = csrow_path / "ue_count"
                
                if csrow_ce.exists():
                    ce_per_region[region_idx] += int(csrow_ce.read_text().strip())
                if csrow_ue.exists():
                    ue_per_region[region_idx] += int(csrow_ue.read_text().strip())
                
                region_idx += 1
    
    except Exception as e:
        logger.error(f"Error reading EDAC: {e}")
    
    return total_ce, total_ue, ce_per_region, ue_per_region
```

class ThermalMonitor:
“”“Monitors DIMM temperature from hwmon or IPMI.”””

```
HWMON_ROOT = Path("/sys/class/hwmon")

def __init__(self, default_temp: float = 40.0):
    self.default_temp = default_temp
    self._dimm_temp_path: Optional[Path] = None
    self._find_dimm_sensor()

def _find_dimm_sensor(self) -> None:
    """Search for DIMM temperature sensor in hwmon."""
    if not self.HWMON_ROOT.exists():
        return
    
    for hwmon in self.HWMON_ROOT.iterdir():
        name_path = hwmon / "name"
        if name_path.exists():
            name = name_path.read_text().strip().lower()
            if "dimm" in name or "ddr" in name or "memory" in name:
                # Look for temp input
                for temp_path in hwmon.glob("temp*_input"):
                    self._dimm_temp_path = temp_path
                    logger.info(f"Found DIMM temperature sensor: {temp_path}")
                    return

def read_temperature(self) -> float:
    """Read current DIMM temperature in Celsius."""
    if self._dimm_temp_path and self._dimm_temp_path.exists():
        try:
            # hwmon temp is in millidegrees
            millidegrees = int(self._dimm_temp_path.read_text().strip())
            return millidegrees / 1000.0
        except Exception:
            pass
    
    return self.default_temp
```

class TelemetryCollector:
“”“Unified telemetry collection from EDAC, thermal, and BMC sources.”””

```
def __init__(self, cfg: DRAMSnifferConfig):
    self.cfg = cfg
    self.n_regions = len(cfg.regions)
    self.edac = EDACReader(n_regions=self.n_regions)
    self.thermal = ThermalMonitor(default_temp=cfg.base_temperature_c)
    
    # Baseline for relative measurements
    self._baseline: Optional[TelemetrySnapshot] = None

@property
def edac_available(self) -> bool:
    return self.edac.available

def snapshot(self) -> TelemetrySnapshot:
    """Take current telemetry snapshot."""
    ce, ue, ce_region, ue_region = self.edac.read_counts()
    temp = self.thermal.read_temperature()
    
    return TelemetrySnapshot(
        timestamp=time.time(),
        ce_count=ce,
        ue_count=ue,
        ce_per_region=ce_region,
        ue_per_region=ue_region,
        temperature_c=temp,
        ecc_syndrome=0,  # Would require MCE parsing
        error_locations=[],
    )

def set_baseline(self) -> TelemetrySnapshot:
    """Set current state as baseline."""
    self._baseline = self.snapshot()
    return self._baseline

def get_delta(self) -> Optional[TelemetryDelta]:
    """Get delta from baseline."""
    if self._baseline is None:
        return None
    current = self.snapshot()
    return current.delta(self._baseline)
```

# =============================================================================

# SECTION 4: EXECUTION BACKENDS

# =============================================================================

class ExecutionBackend(ABC):
“”“Abstract base class for action execution backends.”””

```
@abstractmethod
def execute(
    self,
    action: ActionSpec,
    cfg: DRAMSnifferConfig,
) -> Tuple[TelemetryDelta, Dict[str, Any]]:
    """
    Execute action and return telemetry delta.
    
    Returns: (delta, info_dict)
    """
    pass

@abstractmethod
def reset(self) -> None:
    """Reset backend state for new episode."""
    pass
```

class SyntheticBackend(ExecutionBackend):
“””
Probabilistic simulation backend for Phase 0 development.

```
Implements error model calibrated from field studies:
- CE probability scales with stress score and region vulnerability
- UE probability has two paths:
  1. Escalation: increases with accumulated CEs (predictable UE)
  2. Sudden: base probability, increased by rowhammer-like patterns
"""

def __init__(self, cfg: DRAMSnifferConfig, timed: bool = False):
    self.cfg = cfg
    self.timed = timed
    self.n_regions = len(cfg.regions)
    self.rng = np.random.default_rng()
    
    # Episode state
    self.region_vulnerability: np.ndarray = np.ones(self.n_regions)
    self.region_ce_counts: np.ndarray = np.zeros(self.n_regions, dtype=np.int64)
    self.temperature_c: float = cfg.base_temperature_c
    self.total_ce: int = 0
    self.total_ue: int = 0
    self.simulated_seconds: float = 0.0
    
    # Fixed vulnerability map if configured
    self._fixed_vulnerability: Optional[np.ndarray] = None
    if cfg.fixed_weak_map:
        self._fixed_vulnerability = self._sample_vulnerability()

def _sample_vulnerability(self) -> np.ndarray:
    """Sample region vulnerability map."""
    vul = np.ones(self.n_regions, dtype=np.float32)
    
    # Select weak regions
    n_weak = max(1, int(self.cfg.weak_region_fraction * self.n_regions))
    weak_indices = self.rng.choice(self.n_regions, size=n_weak, replace=False)
    
    # Assign vulnerabilities
    for i in range(self.n_regions):
        if i in weak_indices:
            vul[i] = self.rng.uniform(
                self.cfg.weak_region_vulnerability_min,
                self.cfg.weak_region_vulnerability_max,
            )
        else:
            vul[i] = self.rng.uniform(
                self.cfg.normal_region_vulnerability_min,
                self.cfg.normal_region_vulnerability_max,
            )
    
    return vul

def _temperature_factor(self) -> float:
    """Compute error probability multiplier from temperature."""
    if self.temperature_c >= self.cfg.temperature_error_threshold_c:
        excess = self.temperature_c - self.cfg.temperature_error_threshold_c
        return 1.0 + 0.015 * min(excess, 30.0)
    return 1.0

def _update_temperature(self, action: ActionSpec) -> None:
    """Update simulated temperature based on action."""
    # Duration contribution
    duration_heat = {30: 0.3, 60: 0.6, 120: 1.2, 300: 2.5}
    heat = duration_heat.get(action.duration_s, 1.0)
    
    # Concurrency contribution
    concurrency_heat = {"low": 0.2, "medium": 0.5, "high": 1.0}
    heat += concurrency_heat.get(action.concurrency, 0.5)
    
    # CPU threads contribution
    heat += action.cpu_threads * 0.08
    
    # Warm copy contribution
    if action.warm_copy:
        heat += 0.3
    
    # Cooling
    cool = 0.4
    
    self.temperature_c = float(np.clip(
        self.temperature_c + heat - cool,
        self.cfg.temperature_min_c,
        self.cfg.temperature_max_c,
    ))

def _compute_ce_probability(self, action: ActionSpec) -> float:
    """Compute CE probability for this action."""
    base = self.cfg.base_ce_prob
    
    # Scale by stress and vulnerability
    stress = action.stress_score
    region_vul = self.region_vulnerability[action.region]
    
    # Region history bonus (CE-active regions more likely to have more CEs)
    region_ce = self.region_ce_counts[action.region]
    history_bonus = 1.0 + 0.15 * min(region_ce, 10)
    
    # Temperature factor
    temp_factor = self._temperature_factor()
    
    # Access pattern specifics
    pattern_bonus = 1.0
    if action.access_pattern == "sequential" and action.duration_s >= 120:
        pattern_bonus = 1.15  # Retention stress
    elif action.access_pattern == "rowhammer_like":
        pattern_bonus = 1.10  # Row activation stress
    
    p_ce = base * stress * region_vul * history_bonus * temp_factor * pattern_bonus
    return float(np.clip(p_ce, 0.0, 0.8))

def _compute_ue_probability(self, action: ActionSpec) -> float:
    """
    Compute UE probability combining escalation and sudden paths.
    
    Based on field data:
    - ~57% of UEs have CE precursors (escalation path)
    - ~43% of UEs have no CE precursors (sudden path)
    """
    stress = action.stress_score
    region_vul = self.region_vulnerability[action.region]
    region_ce = self.region_ce_counts[action.region]
    temp_factor = self._temperature_factor()
    
    # Escalation path: probability increases with CEs
    # Calibrated from Google finding: CE→UE correlation 9-400×
    p_ue_escalation = self.cfg.base_ue_prob * stress * region_vul * temp_factor
    if region_ce > 0:
        # Escalation factor increases with CE count
        esc_factor = min(
            self.cfg.ce_escalation_factor * math.log1p(region_ce),
            self.cfg.ce_ue_correlation_max,
        )
        p_ue_escalation *= esc_factor
    
    # Sudden path: ~43% of UEs, higher for rowhammer-like
    p_ue_sudden = self.cfg.base_ue_sudden_prob * stress * temp_factor
    if action.access_pattern == "rowhammer_like":
        p_ue_sudden *= 3.0  # Strong row-hammer effect
    if action.pattern == "butterfly":
        p_ue_sudden *= 1.5  # Cell coupling stress
    if action.duration_s >= 120:
        p_ue_sudden *= 1.2  # Extended stress
    
    # Combined probability (independent events)
    p_ue = 1.0 - (1.0 - np.clip(p_ue_escalation, 0, 0.3)) * (1.0 - np.clip(p_ue_sudden, 0, 0.15))
    return float(np.clip(p_ue, 0.0, 0.35))

def execute(
    self,
    action: ActionSpec,
    cfg: DRAMSnifferConfig,
) -> Tuple[TelemetryDelta, Dict[str, Any]]:
    """Execute simulated action."""
    # Update temperature
    self._update_temperature(action)
    
    # Compute error probabilities
    p_ce = self._compute_ce_probability(action)
    p_ue = self._compute_ue_probability(action)
    
    # Sample errors
    # CE follows Poisson (can have multiple CEs per step)
    delta_ce = int(self.rng.poisson(lam=p_ce))
    # UE is rare binary event
    delta_ue = int(self.rng.random() < p_ue)
    
    # Update state
    self.region_ce_counts[action.region] += delta_ce
    self.total_ce += delta_ce
    self.total_ue += delta_ue
    self.simulated_seconds += action.duration_s
    
    # Optional timing simulation
    actual_elapsed = 0.0
    if self.timed and cfg.sleep_scale > 0:
        sleep_time = action.duration_s * cfg.sleep_scale
        time.sleep(sleep_time)
        actual_elapsed = sleep_time
    
    # Build delta
    delta_ce_per_region = np.zeros(self.n_regions, dtype=np.int64)
    delta_ce_per_region[action.region] = delta_ce
    delta_ue_per_region = np.zeros(self.n_regions, dtype=np.int64)
    delta_ue_per_region[action.region] = delta_ue
    
    delta = TelemetryDelta(
        delta_ce=delta_ce,
        delta_ue=delta_ue,
        delta_ce_per_region=delta_ce_per_region,
        delta_ue_per_region=delta_ue_per_region,
        temperature_c=self.temperature_c,
        elapsed_s=actual_elapsed,
        new_error_locations=[],
    )
    
    info = {
        "p_ce": p_ce,
        "p_ue": p_ue,
        "region_vulnerability": self.region_vulnerability[action.region],
        "simulated_seconds": self.simulated_seconds,
        "actual_elapsed": actual_elapsed,
    }
    
    return delta, info

def reset(self) -> None:
    """Reset for new episode."""
    if self._fixed_vulnerability is not None:
        self.region_vulnerability = self._fixed_vulnerability.copy()
    else:
        self.region_vulnerability = self._sample_vulnerability()
    
    self.region_ce_counts = np.zeros(self.n_regions, dtype=np.int64)
    self.temperature_c = self.cfg.base_temperature_c
    self.total_ce = 0
    self.total_ue = 0
    self.simulated_seconds = 0.0
```

class GSATBackend(ExecutionBackend):
“””
Real hardware backend using stressapptest with EDAC monitoring.

```
For Phase 1/2: actual stress execution on real DDR DIMMs.
"""

def __init__(self, cfg: DRAMSnifferConfig):
    self.cfg = cfg
    self.telemetry = TelemetryCollector(cfg)
    
    # Validate environment
    self._validate_environment()

def _validate_environment(self) -> None:
    """Validate that required tools and interfaces are available."""
    # Check OS
    if os.name != "posix" or sys.platform != "linux":
        raise RuntimeError("GSATBackend requires Linux")
    
    # Check GSAT binary
    if shutil.which(self.cfg.gsat_binary) is None:
        raise RuntimeError(f"GSAT binary not found: {self.cfg.gsat_binary}")
    
    # Check numactl if needed
    if self.cfg.use_numactl and shutil.which("numactl") is None:
        raise RuntimeError("numactl not found but use_numactl=True")
    
    # Check EDAC
    if self.cfg.require_edac and not self.telemetry.edac_available:
        raise RuntimeError("EDAC required but not available")
    
    logger.info("GSATBackend environment validated")

def _run_gsat(self, command: str) -> Tuple[int, float, str, str]:
    """Execute GSAT command with timeout."""
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.cfg.gsat_timeout_s,
        )
        elapsed = time.time() - start
        return proc.returncode, elapsed, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return -1, elapsed, "", "Timeout"

def execute(
    self,
    action: ActionSpec,
    cfg: DRAMSnifferConfig,
) -> Tuple[TelemetryDelta, Dict[str, Any]]:
    """Execute real GSAT stress test."""
    # Pre-execution snapshot
    pre_snapshot = self.telemetry.snapshot()
    
    # Build and run command
    command = action.to_gsat_command(
        gsat_binary=cfg.gsat_binary,
        use_numactl=cfg.use_numactl,
    )
    
    logger.debug(f"Executing: {command}")
    returncode, elapsed, stdout, stderr = self._run_gsat(command)
    
    # Post-execution snapshot
    post_snapshot = self.telemetry.snapshot()
    delta = post_snapshot.delta(pre_snapshot)
    
    info = {
        "gsat_returncode": returncode,
        "gsat_elapsed": elapsed,
        "gsat_stdout": stdout[:2000] if stdout else "",
        "gsat_stderr": stderr[:2000] if stderr else "",
        "gsat_command": command,
    }
    
    return delta, info

def reset(self) -> None:
    """Reset for new episode."""
    self.telemetry.set_baseline()
```

def create_backend(cfg: DRAMSnifferConfig) -> ExecutionBackend:
“”“Factory function to create appropriate backend.”””
backend_type = BackendType(cfg.backend)

```
if backend_type == BackendType.SYNTHETIC_FAST:
    return SyntheticBackend(cfg, timed=False)
elif backend_type == BackendType.SYNTHETIC_TIMED:
    return SyntheticBackend(cfg, timed=True)
elif backend_type == BackendType.GSAT_WRAPPER:
    return GSATBackend(cfg)
else:
    raise ValueError(f"Unknown backend: {cfg.backend}")
```

# =============================================================================

# SECTION 5: GYMNASIUM ENVIRONMENT

# =============================================================================

class DRAMSnifferEnv(gym.Env):
“””
Gymnasium environment for DRAM-SNIFFER RL training.

```
Implements the MDP formulation from the Framework Proposal:
- State: [CE metrics, error locations, temperature, pattern history]
- Action: GSAT configuration selection
- Reward: r = α·ΔCE + β·ΔUE + γ·coverage + exploration - penalty
- Termination: UE detected (success) or max steps reached
"""

metadata = {"render_modes": ["human"]}

def __init__(self, cfg: DRAMSnifferConfig, backend: Optional[ExecutionBackend] = None):
    super().__init__()
    
    self.cfg = cfg
    self.action_library = ActionLibrary(cfg)
    self.backend = backend or create_backend(cfg)
    
    # Define spaces
    self.action_space = spaces.Discrete(len(self.action_library))
    obs_dim = self._compute_observation_dim()
    self.observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
    )
    
    # Episode state
    self._reset_state()
    
    # Action history for exploration bonus
    self.action_counts: Counter = Counter()
    self.global_action_counts: Counter = Counter()  # Persists across episodes

def _compute_observation_dim(self) -> int:
    """Compute observation vector dimension."""
    n_regions = len(self.cfg.regions)
    n_patterns = len(self.cfg.patterns)
    n_concurrencies = len(self.cfg.concurrencies)
    n_access = len(self.cfg.access_patterns)
    n_blocks = len(self.cfg.block_sizes)
    
    return (
        6 +                    # Scalar metrics: CE, UE, rate, accel, temp, coverage
        n_regions +            # CE per region
        n_regions +            # Visit count per region
        n_patterns +           # One-hot last pattern
        n_concurrencies +      # One-hot last concurrency
        n_access +             # One-hot last access
        n_blocks +             # One-hot last block
        6                      # Meta features
    )

def _reset_state(self) -> None:
    """Reset all episode state."""
    self.step_count = 0
    self.total_ce = 0
    self.total_ue = 0
    self.ce_rate = 0.0
    self.prev_ce_rate = 0.0
    self.ce_accel = 0.0
    self.temperature_c = self.cfg.base_temperature_c
    self.region_ce_counts = np.zeros(len(self.cfg.regions), dtype=np.int64)
    self.region_visit_counts = np.zeros(len(self.cfg.regions), dtype=np.int32)
    self.error_regions_seen: Set[int] = set()
    self.coverage_count = 0
    self.simulated_seconds = 0.0
    self.actual_elapsed = 0.0
    
    # Last action encoding
    self.last_pattern_idx = 0
    self.last_concurrency_idx = 0
    self.last_access_idx = 0
    self.last_block_idx = 0
    self.last_duration = 30
    self.last_stress = 0.0
    
    # Action history for episode
    self.action_counts = Counter()
    self.episode_actions: List[int] = []

def _normalize(self, x: float, scale: float) -> float:
    """Normalize value using saturation function."""
    return float(x / (x + scale))

def _one_hot(self, idx: int, size: int) -> np.ndarray:
    """Create one-hot encoding."""
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        arr[idx] = 1.0
    return arr

def _get_observation(self) -> np.ndarray:
    """Build observation vector."""
    n_regions = len(self.cfg.regions)
    n_patterns = len(self.cfg.patterns)
    n_concurrencies = len(self.cfg.concurrencies)
    n_access = len(self.cfg.access_patterns)
    n_blocks = len(self.cfg.block_sizes)
    max_steps = self.cfg.max_steps_per_episode
    
    # Scalar metrics
    scalars = np.array([
        self._normalize(self.total_ce, 10.0),
        min(float(self.total_ue), 1.0),
        np.clip(self.ce_rate / 5.0, 0.0, 1.0),
        np.clip((self.ce_accel + 2.0) / 4.0, 0.0, 1.0),
        np.clip(
            (self.temperature_c - self.cfg.temperature_min_c) /
            (self.cfg.temperature_max_c - self.cfg.temperature_min_c),
            0.0, 1.0
        ),
        self._normalize(self.coverage_count, 6.0),
    ], dtype=np.float32)
    
    # Per-region metrics
    region_ce_norm = np.array([
        self._normalize(x, 5.0) for x in self.region_ce_counts
    ], dtype=np.float32)
    
    region_visit_norm = np.array([
        min(x / max(max_steps, 1), 1.0) for x in self.region_visit_counts
    ], dtype=np.float32)
    
    # One-hot encodings
    pattern_oh = self._one_hot(self.last_pattern_idx, n_patterns)
    concurrency_oh = self._one_hot(self.last_concurrency_idx, n_concurrencies)
    access_oh = self._one_hot(self.last_access_idx, n_access)
    block_oh = self._one_hot(self.last_block_idx, n_blocks)
    
    # Meta features
    meta = np.array([
        self._normalize(self.simulated_seconds, 1800.0),
        min(self.step_count / max_steps, 1.0),
        self._normalize(len(self.error_regions_seen), n_regions),
        self._normalize(self.last_stress, 5.0),
        min(self.last_duration / 300.0, 1.0),
        self._normalize(len(self.action_counts), len(self.action_library) / 10),
    ], dtype=np.float32)
    
    obs = np.concatenate([
        scalars,
        region_ce_norm,
        region_visit_norm,
        pattern_oh,
        concurrency_oh,
        access_oh,
        block_oh,
        meta,
    ])
    
    return obs.astype(np.float32)

def _compute_reward(
    self,
    delta: TelemetryDelta,
    action: ActionSpec,
) -> float:
    """Compute reward following framework proposal."""
    # Base rewards for errors
    ce_reward = self.cfg.alpha_ce * delta.delta_ce
    ue_reward = self.cfg.beta_ue * delta.delta_ue
    
    # Coverage bonus for new error regions
    coverage_bonus = 0.0
    if delta.delta_ce > 0 and action.region not in self.error_regions_seen:
        coverage_bonus = self.cfg.gamma_coverage
    
    # Exploration bonus for trying new actions
    exploration_bonus = 0.0
    if self.action_counts[action.action_id] == 1:  # First time this episode
        exploration_bonus = self.cfg.exploration_bonus * 0.5
    if self.global_action_counts[action.action_id] <= 3:  # Rarely tried globally
        exploration_bonus += self.cfg.exploration_bonus * 0.5
    
    # Time penalty
    time_penalty = self.cfg.time_penalty_coef * action.duration_s
    
    reward = ce_reward + ue_reward + coverage_bonus + exploration_bonus - time_penalty
    return float(reward)

def reset(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reset environment for new episode."""
    super().reset(seed=seed)
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    self._reset_state()
    self.backend.reset()
    
    obs = self._get_observation()
    info = {"episode_start": True}
    
    return obs, info

def step(
    self,
    action_idx: int,
) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    """Execute one step in the environment."""
    action = self.action_library[int(action_idx)]
    
    # Update counters
    self.step_count += 1
    self.action_counts[action.action_id] += 1
    self.global_action_counts[action.action_id] += 1
    self.episode_actions.append(action.action_id)
    self.region_visit_counts[action.region] += 1
    
    # Execute action
    delta, exec_info = self.backend.execute(action, self.cfg)
    
    # Update state from delta
    self.prev_ce_rate = self.ce_rate
    self.ce_rate = float(delta.delta_ce)
    self.ce_accel = self.ce_rate - self.prev_ce_rate
    
    self.total_ce += delta.delta_ce
    self.total_ue += delta.delta_ue
    self.region_ce_counts[action.region] += delta.delta_ce
    self.temperature_c = delta.temperature_c
    self.simulated_seconds += action.duration_s
    self.actual_elapsed += delta.elapsed_s
    
    # Update coverage
    if delta.delta_ce > 0 and action.region not in self.error_regions_seen:
        self.error_regions_seen.add(action.region)
        self.coverage_count += 1
    
    # Update last action encoding
    self.last_pattern_idx = list(self.cfg.patterns).index(action.pattern)
    self.last_concurrency_idx = list(self.cfg.concurrencies).index(action.concurrency)
    self.last_access_idx = list(self.cfg.access_patterns).index(action.access_pattern)
    self.last_block_idx = list(self.cfg.block_sizes).index(action.block_size)
    self.last_duration = action.duration_s
    self.last_stress = action.stress_score
    
    # Compute reward
    reward = self._compute_reward(delta, action)
    
    # Check termination
    terminated = delta.delta_ue > 0  # UE = high-value success
    truncated = self.step_count >= self.cfg.max_steps_per_episode
    
    # Build info dict
    obs = self._get_observation()
    info = {
        "delta_ce": delta.delta_ce,
        "delta_ue": delta.delta_ue,
        "error_type": delta.error_type.name,
        "region": action.region,
        "pattern": action.pattern,
        "concurrency": action.concurrency,
        "access_pattern": action.access_pattern,
        "block_size": action.block_size,
        "duration_s": action.duration_s,
        "stress_score": action.stress_score,
        "temperature_c": self.temperature_c,
        "total_ce": self.total_ce,
        "total_ue": self.total_ue,
        "coverage": self.coverage_count,
        "simulated_seconds": self.simulated_seconds,
        "actual_elapsed": self.actual_elapsed,
        "gsat_command": action.to_gsat_command(self.cfg.gsat_binary, self.cfg.use_numactl),
        **exec_info,
    }
    
    return obs, reward, terminated, truncated, info

def render(self) -> None:
    """Render current state (human-readable)."""
    print(f"\n{'='*60}")
    print(f"Step: {self.step_count}/{self.cfg.max_steps_per_episode}")
    print(f"Total CE: {self.total_ce}, Total UE: {self.total_ue}")
    print(f"CE Rate: {self.ce_rate:.2f}, Acceleration: {self.ce_accel:.2f}")
    print(f"Temperature: {self.temperature_c:.1f}°C")
    print(f"Coverage: {self.coverage_count}/{len(self.cfg.regions)} regions")
    print(f"Simulated time: {self.simulated_seconds:.0f}s")
    print(f"{'='*60}\n")

def get_episode_summary(self) -> Dict[str, Any]:
    """Get summary of current/completed episode."""
    return {
        "steps": self.step_count,
        "total_ce": self.total_ce,
        "total_ue": self.total_ue,
        "coverage": self.coverage_count,
        "unique_actions": len(self.action_counts),
        "simulated_seconds": self.simulated_seconds,
        "actual_elapsed": self.actual_elapsed,
        "error_regions": list(self.error_regions_seen),
        "action_sequence": self.episode_actions.copy(),
    }
```

# =============================================================================

# SECTION 6: TRAINING & EVALUATION

# =============================================================================

class DRAMSnifferCallback(BaseCallback):
“”“Custom callback for training monitoring and checkpointing.”””

```
def __init__(
    self,
    save_path: Path,
    save_freq: int = 10000,
    log_freq: int = 1000,
    verbose: int = 1,
):
    super().__init__(verbose)
    self.save_path = Path(save_path)
    self.save_freq = save_freq
    self.log_freq = log_freq
    
    self.episode_rewards: List[float] = []
    self.episode_lengths: List[int] = []
    self.episode_ces: List[int] = []
    self.episode_ues: List[int] = []
    self.current_episode_reward = 0.0
    self.current_episode_length = 0

def _on_step(self) -> bool:
    # Track episode stats
    self.current_episode_reward += self.locals.get("rewards", [0])[0]
    self.current_episode_length += 1
    
    # Check for episode end
    dones = self.locals.get("dones", [False])
    if dones[0]:
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        
        # Get CE/UE from info
        infos = self.locals.get("infos", [{}])
        self.episode_ces.append(infos[0].get("total_ce", 0))
        self.episode_ues.append(infos[0].get("total_ue", 0))
        
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    # Periodic logging
    if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
        recent_rewards = self.episode_rewards[-100:]
        recent_ces = self.episode_ces[-100:]
        recent_ues = self.episode_ues[-100:]
        
        logger.info(
            f"Step {self.n_calls}: "
            f"Reward={np.mean(recent_rewards):.2f}, "
            f"CE={np.mean(recent_ces):.2f}, "
            f"UE_rate={np.mean([1 if u > 0 else 0 for u in recent_ues]):.2%}"
        )
    
    # Periodic saving
    if self.n_calls % self.save_freq == 0:
        self.model.save(self.save_path / f"checkpoint_{self.n_calls}")
        logger.info(f"Saved checkpoint at step {self.n_calls}")
    
    return True

def get_training_history(self) -> Dict[str, List]:
    """Get training history for plotting."""
    return {
        "episode_rewards": self.episode_rewards.copy(),
        "episode_lengths": self.episode_lengths.copy(),
        "episode_ces": self.episode_ces.copy(),
        "episode_ues": self.episode_ues.copy(),
    }
```

@dataclass
class EvaluationResult:
“”“Results from agent evaluation.”””
agent_name: str
episodes_df: pd.DataFrame
actions_df: pd.DataFrame
summary: Dict[str, float]
tc_library: List[Dict[str, Any]]

```
def save(self, output_dir: Path) -> None:
    """Save all results to directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    self.episodes_df.to_csv(output_dir / f"{self.agent_name}_episodes.csv", index=False)
    self.actions_df.to_csv(output_dir / f"{self.agent_name}_actions.csv", index=False)
    
    with open(output_dir / f"{self.agent_name}_summary.json", "w") as f:
        json.dump(self.summary, f, indent=2, cls=NumpyJSONEncoder)
    
    with open(output_dir / f"{self.agent_name}_tc_library.json", "w") as f:
        json.dump(self.tc_library, f, indent=2, cls=NumpyJSONEncoder)
```

def evaluate_agent(
model: Optional[PPO],
cfg: DRAMSnifferConfig,
n_episodes: int,
seed: int,
agent_name: str,
deterministic: bool = True,
) -> EvaluationResult:
“””
Evaluate agent performance.

```
Args:
    model: Trained PPO model (None for random baseline)
    cfg: Configuration
    n_episodes: Number of evaluation episodes
    seed: Random seed
    agent_name: Name for logging/saving
    deterministic: Use deterministic policy

Returns:
    EvaluationResult with detailed metrics
"""
env = DRAMSnifferEnv(cfg)
action_library = env.action_library

episode_rows = []
action_rows = []
tc_library = []

uniform_entropy = math.log(len(action_library))

for ep in range(n_episodes):
    obs, _ = env.reset(seed=seed + ep)
    done = False
    
    episode_reward = 0.0
    episode_actions = []
    steps_to_first_ce = None
    steps_to_first_ue = None
    sim_seconds_to_first_ce = None
    sim_seconds_to_first_ue = None
    max_ce_step = 0
    entropies = []
    
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            # Compute policy entropy
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            dist = model.policy.get_distribution(obs_tensor)
            entropy = float(dist.distribution.entropy().mean().item())
            entropies.append(entropy)
        else:
            action = env.action_space.sample()
            entropies.append(uniform_entropy)
        
        episode_actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        max_ce_step = max(max_ce_step, info["delta_ce"])
        
        if steps_to_first_ce is None and info["delta_ce"] > 0:
            steps_to_first_ce = env.step_count
            sim_seconds_to_first_ce = info["simulated_seconds"]
        
        if steps_to_first_ue is None and info["delta_ue"] > 0:
            steps_to_first_ue = env.step_count
            sim_seconds_to_first_ue = info["simulated_seconds"]
            
            # Record UE-inducing TC
            tc_library.append({
                "episode": ep,
                "steps": env.step_count,
                "action_sequence": episode_actions.copy(),
                "final_action": action,
                "simulated_seconds": info["simulated_seconds"],
                "total_ce": info["total_ce"],
                "gsat_commands": [
                    action_library[a].to_gsat_command(cfg.gsat_binary)
                    for a in episode_actions
                ],
            })
        
        done = terminated or truncated
    
    # Fill in defaults for episodes without errors
    max_steps = cfg.max_steps_per_episode
    if steps_to_first_ce is None:
        steps_to_first_ce = max_steps + 1
        sim_seconds_to_first_ce = env.simulated_seconds
    if steps_to_first_ue is None:
        steps_to_first_ue = max_steps + 1
        sim_seconds_to_first_ue = env.simulated_seconds
    
    # Episode summary
    episode_rows.append({
        "agent": agent_name,
        "episode": ep,
        "episode_reward": episode_reward,
        "steps_to_first_ce": steps_to_first_ce,
        "steps_to_first_ue": steps_to_first_ue,
        "sim_seconds_to_first_ce": sim_seconds_to_first_ce,
        "sim_seconds_to_first_ue": sim_seconds_to_first_ue,
        "max_ce_step": max_ce_step,
        "total_ce": env.total_ce,
        "total_ue": env.total_ue,
        "ue_found": 1 if env.total_ue > 0 else 0,
        "coverage": env.coverage_count,
        "unique_actions": len(set(episode_actions)),
        "mean_entropy": np.mean(entropies) if entropies else 0.0,
        "simulated_seconds": env.simulated_seconds,
    })
    
    # Action usage
    action_counts = Counter(episode_actions)
    for aid, count in action_counts.items():
        spec = action_library[aid]
        action_rows.append({
            "agent": agent_name,
            "episode": ep,
            "action_id": aid,
            "count": count,
            **spec.to_dict(),
        })

episodes_df = pd.DataFrame(episode_rows)
actions_df = pd.DataFrame(action_rows)

# Compute summary statistics
summary = {
    "mean_reward": float(episodes_df["episode_reward"].mean()),
    "mean_steps_to_first_ce": float(episodes_df["steps_to_first_ce"].mean()),
    "mean_steps_to_first_ue": float(episodes_df["steps_to_first_ue"].mean()),
    "mean_sim_seconds_to_first_ce": float(episodes_df["sim_seconds_to_first_ce"].mean()),
    "mean_sim_seconds_to_first_ue": float(episodes_df["sim_seconds_to_first_ue"].mean()),
    "mean_total_ce": float(episodes_df["total_ce"].mean()),
    "ue_rate": float(episodes_df["ue_found"].mean()),
    "mean_coverage": float(episodes_df["coverage"].mean()),
    "mean_entropy": float(episodes_df["mean_entropy"].mean()),
    "mean_unique_actions": float(episodes_df["unique_actions"].mean()),
}

return EvaluationResult(
    agent_name=agent_name,
    episodes_df=episodes_df,
    actions_df=actions_df,
    summary=summary,
    tc_library=tc_library,
)
```

def train_agent(
cfg: DRAMSnifferConfig,
output_dir: Path,
seed: int = 0,
) -> Tuple[PPO, DRAMSnifferCallback]:
“””
Train PPO agent on DRAM-SNIFFER environment.

```
Returns: (trained_model, callback_with_history)
"""
if not SB3_AVAILABLE:
    raise RuntimeError(f"stable-baselines3 required for training: {SB3_IMPORT_ERROR}")

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Create environment
env = DRAMSnifferEnv(cfg)
env = Monitor(env, str(output_dir / "monitor"))

# Create model
policy_kwargs = {"net_arch": list(cfg.net_arch)}

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=cfg.learning_rate,
    n_steps=cfg.n_steps,
    batch_size=cfg.batch_size,
    n_epochs=cfg.n_epochs,
    gamma=cfg.gamma,
    gae_lambda=cfg.gae_lambda,
    clip_range=cfg.clip_range,
    ent_coef=cfg.ent_coef,
    vf_coef=cfg.vf_coef,
    max_grad_norm=cfg.max_grad_norm,
    policy_kwargs=policy_kwargs,
    verbose=cfg.verbose,
    seed=seed,
)

# Create callback
callback = DRAMSnifferCallback(
    save_path=output_dir,
    save_freq=cfg.save_freq,
    log_freq=1000,
    verbose=cfg.verbose,
)

# Train
logger.info(f"Starting training for {cfg.total_timesteps} timesteps")
model.learn(
    total_timesteps=cfg.total_timesteps,
    callback=callback,
    progress_bar=True,
)

# Save final model
model.save(output_dir / "model_final")
logger.info(f"Training complete. Model saved to {output_dir / 'model_final'}")

return model, callback
```

# =============================================================================

# SECTION 7: VISUALIZATION

# =============================================================================

def plot_comparison(
ppo_result: EvaluationResult,
random_result: EvaluationResult,
output_dir: Path,
) -> None:
“”“Generate comparison plots between PPO and random agents.”””
if not MATPLOTLIB_AVAILABLE:
logger.warning(“Matplotlib not available, skipping plots”)
return

```
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.concat([ppo_result.episodes_df, random_result.episodes_df], ignore_index=True)

# Box plots for key metrics
metrics_box = [
    ("steps_to_first_ce", "Steps to First CE"),
    ("steps_to_first_ue", "Steps to First UE"),
    ("total_ce", "Total CEs per Episode"),
    ("episode_reward", "Episode Reward"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (metric, title) in zip(axes, metrics_box):
    data = [
        df[df["agent"] == "ppo"][metric].values,
        df[df["agent"] == "random"][metric].values,
    ]
    bp = ax.boxplot(data, labels=["PPO", "Random"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#2196F3")
    ax.set_title(title)
    ax.set_ylabel(metric)

plt.tight_layout()
plt.savefig(output_dir / "comparison_boxplots.png", dpi=150)
plt.close()

# Bar chart for summary metrics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

summary_metrics = [
    ("mean_steps_to_first_ce", "Mean Steps to First CE"),
    ("mean_steps_to_first_ue", "Mean Steps to First UE"),
    ("ue_rate", "UE Discovery Rate"),
    ("mean_total_ce", "Mean Total CEs"),
    ("mean_coverage", "Mean Region Coverage"),
    ("mean_entropy", "Mean Action Entropy"),
]

ppo_vals = [ppo_result.summary[m[0]] for m in summary_metrics]
rnd_vals = [random_result.summary[m[0]] for m in summary_metrics]

for ax, (metric, title), ppo_v, rnd_v in zip(axes, summary_metrics, ppo_vals, rnd_vals):
    x = np.arange(2)
    bars = ax.bar(x, [ppo_v, rnd_v], color=["#4CAF50", "#2196F3"])
    ax.set_xticks(x)
    ax.set_xticklabels(["PPO", "Random"])
    ax.set_title(title)
    ax.set_ylabel(metric)
    
    # Add value labels
    for bar, val in zip(bars, [ppo_v, rnd_v]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f"{val:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(output_dir / "comparison_summary.png", dpi=150)
plt.close()

# Action distribution heatmap
if not ppo_result.actions_df.empty:
    ppo_actions = ppo_result.actions_df.groupby(
        ["pattern", "concurrency"]
    )["count"].sum().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ppo_actions.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(ppo_actions.columns)))
    ax.set_xticklabels(ppo_actions.columns)
    ax.set_yticks(range(len(ppo_actions.index)))
    ax.set_yticklabels(ppo_actions.index)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Pattern")
    ax.set_title("PPO Action Distribution (Pattern × Concurrency)")
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_action_heatmap.png", dpi=150)
    plt.close()

logger.info(f"Saved comparison plots to {output_dir}")
```

def plot_training_history(
history: Dict[str, List],
output_dir: Path,
) -> None:
“”“Plot training curves.”””
if not MATPLOTLIB_AVAILABLE:
return

```
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Episode rewards
if history["episode_rewards"]:
    ax = axes[0, 0]
    rewards = history["episode_rewards"]
    ax.plot(rewards, alpha=0.3, label="Raw")
    # Smoothed
    window = min(100, len(rewards) // 10 + 1)
    if window > 1:
        smoothed = pd.Series(rewards).rolling(window).mean()
        ax.plot(smoothed, label=f"Smoothed ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend()

# Episode lengths
if history["episode_lengths"]:
    ax = axes[0, 1]
    ax.plot(history["episode_lengths"], alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Lengths")

# CEs per episode
if history["episode_ces"]:
    ax = axes[1, 0]
    ax.plot(history["episode_ces"], alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("CEs")
    ax.set_title("CEs per Episode")

# UE rate over time
if history["episode_ues"]:
    ax = axes[1, 1]
    ue_binary = [1 if u > 0 else 0 for u in history["episode_ues"]]
    window = min(100, len(ue_binary) // 10 + 1)
    if window > 1:
        ue_rate = pd.Series(ue_binary).rolling(window).mean()
        ax.plot(ue_rate)
    ax.set_xlabel("Episode")
    ax.set_ylabel("UE Rate")
    ax.set_title(f"UE Discovery Rate (rolling {window})")

plt.tight_layout()
plt.savefig(output_dir / "training_history.png", dpi=150)
plt.close()
```

# =============================================================================

# SECTION 8: TC LIBRARY GENERATION

# =============================================================================

def generate_tc_library(
result: EvaluationResult,
action_library: ActionLibrary,
output_dir: Path,
) -> pd.DataFrame:
“””
Generate ranked test case library from evaluation results.

```
Ranks actions by their effectiveness in inducing errors.
"""
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if result.actions_df.empty:
    logger.warning("No action data for TC library generation")
    return pd.DataFrame()

# Aggregate action statistics
action_stats = []

for aid in result.actions_df["action_id"].unique():
    action = action_library[int(aid)]
    action_data = result.actions_df[result.actions_df["action_id"] == aid]
    
    # Find episodes where this action led to errors
    total_uses = action_data["count"].sum()
    
    # Cross-reference with episode data to see if action correlated with errors
    episodes_with_action = set(action_data["episode"].unique())
    episodes_df = result.episodes_df[result.episodes_df["episode"].isin(episodes_with_action)]
    
    ce_episodes = len(episodes_df[episodes_df["total_ce"] > 0])
    ue_episodes = len(episodes_df[episodes_df["ue_found"] == 1])
    
    action_stats.append({
        "action_id": aid,
        "pattern": action.pattern,
        "region": action.region,
        "concurrency": action.concurrency,
        "access_pattern": action.access_pattern,
        "block_size": action.block_size,
        "duration_s": action.duration_s,
        "stress_score": action.stress_score,
        "total_uses": total_uses,
        "ce_correlation": ce_episodes / max(len(episodes_with_action), 1),
        "ue_correlation": ue_episodes / max(len(episodes_with_action), 1),
        "effectiveness_score": (
            0.3 * (ce_episodes / max(len(episodes_with_action), 1)) +
            0.7 * (ue_episodes / max(len(episodes_with_action), 1)) +
            0.1 * min(action.stress_score / 5.0, 1.0)
        ),
        "gsat_command": action.to_gsat_command(),
    })

tc_df = pd.DataFrame(action_stats)
tc_df = tc_df.sort_values("effectiveness_score", ascending=False)

# Save
tc_df.to_csv(output_dir / "tc_library_ranked.csv", index=False)

# Also save top 20 as JSON for easy use
top_tcs = tc_df.head(20).to_dict("records")
with open(output_dir / "top_test_cases.json", "w") as f:
    json.dump(top_tcs, f, indent=2, cls=NumpyJSONEncoder)

logger.info(f"Generated TC library with {len(tc_df)} actions")
return tc_df
```

# =============================================================================

# SECTION 9: MAIN CLI

# =============================================================================

def create_parser() -> argparse.ArgumentParser:
“”“Create argument parser.”””
parser = argparse.ArgumentParser(
description=“DRAM-SNIFFER: DDR Error-Inducing TC Generation via RL”,
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog=”””
Examples:

# Phase 0: Quick simulation training

python dram_sniffer_v3.py –mode train_eval –backend synthetic_fast –timesteps 100000

# Phase 0: Full simulation training with evaluation

python dram_sniffer_v3.py –mode train_eval –backend synthetic_fast –timesteps 500000 –episodes 200

# Phase 1: Real hardware (requires Linux with stressapptest and EDAC)

python dram_sniffer_v3.py –mode train_eval –backend gsat_wrapper –require-edac –timesteps 50000

# Evaluation only with existing model

python dram_sniffer_v3.py –mode eval –model-path runs/model_final.zip –episodes 100

# Generate config template

python dram_sniffer_v3.py –generate-config config.json
“””,
)

```
# Mode selection
parser.add_argument(
    "--mode",
    choices=["train", "eval", "train_eval"],
    default="train_eval",
    help="Operation mode (default: train_eval)",
)

# Configuration
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to JSON config file",
)
parser.add_argument(
    "--generate-config",
    type=str,
    default=None,
    metavar="PATH",
    help="Generate default config template and exit",
)

# Output
parser.add_argument(
    "--output-dir",
    type=str,
    default="runs/dram_sniffer",
    help="Output directory (default: runs/dram_sniffer)",
)

# Training parameters
parser.add_argument(
    "--timesteps",
    type=int,
    default=None,
    help="Total training timesteps",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed",
)

# Evaluation parameters
parser.add_argument(
    "--episodes",
    type=int,
    default=None,
    help="Number of evaluation episodes",
)
parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Path to trained model for evaluation",
)

# Backend selection
parser.add_argument(
    "--backend",
    choices=["synthetic_fast", "synthetic_timed", "gsat_wrapper"],
    default=None,
    help="Execution backend",
)
parser.add_argument(
    "--action-granularity",
    choices=["coarse", "medium", "full"],
    default=None,
    help="Action space granularity",
)

# GSAT backend options
parser.add_argument(
    "--gsat-binary",
    type=str,
    default=None,
    help="Path to stressapptest binary",
)
parser.add_argument(
    "--use-numactl",
    action="store_true",
    help="Use numactl for NUMA binding",
)
parser.add_argument(
    "--require-edac",
    action="store_true",
    help="Require EDAC availability",
)
parser.add_argument(
    "--gsat-timeout",
    type=int,
    default=None,
    help="GSAT command timeout in seconds",
)

# Synthetic backend options
parser.add_argument(
    "--sleep-scale",
    type=float,
    default=None,
    help="Sleep scale for synthetic_timed backend",
)
parser.add_argument(
    "--fixed-weak-map",
    action="store_true",
    help="Use fixed vulnerability map across episodes",
)

# Reward tuning
parser.add_argument(
    "--alpha-ce",
    type=float,
    default=None,
    help="CE reward weight",
)
parser.add_argument(
    "--beta-ue",
    type=float,
    default=None,
    help="UE reward weight",
)
parser.add_argument(
    "--gamma-coverage",
    type=float,
    default=None,
    help="Coverage bonus weight",
)

# Misc
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Verbosity level (0-2)",
)
parser.add_argument(
    "--no-plots",
    action="store_true",
    help="Skip plot generation",
)

return parser
```

def main() -> None:
“”“Main entry point.”””
parser = create_parser()
args = parser.parse_args()

```
# Handle config generation
if args.generate_config:
    cfg = DRAMSnifferConfig()
    cfg.save(args.generate_config)
    print(f"Generated config template: {args.generate_config}")
    return

# Load or create config
if args.config:
    cfg = DRAMSnifferConfig.load(args.config)
else:
    cfg = DRAMSnifferConfig()

# Apply CLI overrides
if args.timesteps is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "total_timesteps": args.timesteps})
if args.episodes is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "eval_episodes": args.episodes})
if args.backend is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "backend": args.backend})
if args.action_granularity is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "action_granularity": args.action_granularity})
if args.gsat_binary is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "gsat_binary": args.gsat_binary})
if args.use_numactl:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "use_numactl": True})
if args.require_edac:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "require_edac": True})
if args.gsat_timeout is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "gsat_timeout_s": args.gsat_timeout})
if args.sleep_scale is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "sleep_scale": args.sleep_scale})
if args.fixed_weak_map:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "fixed_weak_map": True})
if args.alpha_ce is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "alpha_ce": args.alpha_ce})
if args.beta_ue is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "beta_ue": args.beta_ue})
if args.gamma_coverage is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "gamma_coverage": args.gamma_coverage})
if args.verbose is not None:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "verbose": args.verbose})
if args.no_plots:
    cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "generate_plots": False})

cfg = DRAMSnifferConfig(**{**cfg.to_dict(), "output_dir": args.output_dir})

# Setup output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save effective config
cfg.save(output_dir / "config.json")

# Save action library
action_library = ActionLibrary(cfg)
action_library.save(output_dir / "action_library.csv")

logger.info(f"DRAM-SNIFFER v3.0")
logger.info(f"Mode: {args.mode}")
logger.info(f"Backend: {cfg.backend}")
logger.info(f"Action space size: {len(action_library)}")
logger.info(f"Output directory: {output_dir}")

model = None
callback = None

# Training phase
if args.mode in ("train", "train_eval"):
    if not SB3_AVAILABLE:
        logger.error(f"stable-baselines3 required for training: {SB3_IMPORT_ERROR}")
        sys.exit(1)
    
    model, callback = train_agent(cfg, output_dir, seed=args.seed)
    
    # Save training history
    if callback:
        history = callback.get_training_history()
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, cls=NumpyJSONEncoder)
        
        if cfg.generate_plots:
            plot_training_history(history, output_dir)

# Evaluation phase
if args.mode in ("eval", "train_eval"):
    if model is None:
        if args.model_path is None:
            logger.error("--model-path required for eval mode")
            sys.exit(1)
        if not SB3_AVAILABLE:
            logger.error(f"stable-baselines3 required: {SB3_IMPORT_ERROR}")
            sys.exit(1)
        
        # Load model
        env = DRAMSnifferEnv(cfg)
        model = PPO.load(args.model_path, env=env)
        logger.info(f"Loaded model from {args.model_path}")
    
    # Evaluate PPO agent
    logger.info(f"Evaluating PPO agent ({cfg.eval_episodes} episodes)...")
    ppo_result = evaluate_agent(
        model, cfg, cfg.eval_episodes, args.seed, "ppo",
        deterministic=cfg.deterministic_eval,
    )
    ppo_result.save(output_dir)
    
    # Evaluate random baseline
    logger.info(f"Evaluating random baseline ({cfg.eval_episodes} episodes)...")
    random_result = evaluate_agent(
        None, cfg, cfg.eval_episodes, args.seed + 10000, "random",
    )
    random_result.save(output_dir)
    
    # Generate comparison plots
    if cfg.generate_plots:
        plot_comparison(ppo_result, random_result, output_dir)
    
    # Generate TC library
    tc_df = generate_tc_library(ppo_result, action_library, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    summary = {
        "metric": [],
        "PPO": [],
        "Random": [],
        "Improvement": [],
    }
    
    metrics = [
        ("mean_steps_to_first_ce", "Steps to First CE", True),
        ("mean_steps_to_first_ue", "Steps to First UE", True),
        ("mean_total_ce", "Total CEs", False),
        ("ue_rate", "UE Discovery Rate", False),
        ("mean_coverage", "Region Coverage", False),
        ("mean_entropy", "Action Entropy", True),
    ]
    
    for key, name, lower_better in metrics:
        ppo_v = ppo_result.summary[key]
        rnd_v = random_result.summary[key]
        
        if lower_better:
            improvement = (rnd_v - ppo_v) / max(rnd_v, 1e-9) * 100
        else:
            improvement = (ppo_v - rnd_v) / max(rnd_v, 1e-9) * 100
        
        summary["metric"].append(name)
        summary["PPO"].append(f"{ppo_v:.3f}")
        summary["Random"].append(f"{rnd_v:.3f}")
        summary["Improvement"].append(f"{improvement:+.1f}%")
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    # Save combined summary
    combined_summary = {
        "ppo": ppo_result.summary,
        "random": random_result.summary,
        "improvements": {
            "steps_to_first_ce": random_result.summary["mean_steps_to_first_ce"] / max(ppo_result.summary["mean_steps_to_first_ce"], 1e-9),
            "steps_to_first_ue": random_result.summary["mean_steps_to_first_ue"] / max(ppo_result.summary["mean_steps_to_first_ue"], 1e-9),
            "ce_improvement": ppo_result.summary["mean_total_ce"] / max(random_result.summary["mean_total_ce"], 1e-9),
            "ue_rate_improvement": ppo_result.summary["ue_rate"] / max(random_result.summary["ue_rate"], 1e-9),
        },
        "action_space_size": len(action_library),
        "config": cfg.to_dict(),
    }
    
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(combined_summary, f, indent=2, cls=NumpyJSONEncoder)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Top test cases: {output_dir / 'top_test_cases.json'}")
    print(f"TC library: {output_dir / 'tc_library_ranked.csv'}")

logger.info("DRAM-SNIFFER complete")
```

if **name** == “**main**”:
main()






