#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple
import threading
import numpy as np
import math
import time


class BaseCommandBridgeABC(ABC):
    """Abstract, ROS-agnostic base for joint/command bridges.

    Responsibilities:
    - Hold ordered joint names and current joint state buffers
    - Provide threshold-checked `sendCommand` orchestration
    - Provide helpers to read/assign joint state

    ROS- or transport-specific details are deferred to overrides.
    """

    def __init__(
        self,
        ordered_joint_names: Iterable[str],
        *,
        threshold: float = 0.05,
    ) -> None:
        self.ordered_joint_names_: List[str] = list(ordered_joint_names)
        self.threshold: float = float(threshold)

        n = len(self.ordered_joint_names_)
        self._state_lock = threading.Lock()
        self.actual_joint_positions_ = np.full(n, np.nan, dtype=float)
        self.actual_joint_velocities_ = np.full(n, np.nan, dtype=float)
        self.actual_joint_efforts_ = np.full(n, np.nan, dtype=float)

    # -------------------------- Abstract hooks --------------------------
    @abstractmethod
    def _do_publish(self, q: np.ndarray) -> None:
        """Transport-specific publish of the command vector."""
        ...

    @abstractmethod
    def getObstacles(self, max_age_sec: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pos[N,3], vel[N,3], acc[N,3]) for obstacles/humans.
        The base class does not implement this; subclasses decide data sources.
        """
        ...

    # -------------------------- Public API --------------------------
    def sendCommand(self, q: np.ndarray) -> None:
        """Threshold-checked command dispatch.

        Raises:
            ValueError if max(|q - current|) exceeds threshold (for known joints).
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.size != len(self.ordered_joint_names_):
            raise ValueError(
                f"q has length {q_arr.size}, but expected {len(self.ordered_joint_names_)}"
            )

        with self._state_lock:
            curr = self.actual_joint_positions_.copy()

        mask = ~np.isnan(curr)
        max_diff = float(np.max(np.abs(q_arr[mask] - curr[mask]))) if np.any(mask) else 0.0

        if max_diff > self.threshold:
            print(f"Command difference {max_diff:.3f} exceeds threshold {self.threshold:.3f}")
            # print current and commanded for debugging
            # for i, name in enumerate(self.ordered_joint_names_):
                # if mask[i]:
                #     # print(
                #     #     f"  Joint '{name}': current={curr[i]:.4f}, commanded={q_arr[i]:.4f}, "
                #     #     f"diff={abs(q_arr[i]-curr[i]):.4f}"
                #     # )
                # else:
                #     print(f"  Joint '{name}': current=NaN, commanded={q_arr[i]:.4f}")
            raise ValueError(
                f"Command difference {max_diff:.3f} exceeds threshold {self.threshold:.3f}"
            )

        self._do_publish(q_arr)

    # -------------------------- State helpers --------------------------
    def map_joint_state(
        self,
        names: Iterable[str],
        positions: Iterable[float],
        velocities: Optional[Iterable[float]] = None,
        efforts: Optional[Iterable[float]] = None,
    ) -> None:
        """Generic state updater you can call from any transport (e.g., ROS, custom)."""
        name_to_idx = {name: i for i, name in enumerate(list(names))}

        # Prepare source arrays
        pos_src = list(positions) if positions is not None else []
        vel_src = list(velocities) if velocities is not None else []
        eff_src = list(efforts) if efforts is not None else []

        pos = np.full_like(self.actual_joint_positions_, np.nan)
        vel = np.full_like(self.actual_joint_velocities_, np.nan)
        eff = np.full_like(self.actual_joint_efforts_, np.nan)

        for j, joint in enumerate(self.ordered_joint_names_):
            idx = name_to_idx.get(joint)
            if idx is None:
                continue
            if idx < len(pos_src):
                pos[j] = float(pos_src[idx])
            if idx < len(vel_src):
                vel[j] = float(vel_src[idx])
            if idx < len(eff_src):
                eff[j] = float(eff_src[idx])

        with self._state_lock:
            self.actual_joint_positions_[:] = pos
            self.actual_joint_velocities_[:] = vel
            self.actual_joint_efforts_[:] = eff

    def getPositions(self) -> np.ndarray:
        with self._state_lock:
            return self.actual_joint_positions_.copy()

    def getVelocities(self) -> np.ndarray:
        with self._state_lock:
            return self.actual_joint_velocities_.copy()

    def getEfforts(self) -> np.ndarray:
        with self._state_lock:
            return self.actual_joint_efforts_.copy()

    def getJointPosition(self, name: str) -> float:
        idx = self._index_of(name)
        with self._state_lock:
            return float(self.actual_joint_positions_[idx])

    def getJointVelocity(self, name: str) -> float:
        idx = self._index_of(name)
        with self._state_lock:
            return float(self.actual_joint_velocities_[idx])

    def getJointEffort(self, name: str) -> float:
        idx = self._index_of(name)
        with self._state_lock:
            return float(self.actual_joint_efforts_[idx])

    def wait_for_first_state(self, joint_name: str, timeout: float = 5.0) -> float:
        """Utility that blocks until a non-NaN position arrives (or timeout)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            val = self.getJointPosition(joint_name)
            if not math.isnan(val):
                return val
            time.sleep(0.02)
        return self.getJointPosition(joint_name)

    # -------------------------- Private --------------------------
    def _index_of(self, name: str) -> int:
        try:
            return self.ordered_joint_names_.index(name)
        except ValueError as e:
            raise KeyError(f"Unknown joint name: {name}") from e
