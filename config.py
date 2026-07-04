"""Shared experiment configuration for all relaxation schemes."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RunConfig:
    ic: str = "E3"
    output: bool = True
    order: int = 1
    tau: float = 1.0
    T: float = 10000.0
    jump_after_steps: int = 200
    jump_dt: float = 100.0
    jump_tau: float = 1.0
    two_to_single_LM_de: float = 9e-5
    # Automatic two-LM handoff: detect a sustained rise after the minimum
    # energy-decay rate.  This is a cheap proxy for the constraint system
    # becoming ill-conditioned.
    lm_handoff_rise_rtol: float = 1e-4
    lm_handoff_rise_steps: int = 2

    @property
    def is_e3(self):
        return self.ic in ("E3", "E3-positive")

    @property
    def bc(self):
        """Boundary condition fixed by the initial-condition family."""
        if self.is_e3:
            return "periodic"
        if self.ic == "hopf":
            return "closed"
        raise ValueError(f"unknown initial condition: {self.ic}")

    @property
    def periodic(self):
        return self.bc == "periodic"

    @property
    def domain(self):
        if self.ic == "hopf":
            return (8, 8, 20), (8, 8, 10)
        if self.is_e3:
            return (8, 8, 48), (4, 4, 24)
        raise ValueError(f"unknown initial condition: {self.ic}")

    @property
    def dt_init(self):
        return 0.1 if self.is_e3 else 1.0

    @property
    def dirichlet_ids(self):
        return ("on_boundary",) if self.periodic else ("on_boundary", "top", "bottom")


# This is the single place to change settings shared by mixed, lm, and hdiv.
CONFIG = RunConfig(
    ic=os.environ.get("IC", "E3-positive"),
)
