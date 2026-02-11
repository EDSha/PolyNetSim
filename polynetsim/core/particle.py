"""
Модуль coarse-grained частицы.
"""
from enum import Enum
from dataclasses import dataclass
import numpy as np

class ParticleType(Enum):
    """Типы CG-частиц для разных мономеров."""
    HDDA_ACRYLATE = "HDDA_A"
    HDDA_BACKBONE = "HDDA_B"
    DAIF_ALLYL    = "DAIF_A"
    DAIF_AROMATIC = "DAIF_R"
    EPOXY_OXIRANE = "EPOXY_O"
    EPOXY_AMINE   = "EPOXY_N"
    RADICAL       = "RAD"
    NANOPARTICLE  = "NP"

@dataclass
class Particle:
    """Coarse-grained частица с положением, скоростью и реакционной способностью."""
    id: int
    ptype: ParticleType
    position: np.ndarray
    velocity: np.ndarray = None
    mass: float = 1.0
    radius: float = 0.5
    functional_groups: int = 1
    is_reactive: bool = True

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)

    @property
    def volume(self) -> float:
        """Объём частицы (сфера)."""
        return (4/3) * np.pi * self.radius**3