"""
Модуль coarse-grained частицы.
"""
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from typing import List


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
    # simple model
    SIM_MONOMER = "SIM_MONO"   # упрощённый мономер
    SIM_RADICAL = "SIM_RAD"    # упрощённый радикал
    SIM_INERT = "SIM_INERT"    # упрощённое инертное звено цепи
    INITIATOR = "INIT"       # молекула инициатора

    SIM_VINYL = "SIM_VINYL"           # винильная группа (активная)
    SIM_BACKBONE = "SIM_BACKBONE"     # остовная группа (инертная)
    SIM_RADICAL_SLOW = "SIM_RAD_S"    # медленный радикал

    # Можно пока оставить SIM_MONOMER, SIM_RADICAL, но для ясности заведём новые.

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
    bonded_to: List[int] = field(default_factory=list)  # индексы связанных частиц
    chain_id: int = -1      # идентификатор полимерной цепи (-1 = не назначен)
    is_free: bool = True    # True для частиц свободного мономера


    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)

    @property
    def volume(self) -> float:
        """Объём частицы (сфера)."""
        return (4/3) * np.pi * self.radius**3