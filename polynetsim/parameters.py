"""
Модуль с физическими параметрами coarse-grained моделей.
Все значения в условных единицах (нм, масса в а.е.м., энергия в кДж/моль или усл. ед.)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class LJParameters:
    """Параметры потенциала Леннард-Джонса для всех частиц (пока общие)."""
    epsilon: float = 1.0       # глубина ямы
    sigma: float = 1.0          # эффективный диаметр (нм)
    cutoff_ratio: float = 2.5   # обрезание = sigma * cutoff_ratio

@dataclass
class BondParameters:
    """Параметры гармонической связи (общие для всех связей)."""
    stiffness: float = 100.0  # в единицах ε/σ²
    length: float = 1.0        # в единицах σ

@dataclass
class ParticleTypeParams:
    """Параметры для каждого типа CG-частиц."""
    mass: float = 1.0
    radius: float = 0.3
    # можно добавить другие характеристики (заряд, цвет для визуализации)

# Словарь параметров для разных типов частиц
PARTICLE_PARAMS: Dict[str, ParticleTypeParams] = {
    'HDDA_ACRYLATE': ParticleTypeParams(mass=1.2, radius=0.32),
    'HDDA_BACKBONE': ParticleTypeParams(mass=0.8, radius=0.28),
    'DAIF_ALLYL':    ParticleTypeParams(mass=1.1, radius=0.30),
    'DAIF_AROMATIC': ParticleTypeParams(mass=1.5, radius=0.35),
    'RADICAL':       ParticleTypeParams(mass=1.0, radius=0.30),  # примерно как акрилат
    # ... и так далее
}

@dataclass
class ReactionParameters:
    """Кинетические параметры реакций (пока заглушка)."""
    k_initiation: float = 0.01   # константа инициирования (в единицах 1/τ)
    k_propagation: float = 1.0    # константа роста (в единицах 1/τ)
    reaction_radius: float = 2.0  # Радиус захвата радикала (в единицах σ)
    avg_monomer_radius: float = 0.3  # средний радиус CG-частицы мономера (в нм)
    # позже добавим другие

@dataclass
class ModelParameters:
    """Сводные параметры модели."""
    lj: LJParameters = field(default_factory=LJParameters)
    bond: BondParameters = field(default_factory=BondParameters)
    reaction: ReactionParameters = field(default_factory=ReactionParameters)
    particle_params: Dict[str, ParticleTypeParams] = field(default_factory=lambda: PARTICLE_PARAMS)