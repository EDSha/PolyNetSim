import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

# Создаём реактор
config = ReactorConfig(size=(20.0, 20.0, 20.0), boundary_conditions="periodic")
reactor = Reactor(config)

# Добавляем одну частицу с начальной скоростью
p = Particle(
    id=1,
    ptype=ParticleType.HDDA_ACRYLATE,
    position=np.array([10.0, 10.0, 10.0]),
    velocity=np.array([1.0, 0.5, 0.0]),
    mass=1.0
)
reactor.add_particle(p)

print("До интеграции:")
print(f"Позиция: {p.position}, Скорость: {p.velocity}")

# Выполняем 10 шагов с dt=0.1
reactor.integrate(steps=10, dt=0.1)

print("\nПосле 10 шагов (dt=0.1):")
print(f"Позиция: {p.position}, Скорость: {p.velocity}")
print(f"Время симуляции: {reactor.time}, шаг: {reactor.step}")