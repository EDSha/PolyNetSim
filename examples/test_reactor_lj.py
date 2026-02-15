import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

# Конфигурация с периодическими границами (большой бокс, чтобы избежать взаимодействия с копиями)
config = ReactorConfig(size=(50.0, 50.0, 50.0), boundary_conditions="periodic")
reactor = Reactor(config)

# Две частицы на расстоянии, меньшем cut-off (например, 2.0 при sigma=1.0, cutoff=2.5)
p1 = Particle(id=1, ptype=ParticleType.HDDA_ACRYLATE,
              position=np.array([25.0, 25.0, 25.0]),
              velocity=np.array([0.0, 0.0, 0.0]),
              mass=1.0)
p2 = Particle(id=2, ptype=ParticleType.HDDA_ACRYLATE,
              position=np.array([27.0, 25.0, 25.0]),  # расстояние 2.0
              velocity=np.array([0.0, 0.0, 0.0]),
              mass=1.0)

reactor.add_particles([p1, p2])

print("Начальные позиции:")
print(f"p1: {p1.position}, p2: {p2.position}")

# Вычисляем силы
forces = reactor.compute_forces()
print("\nСилы на частицах:")
print(f"Force on p1: {forces[0]}")
print(f"Force on p2: {forces[1]}")

# Выполняем один шаг интегрирования, чтобы увидеть изменение позиций
dt = 0.01
reactor.velocity_verlet_step(dt)

print(f"\nПосле одного шага dt={dt}:")
print(f"p1: {p1.position}, скорость: {p1.velocity}")
print(f"p2: {p2.position}, скорость: {p2.velocity}")