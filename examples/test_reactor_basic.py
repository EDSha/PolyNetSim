import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

# Создаём конфигурацию
config = ReactorConfig(size=(20.0, 20.0, 20.0), name="test_reactor")

# Создаём реактор
reactor = Reactor(config)

# Создаём несколько частиц
p1 = Particle(id=1, ptype=ParticleType.HDDA_ACRYLATE, position=np.array([1.0, 2.0, 3.0]))
p2 = Particle(id=2, ptype=ParticleType.RADICAL, position=np.array([4.0, 5.0, 6.0]))
p3 = Particle(id=3, ptype=ParticleType.HDDA_BACKBONE, position=np.array([7.0, 8.0, 9.0]))

# Добавляем частицы в реактор
reactor.add_particles([p1, p2, p3])

# Выводим информацию
print(reactor.summary())
print("Первая частица:", reactor.particles[0])

# Тестируем граничные условия
pos = np.array([21.0, -1.0, 10.0])
corrected = reactor.apply_boundary_conditions(pos)
print(f"Исходная позиция: {pos}, после BC: {corrected}")