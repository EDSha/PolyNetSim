import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь поиска модулей
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from polynetsim.core.particle import Particle, ParticleType
import numpy as np

p = Particle(id=1, ptype=ParticleType.HDDA_ACRYLATE, position=np.array([0., 0., 0.]))
print(p)
print("Volume:", p.volume)