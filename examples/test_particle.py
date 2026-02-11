import sys
sys.path.append('..')

from polynetsim.core.particle import Particle, ParticleType
import numpy as np

p = Particle(id=1, ptype=ParticleType.HDDA_ACRYLATE, position=np.array([0.,0.,0.]))
print(p)
print("Volume:", p.volume)