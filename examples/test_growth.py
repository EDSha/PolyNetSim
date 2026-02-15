import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

def main():
    config = ReactorConfig(size=(20.0,20.0,20.0), temperature=1.0)
    # Настроим параметры реакций
    config.params.reaction.k_propagation = 10.0   # большая константа для теста
    config.params.reaction.reaction_radius = 0.8
    reactor = Reactor(config)
    
    # Создаём один радикал
    radical = Particle(id=0, ptype=ParticleType.SIM_RADICAL,
                       position=np.array([10.0,10.0,10.0]),
                       velocity=np.zeros(3), mass=1.0, radius=0.3)
    reactor.add_particle(radical)
    
    # Создаём несколько мономеров вокруг
    n_monomers = 10
    for i in range(n_monomers):
        pos = np.array([10.0 + 0.5*np.cos(2*np.pi*i/n_monomers),
                        10.0 + 0.5*np.sin(2*np.pi*i/n_monomers),
                        10.0])
        monomer = Particle(id=i+1, ptype=ParticleType.SIM_MONOMER,
                           position=pos, velocity=np.zeros(3),
                           mass=1.0, radius=0.3)
        reactor.add_particle(monomer)
    
    print(f"Начало: радикалов: {sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_RADICAL)}")
    
    # Запускаем несколько шагов
    dt = 0.01
    steps = 100
    for step in range(steps):
        reactor.velocity_verlet_step(dt, gamma=0.0)
        reactor.react(dt)
        if step % 20 == 0:
            n_rad = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_RADICAL)
            n_inert = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_INERT)
            print(f"Шаг {step}: радикалов {n_rad}, инертных {n_inert}")
    
    print("Готово.")

if __name__ == "__main__":
    main()