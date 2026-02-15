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
    config.params.reaction.k_initiation = 1.0    # достаточно большая, чтобы быстро распадалось
    config.params.reaction.k_propagation = 10.0
    config.params.reaction.reaction_radius = 0.8
    reactor = Reactor(config)
    
    # Добавляем один инициатор в центр
    init = Particle(id=0, ptype=ParticleType.INITIATOR,
                    position=np.array([10.0,10.0,10.0]),
                    velocity=np.zeros(3), mass=1.0, radius=0.3)
    reactor.add_particle(init)
    
    # Добавляем 5 мономеров вокруг
    n_monomers = 5
    for i in range(n_monomers):
        angle = 2*np.pi*i/n_monomers
        pos = np.array([10.0 + 0.6*np.cos(angle),
                        10.0 + 0.6*np.sin(angle),
                        10.0])
        monomer = Particle(id=i+1, ptype=ParticleType.SIM_MONOMER,
                           position=pos, velocity=np.zeros(3),
                           mass=1.0, radius=0.3)
        reactor.add_particle(monomer)
    
    print("Начальные частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")
    
    # Запускаем несколько шагов
    dt = 0.01
    steps = 50
    for step in range(steps):
        reactor.velocity_verlet_step(dt, gamma=0.0)
        reactor.react(dt)
        if step % 10 == 0:
            n_init = sum(1 for p in reactor.particles if p.ptype==ParticleType.INITIATOR)
            n_rad = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_RADICAL)
            n_mon = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_MONOMER)
            print(f"Шаг {step}: инициаторов {n_init}, радикалов {n_rad}, мономеров {n_mon}")
    
    print("\nИтоговые частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")

if __name__ == "__main__":
    main()