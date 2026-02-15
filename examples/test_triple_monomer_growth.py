import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

# def create_triple_monomer(center, pid_start):
#     d = 0.4
#     particles = [
#         Particle(id=pid_start,   ptype=ParticleType.SIM_BACKBONE, position=center, radius=0.3, mass=1.0),
#         Particle(id=pid_start+1, ptype=ParticleType.SIM_VINYL,   position=center+np.array([d,0,0]), radius=0.3, mass=1.0),
#         Particle(id=pid_start+2, ptype=ParticleType.SIM_VINYL,   position=center+np.array([-d,0,0]), radius=0.3, mass=1.0),
#     ]
#     bonds = [(0,1), (0,2)]
#     return particles, bonds

def create_triple_monomer(center, orientation='x'):
    """
    Создаёт мономер из трёх частиц:
    - две винильные группы (SIM_VINYL) по бокам
    - одна остовная (SIM_BACKBONE) в центре
    Возвращает список частиц и список связей (0-1, 0-2).
    """
    d = 0.4  # расстояние от центра до винильных групп
    if orientation == 'x':
        pos0 = center
        pos1 = center + np.array([ d, 0, 0])
        pos2 = center + np.array([-d, 0, 0])
    else:
        # можно добавить другие ориентации
        pos0 = center
        pos1 = center + np.array([0, d, 0])
        pos2 = center + np.array([0,-d, 0])
    
    particles = [
        Particle(id=0, ptype=ParticleType.SIM_BACKBONE, position=pos0, radius=0.3, mass=1.0),
        Particle(id=1, ptype=ParticleType.SIM_VINYL,   position=pos1, radius=0.3, mass=1.0),
        Particle(id=2, ptype=ParticleType.SIM_VINYL,   position=pos2, radius=0.3, mass=1.0),
    ]
    bonds = [(0,1), (0,2)]
    return particles, bonds


def main():
    config = ReactorConfig(size=(20.0,20.0,20.0), temperature=1.0)
    config.params.reaction.k_initiation = 1.0
    config.params.reaction.k_propagation = 10.0
    config.params.reaction.reaction_radius = 0.8
    reactor = Reactor(config)

    # Добавляем один инициатор
    init = Particle(id=0, ptype=ParticleType.INITIATOR,
                    position=[10.0,10.0,10.0], velocity=np.zeros(3),
                    mass=1.0, radius=0.3)
    reactor.add_particle(init)

    # Добавляем несколько мономеров (каждый из 3 частиц)
    n_monomers = 3
    next_id = 1
    for i in range(n_monomers):
        angle = 2*np.pi*i/n_monomers
        center = np.array([10.0 + 0.8*np.cos(angle), 10.0 + 0.8*np.sin(angle), 10.0])
        parts, bonds = create_triple_monomer(center, next_id)
        for p in parts:
            reactor.add_particle(p)
        # Преобразуем локальные индексы связей (0,1,2) в глобальные с учётом next_id
        for b in bonds:
            gi = next_id + b[0]
            gj = next_id + b[1]
            reactor.bonds.append((gi, gj))
            if gi < gj:
                reactor.bond_set.add((gi, gj))
            else:
                reactor.bond_set.add((gj, gi))
            reactor.particles[gi].bonded_to.append(gj)
            reactor.particles[gj].bonded_to.append(gi)
        next_id += 3

    print("Начальные частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")

    dt = 0.01
    steps = 200
    for step in range(steps):
        reactor.velocity_verlet_step(dt, gamma=0.0)
        reactor.react(dt)
        if step % 50 == 0:
            n_init = sum(1 for p in reactor.particles if p.ptype==ParticleType.INITIATOR)
            n_rad_n = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_RADICAL_NORMAL)
            n_rad_s = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_RADICAL_SLOW)
            n_vinyl = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_VINYL)
            n_inert = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_INERT)
            print(f"Шаг {step}: иниц={n_init}, рад_N={n_rad_n}, рад_S={n_rad_s}, винил={n_vinyl}, инерт={n_inert}")

    print("\nИтоговые частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")

if __name__ == "__main__":
    main()