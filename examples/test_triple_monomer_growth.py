import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

def create_triple_monomer(center, chain_id, orientation='x'):
    """Создаёт мономер из трёх частиц: остов и две винильные группы."""
    d = 0.5  # расстояние в единицах σ
    if orientation == 'x':
        pos0 = center
        pos1 = center + np.array([ d, 0, 0])
        pos2 = center + np.array([-d, 0, 0])
    else:
        pos0 = center
        pos1 = center + np.array([0, d, 0])
        pos2 = center + np.array([0,-d, 0])
    particles = [
        Particle(id=0, ptype=ParticleType.SIM_BACKBONE, position=pos0,
                 radius=0.3, mass=1.0, chain_id=chain_id, is_free=True),
        Particle(id=1, ptype=ParticleType.SIM_VINYL,   position=pos1,
                 radius=0.3, mass=1.0, chain_id=chain_id, is_free=True),
        Particle(id=2, ptype=ParticleType.SIM_VINYL,   position=pos2,
                 radius=0.3, mass=1.0, chain_id=chain_id, is_free=True),
    ]
    bonds = [(0,1), (0,2)]
    return particles, bonds

def main():
    # Безразмерные параметры
    config = ReactorConfig(size=(20.0, 20.0, 20.0), temperature=1.0)
    
    config.params.lj.epsilon = 0.0
    config.params.lj.sigma = 1.0
    
    config.params.bond.stiffness = 1000.0   # жёсткая связь
    config.params.bond.length = 1.0        # длина связи = 1σ
    
    config.params.reaction.k_initiation = 10.0
    config.params.reaction.k_propagation = 1.0
    config.params.reaction.reaction_radius = 1.5  # немного больше длины связи
    
    reactor = Reactor(config)

    # Добавляем один инициатор в центр
    init = Particle(id=0, ptype=ParticleType.INITIATOR,
                    position=[10.0,10.0,10.0], velocity=np.zeros(3),
                    mass=1.0, radius=0.3, chain_id=-1, is_free=False)
    reactor.add_particle(init)

    # Добавляем 3 мономера (каждый с уникальным chain_id)
    next_id = 1
    next_chain = 1
    for i in range(3):
        angle = 2*np.pi*i/3
        center = np.array([10.0 + 0.8*np.cos(angle), 10.0 + 0.8*np.sin(angle), 10.0])
        parts, bonds = create_triple_monomer(center, next_chain, orientation='x')
        
        # Добавляем частицы в реактор
        for p in parts:
            p.id = next_id
            reactor.add_particle(p)
            next_id += 1

        for p in reactor.particles:
            p.velocity = np.random.normal(0, np.sqrt(1.0), 3)  # дисперсия = kT/m, m=1
        
        # Добавляем внутримолекулярные связи вручную (зная глобальные индексы)
        # Индексы: backbone = next_id-3, vinyl1 = next_id-2, vinyl2 = next_id-1
        gi = next_id - 3
        gj1 = next_id - 2
        gj2 = next_id - 1
        # Добавляем в список bonds и bond_set
        reactor.bonds.append((gi, gj1))
        reactor.bonds.append((gi, gj2))
        if gi < gj1:
            reactor.bond_set.add((gi, gj1))
        else:
            reactor.bond_set.add((gj1, gi))
        if gi < gj2:
            reactor.bond_set.add((gi, gj2))
        else:
            reactor.bond_set.add((gj2, gi))
        # Заполняем bonded_to
        reactor.particles[gi].bonded_to.extend([gj1, gj2])
        reactor.particles[gj1].bonded_to.append(gi)
        reactor.particles[gj2].bonded_to.append(gi)
        
        next_chain += 1

    print("Начальные частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")

    reactor.plot(step=0, save_path="initial.png")

    # Параметры интегрирования
    dt = 0.001          # уменьшенный шаг
    steps = 1500
    record_interval = 50
    gamma = 0.0         # включаем термостат

    for step in range(steps):
        reactor.velocity_verlet_step(dt, gamma=gamma)
        reactor.react(dt)

        if step % record_interval == 0:
            n_init = sum(1 for p in reactor.particles if p.ptype==ParticleType.INITIATOR)
            n_rad = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_RADICAL)
            n_rad_s = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_RADICAL_SLOW)
            n_vinyl = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_VINYL)
            n_inert = sum(1 for p in reactor.particles if p.ptype==ParticleType.SIM_INERT)
            ke = 0.5 * sum(p.mass * np.dot(p.velocity, p.velocity) for p in reactor.particles)
            pe = 0.0
            for i, j in reactor.bonds:
                delta = reactor.particles[i].position - reactor.particles[j].position
                # учёт PBC при желании
                r = np.linalg.norm(delta)
                pe += 0.5 * config.params.bond.stiffness * (r - config.params.bond.length)**2
            total = ke + pe
            print(f"Шаг {step}: KE={ke:.4f}, PE={pe:.4f}, E={total:.4f}")
            reactor.plot(step=step)

    print("\nИтоговые частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")

if __name__ == "__main__":
    main()