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
    d = 0.5
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
    # Параметры
    n_monomers = 1
    volume_fraction = 0.1
    # Объём одной частицы (радиус 0.3)
    particle_vol = (4/3) * np.pi * 0.3**3  # ≈ 0.113
    total_particles = n_monomers * 3  # без инициатора
    box_size = (total_particles * particle_vol / volume_fraction) ** (1/3)
    
    print(f"Число мономеров: {n_monomers}")
    print(f"Всего частиц: {total_particles}")
    print(f"Размер реактора: {box_size:.3f}^3")
    print(f"Объёмная доля (целевая): {volume_fraction:.2f}")
    
    config = ReactorConfig(size=(box_size, box_size, box_size), temperature=1.0)
    reactor = Reactor(config)
    
    # Создаём мономеры со случайными координатами
    np.random.seed(42)  # для воспроизводимости
    next_id = 0
    next_chain = 1
    for _ in range(n_monomers):
        center = np.random.uniform(0.5, box_size, 3)
        parts, bonds = create_triple_monomer(center, next_chain, orientation='x')
        for p in parts:
            p.id = next_id
            reactor.add_particle(p)
            next_id += 1
        # Регистрируем внутримолекулярные связи
        gi = next_id - 3
        gj1 = next_id - 2
        gj2 = next_id - 1
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
        reactor.particles[gi].bonded_to.extend([gj1, gj2])
        reactor.particles[gj1].bonded_to.append(gi)
        reactor.particles[gj2].bonded_to.append(gi)
        next_chain += 1

    reactor.update_grid()
    reactor.plot_grid_slice('z')
    print(f"Фактическое число частиц в реакторе: {len(reactor.particles)}")
    reactor.plot(step=0, show_connections=True)
    # Визуализация
    reactor.plot(step=0, show_connections=True, save_path="initial_packing.png")
    print("Изображение сохранено как initial_packing.png")

if __name__ == "__main__":
    main()