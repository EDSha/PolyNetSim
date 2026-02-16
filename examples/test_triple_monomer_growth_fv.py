import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig
from polynetsim.analysis.free_volume import FreeVolumeAnalyzer

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
    # Конфигурация реактора
    config = ReactorConfig(size=(20.0,20.0,20.0), temperature=1.0)
    
    # Параметры (можно настроить)
    config.params.lj.epsilon = 0.001   # слабое невалентное взаимодействие
    config.params.bond.stiffness = 5000.0
    config.params.bond.length = 0.5
    config.params.reaction.k_initiation = 10.0
    config.params.reaction.k_propagation = 10.0
    config.params.reaction.reaction_radius = 0.8
    
    reactor = Reactor(config)

    # Добавляем один инициатор в центр
    init = Particle(id=0, ptype=ParticleType.INITIATOR,
                    position=[10.0,10.0,10.0], velocity=np.zeros(3),
                    mass=1.0, radius=0.3, chain_id=-1, is_free=False)
    reactor.add_particle(init)

    # Добавляем несколько мономеров (например, 5) с уникальными chain_id
    n_monomers = 5
    next_id = 1
    next_chain = 1
    angles = np.linspace(0, 2*np.pi, n_monomers, endpoint=False)
    for angle in angles:
        center = np.array([10.0 + 1.0*np.cos(angle), 10.0 + 1.0*np.sin(angle), 10.0])
        parts, bonds = create_triple_monomer(center, next_chain, orientation='x')
        
        # Добавляем частицы в реактор с присвоением ID
        for p in parts:
            p.id = next_id
            reactor.add_particle(p)
            next_id += 1
        
        # Добавляем внутримолекулярные связи вручную (зная глобальные индексы)
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

    print("Начальные частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")

    reactor.plot(step=0, save_path="initial.png")

    # Инициализация анализатора свободного объёма
    fv_analyzer = FreeVolumeAnalyzer(probe_radius=0.2)

    # Параметры интегрирования
    dt = 0.0002
    steps = 5000
    record_interval = 50
    gamma = 0.05   # термостат отключён, релаксация будет после реакций

    # Списки для сохранения данных
    steps_recorded = []
    free_volume_values = []
    kinetic_energy_values = []
    n_radicals = []
    n_inert = []

    for step in range(steps):
        reactor.velocity_verlet_step(dt, gamma=gamma)
        reactor.react(dt)

        if step % record_interval == 0:
            # Сбор статистики
            ke = 0.5 * sum(p.mass * np.dot(p.velocity, p.velocity) for p in reactor.particles)
            pe = 0.0
            for i, j in reactor.bonds:
                delta = reactor.particles[i].position - reactor.particles[j].position
                r = np.linalg.norm(delta)
                pe += 0.5 * config.params.bond.stiffness * (r - config.params.bond.length)**2
            total = ke + pe
            n_rad = sum(1 for p in reactor.particles if p.ptype in (ParticleType.SIM_RADICAL, ParticleType.SIM_RADICAL_SLOW))
            n_in = sum(1 for p in reactor.particles if p.ptype == ParticleType.SIM_INERT)
            n_vinyl = sum(1 for p in reactor.particles if p.ptype == ParticleType.SIM_VINYL)
            n_init = sum(1 for p in reactor.particles if p.ptype == ParticleType.INITIATOR)
            print(f"Шаг {step}: KE={ke:.1f}, PE={pe:.1f}, E={total:.1f}, "
                  f"иниц={n_init}, рад={n_rad}, винил={n_vinyl}, инерт={n_in}")
            
            # Расчёт свободного объёма
            fv = fv_analyzer.geometric_free_volume(reactor, n_samples=3000)
            print(f"  Свободный объём = {fv:.4f}")
            steps_recorded.append(step)
            free_volume_values.append(fv)
            kinetic_energy_values.append(ke)
            n_radicals.append(n_rad)
            n_inert.append(n_in)

            
            # Визуализация (можно закомментировать для скорости)
            # reactor.plot(step=step)

    print("\nИтоговые частицы:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")

    # Построение графиков
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,2,1)
    plt.plot(steps_recorded, free_volume_values, 'b-')
    plt.xlabel('Шаг')
    plt.ylabel('Свободный объём (доля)')
    plt.title('Изменение свободного объёма')
    plt.grid(True)
    
    plt.subplot(2,2,2)
    plt.plot(steps_recorded, kinetic_energy_values, 'r-')
    plt.xlabel('Шаг')
    plt.ylabel('Кинетическая энергия')
    plt.title('Кинетическая энергия')
    plt.grid(True)
    
    plt.subplot(2,2,3)
    plt.plot(steps_recorded, n_radicals, 'g-', label='Радикалы')
    plt.plot(steps_recorded, n_inert, 'k-', label='Инертные')
    plt.xlabel('Шаг')
    plt.ylabel('Количество')
    plt.title('Число радикалов и инертных звеньев')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,2,4)
    # Гистограмма конечных позиций (проекция) для визуализации
    xs = [p.position[0] for p in reactor.particles]
    ys = [p.position[1] for p in reactor.particles]
    plt.scatter(xs, ys, c='blue', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Проекция частиц на XY')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('free_volume_evolution.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()