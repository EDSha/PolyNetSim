import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
import time
import h5py  # для сохранения снэпшотов
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

def save_snapshot(reactor, filename, step, time, config):
    """
    Сохраняет полное состояние реактора в HDF5 файл.
    """
    with h5py.File(filename, 'a') as f:
        grp = f.create_group(f"step_{step:06d}")
        grp.attrs['time'] = time
        grp.attrs['step'] = step
        grp.attrs['box_size'] = config.size
        
        # Сохраняем частицы
        n = len(reactor.particles)
        pos = np.array([p.position for p in reactor.particles])
        types = [p.ptype.value for p in reactor.particles]
        chain_ids = np.array([p.chain_id for p in reactor.particles])
        is_free = np.array([p.is_free for p in reactor.particles])
        
        grp.create_dataset('positions', data=pos)
        grp.create_dataset('chain_ids', data=chain_ids)
        grp.create_dataset('is_free', data=is_free)
        grp.create_dataset('types', data=np.array(types, dtype='S20'))
        
        # Сохраняем связи
        bonds = np.array(reactor.bonds) if reactor.bonds else np.empty((0,2), dtype=int)
        grp.create_dataset('bonds', data=bonds)
        
        print(f"  Снэпшот шага {step} сохранён в {filename}")

def main():
    start_time = time.time()
    
    # ===== Параметры моделирования =====
    n_monomers = 10
    # Объём одной частицы: 4/3*pi*0.3^3 ≈ 0.113
    # Всего частиц: n_monomers*3 + 1 инициатор
    total_particles = n_monomers * 3 + 1
    # Желаемая объёмная доля (например, 0.4)
    volume_fraction = 0.4
    box_size = (total_particles * 0.113 / volume_fraction) ** (1/3)
    print("box_size = ", box_size)
    
    config = ReactorConfig(size=(box_size, box_size, box_size), temperature=1.0)
    
    # Параметры взаимодействий
    config.params.lj.epsilon = 0.0
    config.params.lj.sigma = 0.3
    config.params.bond.stiffness = 5000.0
    config.params.bond.length = 0.5
    config.params.reaction.k_initiation = 10.0
    config.params.reaction.k_propagation = 10.0
    config.params.reaction.reaction_radius = 0.8
    
    reactor = Reactor(config)
    
    # Инициатор в центре
    init = Particle(id=0, ptype=ParticleType.INITIATOR,
                    position=[box_size/2, box_size/2, box_size/2],
                    velocity=np.zeros(3), mass=1.0, radius=0.3,
                    chain_id=-1, is_free=False)
    reactor.add_particle(init)
    
    # Создаём мономеры
    np.random.seed(42)
    next_id = 1
    next_chain = 1
    for _ in range(n_monomers):
        center = np.random.uniform(0, box_size, 3)
        parts, bonds = create_triple_monomer(center, next_chain, orientation='x')
        for p in parts:
            p.id = next_id
            reactor.add_particle(p)
            next_id += 1
        # Внутримолекулярные связи
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
    
    print(f"Всего частиц: {len(reactor.particles)}")
    print(f"Размер реактора: {box_size:.2f}^3")
    print(f"Объёмная доля частиц (оценка): {len(reactor.particles)*0.113/box_size**3:.3f}")
    
    # Анализатор свободного объёма
    fv_analyzer = FreeVolumeAnalyzer(probe_radius=0.2)
    
    # Параметры интегрирования
    dt = 0.0002
    steps = 1000
    record_interval = 50
    gamma = 0.05
    
    # ===== НОВОЕ: Файлы для логирования =====
    # Текстовый файл для временных рядов
    data_file = open("simulation_data.txt", "w")
    data_file.write("# step\ttime\tFV\tKE\tPE\tn_radicals\tn_inert\tn_vinyl\n")
    
    # HDF5 файл для снэпшотов (будем сохранять каждые 500 шагов, например)
    snapshot_interval = 50
    snapshot_file = "trajectory.h5"
    # Если файл уже существует, удалим или будем дописывать? Для начала перезапишем.
    with h5py.File(snapshot_file, 'w') as f:
        f.attrs['description'] = 'PolyNetSim trajectory'
        f.attrs['dt'] = dt
        f.attrs['box_size'] = config.size
    
    # Списки для графиков
    steps_recorded = []
    free_volume_values = []
    kinetic_energy_values = []
    n_radicals = []
    n_inert = []
    
    print("Запуск моделирования...")
    loop_start = time.time()
    
    for step in range(steps):
        reactor.velocity_verlet_step(dt, gamma=gamma)
        reactor.react(dt)
        
        if step % record_interval == 0:
            # Расчёт энергий и статистики
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
            fv = fv_analyzer.geometric_free_volume(reactor, n_samples=3000)
            
            # Сохраняем в списки для графиков
            steps_recorded.append(step)
            free_volume_values.append(fv)
            kinetic_energy_values.append(ke)
            n_radicals.append(n_rad)
            n_inert.append(n_in)
            
            # Запись в текстовый файл
            current_time = step * dt
            data_file.write(f"{step}\t{current_time:.6f}\t{fv:.6f}\t{ke:.2f}\t{pe:.2f}\t{n_rad}\t{n_in}\t{n_vinyl}\n")
            data_file.flush()  # гарантирует запись на диск
            
            # Вывод в консоль
            print(f"Шаг {step}: KE={ke:.1f}, PE={pe:.1f}, E={total:.1f}, "
                  f"иниц={n_init}, рад={n_rad}, винил={n_vinyl}, инерт={n_in}, FV={fv:.4f}")
        
        # Сохранение снэпшота
        if step % snapshot_interval == 0 and step > 0:
            save_snapshot(reactor, snapshot_file, step, step*dt, config)
    
    # Закрываем файл с данными
    data_file.close()
    
    loop_end = time.time()
    print(f"Моделирование завершено за {loop_end - loop_start:.2f} с")
    print(f"Всего шагов: {steps}")
    print(f"Текстовые данные сохранены в simulation_data.txt")
    print(f"Снэпшоты сохранены в {snapshot_file}")
    
    # Построение графиков
    plt.figure(figsize=(12, 10))
    
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
    xs = [p.position[0] for p in reactor.particles]
    ys = [p.position[1] for p in reactor.particles]
    plt.scatter(xs, ys, c='blue', alpha=0.5, s=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Проекция частиц на XY')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_100_monomers.png', dpi=150)
    plt.show()
    
    total_time = time.time() - start_time
    print(f"Общее время выполнения: {total_time:.2f} с")

if __name__ == "__main__":
    main()