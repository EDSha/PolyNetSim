import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

def create_hdda_molecule(center):
    """
    Создаёт coarse-grained модель молекулы HDDA:
    - центр (тип HDDA_BACKBONE) - 1 частица
    - две акрилатные группы (тип HDDA_ACRYLATE) по бокам
    Возвращает список частиц и список связей (локальные индексы 0,1,2).
    """
    d = 0.5  # расстояние от центра до акрилатной группы в нм
    
    particles = [
        Particle(id=0, ptype=ParticleType.HDDA_BACKBONE,
                 position=center + np.array([0.0, 0.0, 0.0]),
                 radius=0.3, mass=1.0),
        Particle(id=1, ptype=ParticleType.HDDA_ACRYLATE,
                 position=center + np.array([ d, 0.0, 0.0]),
                 radius=0.3, mass=1.0),
        Particle(id=2, ptype=ParticleType.HDDA_ACRYLATE,
                 position=center + np.array([-d, 0.0, 0.0]),
                 radius=0.3, mass=1.0),
    ]
    bonds = [(0, 1), (0, 2)]
    return particles, bonds

def main():
    # Большой бокс, чтобы избежать взаимодействия с периодическими копиями
    config = ReactorConfig(size=(20.0, 20.0, 20.0),
                          boundary_conditions="periodic",
                          temperature=1.0)
    reactor = Reactor(config)
    
    # Добавляем одну молекулу в центр
    center = np.array([10.0, 10.0, 10.0])
    particles, bonds = create_hdda_molecule(center)
    reactor.add_molecule(particles, bonds)
    
    print(f"Добавлена молекула из {len(reactor.particles)} частиц, связей: {len(reactor.bonds)}")
    
    # Параметры интегрирования
    dt = 0.001
    steps = 2000
    record_interval = 20
    
    times = []
    kinetic_energy = []
    bond_energy = []
    distance_01 = []

    # Небольшое случайное смещение одной частицы, чтобы вывести из равновесия
    reactor.particles[1].position[0] += 0.01   # смещаем акрилат на 0.01 нм  ЭТА СТРОКА ДОБАВЛЕНА
    
    print("Запуск моделирования...")
    for step in range(steps):
        # Кинетическая энергия
        ke = 0.5 * sum(p.mass * np.dot(p.velocity, p.velocity) for p in reactor.particles)
        
        # Потенциальная энергия связей (гармоническая)
        pe_bonds = 0.0
        k_bond = 100.0
        r0 = 0.5
        for i, j in reactor.bonds:
            p_i = reactor.particles[i]
            p_j = reactor.particles[j]
            delta = p_j.position - p_i.position
            if reactor.config.boundary_conditions == "periodic":
                box = np.array(reactor.config.size)
                delta = delta - box * np.round(delta / box)
            r = np.linalg.norm(delta)
            pe_bonds += 0.5 * k_bond * (r - r0)**2
        
        if step % record_interval == 0:
            times.append(step * dt)
            kinetic_energy.append(ke)
            bond_energy.append(pe_bonds)
            # Расстояние между частицами 0 и 1
            p0 = reactor.particles[0].position
            p1 = reactor.particles[1].position
            delta = p1 - p0
            if reactor.config.boundary_conditions == "periodic":
                box = np.array(reactor.config.size)
                delta = delta - box * np.round(delta / box)
            dist = np.linalg.norm(delta)
            distance_01.append(dist)
        
        # Шаг интегрирования (без термостата, чтобы видеть чисто динамику связей)
        reactor.velocity_verlet_step(dt, gamma=0.0)
    
    print("Моделирование завершено.")
    print(f"Начальное расстояние 0-1: {distance_01[0]:.4f} нм")
    print(f"Конечное расстояние 0-1:  {distance_01[-1]:.4f} нм")
    print(f"Средняя кинетическая энергия: {np.mean(kinetic_energy):.4f}")
    print(f"Средняя энергия связей: {np.mean(bond_energy):.4f}")
    
    # Графики
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,2,1)
    plt.plot(times, kinetic_energy)
    plt.xlabel('Время')
    plt.ylabel('Кинетическая энергия')
    plt.title('Кинетическая энергия')
    
    plt.subplot(2,2,2)
    plt.plot(times, bond_energy)
    plt.xlabel('Время')
    plt.ylabel('Потенциальная энергия связей')
    plt.title('Энергия связей')
    
    plt.subplot(2,2,3)
    plt.plot(times, distance_01)
    plt.axhline(y=0.5, color='r', linestyle='--', label='равновесное r0')
    plt.xlabel('Время')
    plt.ylabel('Расстояние (нм)')
    plt.title('Расстояние между центром и акрилатной группой')
    plt.legend()
    
    plt.subplot(2,2,4)
    # Визуализация конечной конфигурации (проекция на XY)
    for i, p in enumerate(reactor.particles):
        plt.scatter(p.position[0], p.position[1], s=200, label=f'P{i}' if i==0 else "")
    plt.xlim(9.5, 10.5)
    plt.ylim(9.5, 10.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Конечные позиции (проекция XY)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_molecule.png')
    plt.show()

if __name__ == "__main__":
    main()