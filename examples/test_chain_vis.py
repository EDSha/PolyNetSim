import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

def main():
    # Конфигурация реактора
    config = ReactorConfig(size=(20.0,20.0,20.0), temperature=1.0)
    config.params.lj.epsilon = 0.0  # отключаем Леннард-Джонс для чистоты
    reactor = Reactor(config)
    
    # Параметры
    d = 0.5  # расстояние от остова до винилов
    
    # Создаём частицы с явными ID
    particles = []
    
    # Мономер 1 (центр в 10,10,10)
    b1 = [10.0, 10.0, 10.0]
    v1_left = [b1[0] - d, b1[1], b1[2]]
    v1_right = [b1[0] + d, b1[1], b1[2]]
    
    particles.append(Particle(id=0, ptype=ParticleType.SIM_BACKBONE, position=b1,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=1, ptype=ParticleType.SIM_VINYL, position=v1_left,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=2, ptype=ParticleType.SIM_VINYL, position=v1_right,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    
    # Мономер 2 (центр в 12,10,10)
    b2 = [12.0, 10.0, 10.0]
    v2_left = [b2[0] - d, b2[1], b2[2]]
    v2_right = [b2[0] + d, b2[1], b2[2]]
    
    particles.append(Particle(id=3, ptype=ParticleType.SIM_BACKBONE, position=b2,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=4, ptype=ParticleType.SIM_VINYL, position=v2_left,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=5, ptype=ParticleType.SIM_VINYL, position=v2_right,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    
    # Мономер 3 (центр в 14,10,10)
    b3 = [14.0, 10.0, 10.0]
    v3_left = [b3[0] - d, b3[1], b3[2]]
    v3_right = [b3[0] + d, b3[1], b3[2]]
    
    particles.append(Particle(id=6, ptype=ParticleType.SIM_BACKBONE, position=b3,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=7, ptype=ParticleType.SIM_VINYL, position=v3_left,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    # Правый винил третьего мономера делаем радикалом
    particles.append(Particle(id=8, ptype=ParticleType.SIM_RADICAL, position=v3_right,
                               radius=0.3, mass=1.0, chain_id=1, is_free=False))
    
    # Добавляем все частицы в реактор (их id уже заданы)
    reactor.add_particles(particles)
    
    # Определяем связи
    bonds = []
    
    # Внутримономерные связи
    # Мономер1: 0-1, 0-2
    bonds.append((0,1))
    bonds.append((0,2))
    # Мономер2: 3-4, 3-5
    bonds.append((3,4))
    bonds.append((3,5))
    # Мономер3: 6-7, 6-8
    bonds.append((6,7))
    bonds.append((6,8))
    
    # Межмономерные связи (полимерная цепь)
    # Между правым винилом мономера1 (id2) и левым винилом мономера2 (id4)
    bonds.append((2,4))
    # Между правым винилом мономера2 (id5) и левым винилом мономера3 (id7)
    bonds.append((5,7))
    
    # Регистрируем связи в реакторе
    for i, j in bonds:
        reactor.bonds.append((i, j))
        if i < j:
            reactor.bond_set.add((i, j))
        else:
            reactor.bond_set.add((j, i))
        reactor.particles[i].bonded_to.append(j)
        reactor.particles[j].bonded_to.append(i)
    
    # Добавим небольшие случайные скорости для красоты (не обязательно)
    for p in reactor.particles:
        p.velocity = np.random.normal(0, 0.1, 3)
    
    print("Полимерная цепь из трёх мономеров с радикалом на конце")
    print(f"Всего частиц: {len(reactor.particles)}")
    print(f"Всего связей: {len(reactor.bonds)}")
    print("\nИнтерактивное 3D-окно. Используйте мышь для вращения и масштабирования.")
    
    # Визуализация
    reactor.plot(step=0, show_connections=True)

if __name__ == "__main__":
    main()