import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

# Добавим новый тип для прореагировавших винилов (Dx), если его нет в ParticleType
# Временно определим его как расширение, но лучше добавить в основной файл.
# Для этого теста мы можем использовать существующий SIM_INERT, но чтобы отличать цветом, 
# создадим псевдоним. Однако для корректной работы нужно, чтобы тип был определён.
# Поэтому перед использованием проверим, есть ли SIM_VINYL_REACTED в ParticleType.
# Если нет, добавим его вручную в этот тест (но это не сработает, так как ParticleType - Enum).
# Поэтому проще использовать SIM_INERT и покрасить его в оранжевый вручную в plot, но метод plot уже имеет style_map.
# Можно модифицировать style_map прямо в тесте, но метод plot находится в reactor.py и использует свою карту.
# Поэтому лучше создать новый тип в основном файле particle.py. 
# Для этого теста я предлагаю временно использовать SIM_INERT, но в plot он чёрный. 
# Чтобы различать, я изменю style_map прямо в reactor.py, добавив оранжевый для SIM_INERT, но это повлияет на все тесты.
# Проще для этого теста создать отдельную визуализацию без использования reactor.plot, но это долго.
# Поэтому я поступлю так: добавлю в reactor.py временно новый тип, но чтобы не усложнять, 
# я просто буду использовать SIM_INERT и понимать, что это Dx. В plot он будет чёрным.
# Для наглядности можно переопределить цвет в plot, но это потребует изменения reactor.py.
# Я предлагаю пользователю самостоятельно добавить в reactor.py в словарь style_map запись для SIM_INERT с оранжевым цветом, если он хочет различать.
# В коде ниже я буду использовать SIM_INERT для Dx, а комментарием укажу, как изменить цвет.

def main():
    # Конфигурация реактора
    config = ReactorConfig(size=(20.0,20.0,20.0), temperature=1.0)
    config.params.lj.epsilon = 0.0  # отключаем Леннард-Джонс
    reactor = Reactor(config)
    
    # Если вы хотите, чтобы прореагировавшие винилы (Dx) отображались оранжевым,
    # добавьте в reactor.py в словарь style_map внутри метода plot строку:
    # ParticleType.SIM_INERT: {'color': 'orange', 'size': 60, 'marker': 'o'},
    # Сейчас SIM_INERT имеет чёрный цвет, но это тоже нормально.

    # Определим координаты с шагом 1.0
    # Основная цепь из прореагировавших винилов (Dx) вдоль X: Dx1, Dx2, Dx3, затем радикал
    dx1 = [0.0, 0.0, 0.0]
    dx2 = [2.0, 0.0, 0.0]
    dx3 = [4.0, 0.0, 0.0]
    rad = [6.0, 0.0, 0.0]

    # Создаём частицы с явными ID
    particles = []
    # Dx1, Dx2, Dx3 – используем SIM_INERT (можно было бы завести отдельный тип, но пока так)
    particles.append(Particle(id=0, ptype=ParticleType.SIM_INERT, position=dx1,
                              radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=1, ptype=ParticleType.SIM_INERT, position=dx2,
                              radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=2, ptype=ParticleType.SIM_INERT, position=dx3,
                              radius=0.3, mass=1.0, chain_id=1, is_free=False))
    # Радикал
    particles.append(Particle(id=3, ptype=ParticleType.SIM_RADICAL, position=rad,
                              radius=0.3, mass=1.0, chain_id=1, is_free=False))

    # Олигомерные блоки X (SIM_BACKBONE) – по два на каждый Dx (верх и низ)
    # Для Dx1
    x1_up = [0.0, 1.0, 0.0]
    #x1_down = [0.0, -1.0, 0.0]
    # Для Dx2
    #x2_up = [2.0, 1.0, 0.0]
    x2_down = [2.0, -1.0, 0.0]
    # Для Dx3
    x3_up = [4.0, 1.0, 0.0]
    #x3_down = [4.0, -1.0, 0.0]

    particles.append(Particle(id=4, ptype=ParticleType.SIM_BACKBONE, position=x1_up, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    #particles.append(Particle(id=5, ptype=ParticleType.SIM_BACKBONE, position=x1_down, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    #particles.append(Particle(id=6, ptype=ParticleType.SIM_BACKBONE, position=x2_up, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=5, ptype=ParticleType.SIM_BACKBONE, position=x2_down, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=6, ptype=ParticleType.SIM_BACKBONE, position=x3_up, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    #particles.append(Particle(id=9, ptype=ParticleType.SIM_BACKBONE, position=x3_down, radius=0.3, mass=1.0, chain_id=1, is_free=False))

    # Винилы D (SIM_VINYL) – по одному на каждый X (продолжение вверх или вниз)
    # Для x1_up
    d1_up = [0.0, 2.0, 0.0]
    # Для x1_down
    #d1_down = [0.0, -2.0, 0.0]
    # Для x2_up
    #d2_up = [2.0, 2.0, 0.0]
    # Для x2_down
    d2_down = [2.0, -2.0, 0.0]
    # Для x3_up
    d3_up = [4.0, 2.0, 0.0]
    # Для x3_down
    #d3_down = [4.0, -2.0, 0.0]

    particles.append(Particle(id=7, ptype=ParticleType.SIM_VINYL, position=d1_up, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    #particles.append(Particle(id=11, ptype=ParticleType.SIM_VINYL, position=d1_down, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    #particles.append(Particle(id=12, ptype=ParticleType.SIM_VINYL, position=d2_up, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=8, ptype=ParticleType.SIM_VINYL, position=d2_down, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    particles.append(Particle(id=9, ptype=ParticleType.SIM_VINYL, position=d3_up, radius=0.3, mass=1.0, chain_id=1, is_free=False))
    #particles.append(Particle(id=15, ptype=ParticleType.SIM_VINYL, position=d3_down, radius=0.3, mass=1.0, chain_id=1, is_free=False))

    # Добавляем все частицы в реактор
    reactor.add_particles(particles)

    # Определяем связи
    bonds = []

    # Связи основной цепи: Dx1-Dx2, Dx2-Dx3, Dx3-радикал
    bonds.append((0, 1))  # dx1-dx2
    bonds.append((1, 2))  # dx2-dx3
    bonds.append((2, 3))  # dx3-rad

    # Связи Dx с X (каждый Dx связан с верхним и нижним X)
    bonds.append((0, 4))  # dx1 - x1_up
    #bonds.append((0, 5))  # dx1 - x1_down
    #bonds.append((1, 6))  # dx2 - x2_up
    bonds.append((1, 5))  # dx2 - x2_down
    bonds.append((2, 6))  # dx3 - x3_up
    #bonds.append((2, 9))  # dx3 - x3_down

    # Связи X с D (каждый X с одним винилом)
    bonds.append((4, 7))  # x1_up - d1_up
    #bonds.append((5, 11))  # x1_down - d1_down
    #bonds.append((6, 12))  # x2_up - d2_up
    bonds.append((5, 8))  # x2_down - d2_down
    bonds.append((6, 9))  # x3_up - d3_up
    #bonds.append((9, 15))  # x3_down - d3_down

    # Регистрируем все связи в реакторе
    for i, j in bonds:
        reactor.bonds.append((i, j))
        if i < j:
            reactor.bond_set.add((i, j))
        else:
            reactor.bond_set.add((j, i))
        reactor.particles[i].bonded_to.append(j)
        reactor.particles[j].bonded_to.append(i)

    # Добавим небольшие случайные скорости (необязательно)
    for p in reactor.particles:
        p.velocity = np.random.normal(0, 0.1, 3)

    print("Трёхмерная полимерная структура с прореагировавшими винилами (Dx), олигомерными блоками (X) и винилами (D)")
    print(f"Всего частиц: {len(reactor.particles)}")
    print(f"Всего связей: {len(reactor.bonds)}")
    print("\nИнтерактивное 3D-окно. Используйте мышь для вращения и масштабирования.")
    print("Примечание: прореагировавшие винилы (Dx) сейчас имеют тип SIM_INERT и отображаются чёрным.")
    print("Чтобы изменить их цвет на оранжевый, добавьте в reactor.py в метод plot в словарь style_map строку:")
    print("ParticleType.SIM_INERT: {'color': 'orange', 'size': 60, 'marker': 'o'},")

    # Визуализация
    reactor.plot(step=0, show_connections=True)

if __name__ == "__main__":
    main()