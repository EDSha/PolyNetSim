import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

def create_triple_monomer(center, chain_id, orientation='x'):
    d = 0.4
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
    config = ReactorConfig(size=(20.0,20.0,20.0), temperature=1.0)
    config.params.reaction.k_initiation = 10.0   # высокая, чтобы сразу распался
    config.params.reaction.k_propagation = 10.0
    config.params.reaction.reaction_radius = 0.8
    # Увеличим жёсткость связей для теста
    config.params.bond.stiffness = 1000.0
    reactor = Reactor(config)

    # Добавляем один инициатор в центр
    init = Particle(id=0, ptype=ParticleType.INITIATOR,
                    position=[10.0,10.0,10.0], velocity=np.zeros(3),
                    mass=1.0, radius=0.3, chain_id=-1, is_free=False)
    reactor.add_particle(init)

    # Добавляем 3 мономера вокруг (каждый с уникальным chain_id)
    next_id = 1
    next_chain = 1
    for i in range(3):
        angle = 2*np.pi*i/3
        center = np.array([10.0 + 0.8*np.cos(angle), 10.0 + 0.8*np.sin(angle), 10.0])
        parts, bonds = create_triple_monomer(center, next_chain, orientation='x')
        for p in parts:
            p.id = next_id
            reactor.add_particle(p)
            next_id += 1
        # Добавляем внутримолекулярные связи
        for b in bonds:
            gi = parts[0].id + b[0]  # здесь нужно аккуратно, проще использовать индексы после добавления
        # Но проще: после добавления всех частиц, мы знаем их глобальные индексы. 
        # В данной реализации мы не можем легко получить индексы, поэтому лучше добавить связи через add_molecule, но у нас нет такого метода для трёхчастичных.
        # Временно пропустим связи? Нет, они важны.
        # Переделаем: будем добавлять мономеры через add_molecule, но для этого нужно доработать функцию create_triple_monomer, чтобы она возвращала список частиц и связи, а затем использовать reactor.add_molecule.
        # Но в reactor.add_molecule уже есть код для добавления связей. Давайте использовать его.
        # Для этого нужно, чтобы частицы имели правильные локальные id. У нас они заданы как 0,1,2, но при добавлении через add_molecule они получат глобальные.
        # Поэтому проще создать функцию, которая возвращает частицы и связи, и затем вызвать reactor.add_molecule.
        # Но сейчас у нас нет такой функции в тесте, поэтому временно добавим мономеры по одному и связи вручную.

    # Однако это усложняет. Предлагаю упростить: использовать одиночные мономеры (одну частицу) для этого теста, чтобы проверить только рост цепи без внутренней структуры.
    # Но мы уже зашли в трёхчастичные. Можно пока оставить как есть и просто добавить связи вручную после добавления всех частиц, зная их глобальные индексы.

    # Я перепишу тест, чтобы использовать упрощённые мономеры (одна частица) для отладки роста.
    # Это позволит быстрее увидеть проблему.

    # Но чтобы не тратить время, давайте сделаем тест с одной частицей-мономером, как в самом начале, но с новыми типами.

    # Создадим новый тест отдельно.

if __name__ == "__main__":
    main()