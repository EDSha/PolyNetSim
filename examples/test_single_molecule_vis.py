import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # необходимо для проекции 3d
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
    # Конфигурация реактора
    config = ReactorConfig(size=(20.0,20.0,20.0), temperature=1.0)
    config.params.lj.epsilon = 0.0  # отключаем Леннард-Джонс для чистоты
    reactor = Reactor(config)
    
    # Создаём одну молекулу в центре
    center = np.array([10.0, 10.0, 10.0])
    particles, bonds = create_triple_monomer(center, chain_id=1, orientation='x')
    
    # Добавляем молекулу через add_molecule (регистрирует частицы и связи)
    reactor.add_molecule(particles, bonds)
    
    # Добавим небольшие случайные скорости (не влияют на визуализацию)
    for p in reactor.particles:
        p.velocity = np.random.normal(0, 0.1, 3)
    
    print("Одна молекула из трёх частиц:")
    for p in reactor.particles:
        print(f"  {p.ptype} at {p.position}")
    
    print(f"Количество связей: {len(reactor.bonds)}")
    print(f"bond_set: {reactor.bond_set}")
    print("\nИнтерактивное окно с 3D-визуализацией.")
    print("Используйте мышь для вращения и масштабирования.")
    
    # Визуализация (без save_path, чтобы открылось интерактивное окно)
    reactor.plot(step=0, show_connections=True)

if __name__ == "__main__":
    main()