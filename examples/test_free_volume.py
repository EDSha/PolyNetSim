import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig
from polynetsim.analysis.free_volume import FreeVolumeAnalyzer

# Создаём пустой реактор
config = ReactorConfig(size=(10.0, 10.0, 10.0), boundary_conditions="periodic")
reactor = Reactor(config)

# Анализатор с пробной сферой радиуса 0.2
analyzer = FreeVolumeAnalyzer(probe_radius=0.2)

print("1. Пустой реактор:")
fv = analyzer.geometric_free_volume(reactor, n_samples=5000)
print(f"   Свободный объём = {fv:.4f} (должно быть 1.0)")

# Добавляем одну частицу в центре
p = Particle(
    id=1,
    ptype=ParticleType.HDDA_ACRYLATE,
    position=np.array([5.0, 5.0, 5.0]),
    radius=1.0,
    mass=1.0
)
reactor.add_particle(p)

print("\n2. Одна частица радиусом 1.0 в центре:")
fv = analyzer.geometric_free_volume(reactor, n_samples=5000)
print(f"   Свободный объём = {fv:.4f}")

# Теоретическая оценка: объём шара 4/3 π r³ ≈ 4.18879, объём куба 1000, свободная доля ≈ 0.9958
expected = 1 - (4/3 * np.pi * 1.0**3) / 1000
print(f"   Теоретически ≈ {expected:.4f}")

# Добавляем вторую частицу рядом
p2 = Particle(
    id=2,
    ptype=ParticleType.HDDA_ACRYLATE,
    position=np.array([7.0, 5.0, 5.0]),
    radius=1.0,
    mass=1.0
)
reactor.add_particle(p2)

print("\n3. Две частицы:")
fv = analyzer.geometric_free_volume(reactor, n_samples=5000)
print(f"   Свободный объём = {fv:.4f}")

# Построим карту свободного объёма (если matplotlib доступен)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    free_map = analyzer.free_volume_map(reactor, grid=(30,30,30))
    
    # Визуализируем срез по середине (z=15)
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    ax[0].imshow(free_map[:,:,15].T, origin='lower', extent=[0,10,0,10], cmap='gray')
    ax[0].set_title('Срез свободного объёма (z≈5)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    
    # Изолинии потенциала (можно добавить позиции частиц)
    ax[1].scatter([p.position[0], p2.position[0]], [p.position[1], p2.position[1]], 
                  s=100, c='red', label='частицы')
    ax[1].set_xlim(0,10)
    ax[1].set_ylim(0,10)
    ax[1].set_title('Позиции частиц (проекция на xy)')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('free_volume_test.png')
    plt.show()
except ImportError:
    print("Matplotlib не установлен, визуализация пропущена.")

print("\nТест завершён.")