import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import time
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig
from polynetsim.analysis.free_volume import FreeVolumeAnalyzer

# Конфигурация
config = ReactorConfig(size=(20.0, 20.0, 20.0), boundary_conditions="periodic", temperature=1.0)
reactor = Reactor(config)

# Создаём несколько частиц (например, 10 штук) со случайными начальными положениями
np.random.seed(42)
n_particles = 10
particles = []
for i in range(n_particles):
    pos = np.random.rand(3) * config.size
    p = Particle(
        id=i,
        ptype=ParticleType.HDDA_ACRYLATE,
        position=pos,
        velocity=np.zeros(3),
        radius=0.5,
        mass=1.0
    )
    particles.append(p)
reactor.add_particles(particles)

# Запускаем моделирование на некоторое время, записывая траекторию
dt = 0.01
steps = 200  # всего 200 шагов
record_interval = 10  # записываем каждые 10 шагов

positions_history = []
print("Запускаем моделирование для сбора траектории...")
start_time = time.time()
for step in range(steps):
    reactor.velocity_verlet_step(dt, gamma=1.0)
    if step % record_interval == 0:
        positions_history.append(np.array([p.position.copy() for p in reactor.particles]))
print(f"Моделирование завершено за {time.time() - start_time:.2f} с")
print(f"Собрано {len(positions_history)} кадров траектории")

# Анализируем флуктуационный свободный объём
analyzer = FreeVolumeAnalyzer()
radii = np.array([p.radius for p in reactor.particles])
result = analyzer.fluctuational_free_volume(positions_history, radii=radii)

print("\n--- Результаты анализа флуктуационного свободного объёма ---")
print(f"Среднеквадратичное смещение (MSD): {result['mean_squared_displacement']:.6f}")
print(f"Среднеквадратичное отклонение (RMS): {result['rms']:.6f}")
print(f"Объём флуктуаций (сфера RMS): {result['fluctuation_volume']:.6f}")
print(f"Относительный объём флуктуаций (к объёму частицы): {result['relative_fluctuation']:.6f}")
print(f"Среднеквадратичные смещения по компонентам: {result['msd_components']}")

# Выведем MSD для каждой частицы, чтобы увидеть разброс
print("\nMSD по частицам:")
for i, msd in enumerate(result['msd_per_particle']):
    print(f"  Частица {i}: {msd:.6f}")

# Также можем вычислить геометрический свободный объём для сравнения
geo_fv = analyzer.geometric_free_volume(reactor, n_samples=5000)
print(f"\nГеометрический свободный объём (для последнего кадра): {geo_fv:.4f}")

print("\nТест завершён.")