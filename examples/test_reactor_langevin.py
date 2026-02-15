import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

# Конфигурация реактора
config = ReactorConfig(size=(20.0, 20.0, 20.0), boundary_conditions="periodic", temperature=1.0)
reactor = Reactor(config)

# Добавляем одну частицу (для простоты)
p = Particle(
    id=1,
    ptype=ParticleType.HDDA_ACRYLATE,
    position=np.array([10.0, 10.0, 10.0]),
    velocity=np.array([0.0, 0.0, 0.0]),
    mass=1.0
)
reactor.add_particle(p)

# Параметры термостата
gamma = 1.0
dt = 0.01
steps = 5000

print("Запускаем моделирование с термостатом Ланжевена...")
print(f"Температура: {config.temperature}, gamma={gamma}, dt={dt}, шагов={steps}")

# Массивы для записи кинетической энергии и скорости
kinetic_energy = []
velocities = []

for step in range(steps):
    reactor.velocity_verlet_step(dt, gamma)
    v = reactor.particles[0].velocity
    ke = 0.5 * np.dot(v, v)  # масса = 1
    kinetic_energy.append(ke)
    velocities.extend(v)  # собираем все компоненты скорости для гистограммы

# Ожидаемая средняя кинетическая энергия: (3/2) kT = 1.5 * 1.0 = 1.5
avg_ke = np.mean(kinetic_energy)
print(f"\nСредняя кинетическая энергия: {avg_ke:.4f} (ожидается ~1.5)")

# Построим графики для визуальной проверки (если есть matplotlib)
try:
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(kinetic_energy[:500])  # первые 500 шагов
    plt.xlabel('Шаг')
    plt.ylabel('Кинетическая энергия')
    plt.title('Флуктуации кинетической энергии')
    
    plt.subplot(1,2,2)
    plt.hist(velocities, bins=50, density=True, alpha=0.7, label='Моделирование')
    # Теоретическое распределение Максвелла для одной компоненты: гаусс с дисперсией kT/m = 1
    x = np.linspace(-4,4,200)
    theory = np.exp(-x**2/(2)) / np.sqrt(2*np.pi)
    plt.plot(x, theory, 'r-', label='Теория (kT=1)')
    plt.xlabel('Скорость')
    plt.ylabel('Плотность вероятности')
    plt.title('Распределение компонент скорости')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('langevin_test.png')
    plt.show()
except ImportError:
    print("Matplotlib не установлен, пропускаем графики.")

print("Тест завершён.")