"""
Модуль виртуального реактора.
Содержит класс Reactor для хранения частиц, управления геометрией и граничными условиями.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import random
from collections import defaultdict
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.parameters import ModelParameters
from polynetsim.chemistry.reactions import try_propagation, try_initiation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


@dataclass
class ReactorConfig:
    """Конфигурация реактора."""
    size: Tuple[float, float, float] = (10.0, 10.0, 10.0)  # размеры в нм
    boundary_conditions: str = "periodic"  # "periodic" или "reflective"
    temperature: float = 300.0  # Кельвины
    name: str = "default"

    params: ModelParameters = field(default_factory=ModelParameters)  # параметры модели


class Reactor:
    """Виртуальный реактор для моделирования полимеризации."""
    
    def __init__(self, config: ReactorConfig):
        self.config = config
        self.particles = []          # список объектов Particle
        self.time = 0.0               # текущее время симуляции
        self.step = 0                  # номер шага
        self.bonds = []          # список кортежей (i, j) для ковалентных связей
        self.bond_set = set()  # множество кортежей (i, j) с i < j
        
    def add_particle(self, particle):
        """Добавить одну частицу в реактор."""
        self.particles.append(particle)
        
    def add_particles(self, particles: List):
        """Добавить список частиц."""
        self.particles.extend(particles)
    
    def apply_boundary_conditions(self, pos: np.ndarray) -> np.ndarray:
        """
        Применить граничные условия к одной позиции.
        Возвращает скорректированную позицию.
        """
        if self.config.boundary_conditions == "periodic":
            # периодические: позиция приводится к [0, size)
            return pos % np.array(self.config.size)
        elif self.config.boundary_conditions == "reflective":
            # отражающие: при выходе за границу меняем знак скорости (будет реализовано позже)
            # пока просто обрезаем
            size = np.array(self.config.size)
            return np.clip(pos, 0, size)
        else:
            return pos

    def compute_forces(self):
        """
        Вычисление сил, действующих на частицы, с использованием потенциала Леннарда-Джонса.
        Для всех пар частиц рассчитывается сила и добавляется к обеим частицам.
        """
        #print("bond_set содержит:", self.bond_set)  # временно

        n = len(self.particles)
        forces = [np.zeros(3) for _ in range(n)]

            # Параметры потенциала Леннард-Джонса
        lj = self.config.params.lj
        epsilon = lj.epsilon    # глубина ямы
        sigma = lj.sigma         # эффективный диаметр частицы (можно связать с radius)
        cutoff = sigma * lj.cutoff_ratio    # радиус обрезания (для ускорения)

        # Параметры связей
        bond = self.config.params.bond
        k_bond = bond.stiffness
        r0 = bond.length        
 
        
        # Перебор всех уникальных пар
        for i in range(n):
            for j in range(i + 1, n):

                
                # Пропускаем, если частицы связаны химической связью
                if (i, j) in self.bond_set:
                    continue
                
                p_i = self.particles[i]
                p_j = self.particles[j]
                
                delta = p_j.position - p_i.position
                if self.config.boundary_conditions == "periodic":
                    box = np.array(self.config.size)
                    delta = delta - box * np.round(delta / box)
                
                r2 = np.dot(delta, delta)
                if r2 > cutoff * cutoff:
                    continue
                
                # Сила Леннарда-Джонса: F = (48*epsilon/r^2) * ( (sigma/r)^12 - 0.5*(sigma/r)^6 ) * r_vec
                r = np.sqrt(r2)
                inv_r2 = 1.0 / r2
                s2 = sigma * sigma * inv_r2
                s6 = s2 * s2 * s2
                s12 = s6 * s6
                
                # Скаляр силы (по модулю)
                f_scalar = - 48 * epsilon * inv_r2 * (s12 - 0.5 * s6)
                f_vec = f_scalar * delta  # направление от i к j
                
                # Принцип равенства действия и противодействия
                forces[i] += f_vec
                forces[j] -= f_vec

        
        for (b_i, b_j) in self.bonds:   # используем другие имена, чтобы не путать с i,j из циклов выше
            p_i = self.particles[b_i]
            p_j = self.particles[b_j]
            
            delta = p_j.position - p_i.position
            if self.config.boundary_conditions == "periodic":
                box = np.array(self.config.size)
                delta = delta - box * np.round(delta / box)
            
            r = np.linalg.norm(delta)
            if r < 1e-6:
                continue
            
            f_scalar = k_bond * (r - r0) / r
            f_vec = f_scalar * delta
            
            forces[b_i] += f_vec
            forces[b_j] -= f_vec

        return forces

    def velocity_verlet_step(self, dt: float, gamma: float = 0.0):
        """
        Один шаг интегрирования по алгоритму Верле с опциональным термостатом Ланжевена.
        
        Параметры:
        dt - шаг по времени
        gamma - коэффициент трения (0 = без термостата)
        """
        n = len(self.particles)
        if n == 0:
            return
        
        # Получаем текущие консервативные силы
        forces = self.compute_forces()
        
        # Предварительное обновление позиций и половинное обновление скоростей (Verle)
        for i, p in enumerate(self.particles):
            # Обновляем позиции
            p.position += p.velocity * dt + 0.5 * forces[i] / p.mass * dt**2
            p.position = self.apply_boundary_conditions(p.position)
            
            # Половинное обновление скоростей (без учёта диссипативных/случайных сил)
            p.velocity += 0.5 * forces[i] / p.mass * dt
        
        # Пересчитываем силы для новых позиций
        new_forces = self.compute_forces()
        
        # Если включён термостат Ланжевена, добавляем диссипативные и случайные силы
        if gamma > 0.0:
            kT = self.config.temperature * 1.380649e-23  # постоянная Больцмана в Дж/К (но мы работаем в условных единицах, можно просто kT как параметр)
            # В наших единицах проще использовать kT = temperature (в энергетических единицах), если мы приняли epsilon=1 за энергию.
            # Для простоты будем считать, что температура задана в тех же единицах, что и энергия (т.е. kT = temperature).
            # В потенциале Леннард-Джонса epsilon = 1, значит kT = 1 соответствует температуре, при которой средняя энергия ~1.
            # Так и оставим: kT = self.config.temperature.
            kT = self.config.temperature
            
            # Коэффициент диффузии в скорости: sqrt(2 * gamma * kT / m * dt) для случайной силы
            for i, p in enumerate(self.particles):
                # Сила трения
                friction = -gamma * p.velocity
                
                # Случайная сила (нормальное распределение с нулевым средним и дисперсией 2*gamma*kT*m/dt)
                # Дисперсия каждой компоненты: var = 2 * gamma * kT * m / dt
                std = np.sqrt(4.0 * gamma * kT * p.mass / dt)
                random_force = np.array([random.gauss(0, std) for _ in range(3)])
                
                # Общая сила для финального обновления скорости
                total_force = new_forces[i] + friction + random_force
                
                # Завершаем обновление скорости
                p.velocity += 0.5 * total_force / p.mass * dt
        else:
            # Без термостата — стандартное завершение Verlet
            for i, p in enumerate(self.particles):
                p.velocity += 0.5 * new_forces[i] / p.mass * dt
        
        self.time += dt
        self.step += 1

    def integrate(self, steps: int, dt: float, method='verlet', gamma=0.0):
        """
        Запуск интеграции на несколько шагов.
        gamma - коэффициент трения для термостата Ланжевена (0 = выключен).
        """
        for _ in range(steps):
            if method == 'verlet':
                self.velocity_verlet_step(dt, gamma)
            else:
                raise ValueError(f"Unknown method: {method}")
            self.react(dt)   # добавляем вызов реакций
        
    def add_molecule(self, particles: List[Particle], bonds: List[Tuple[int, int]]):
        """
        Добавляет молекулу, состоящую из нескольких частиц, и регистрирует внутримолекулярные связи.
        particles - список новых частиц (их id будут присвоены автоматически при добавлении).
        bonds - список пар локальных индексов (внутри списка particles), указывающих на связи.
        """
        start_idx = len(self.particles)
        # Добавляем частицы с присвоением глобальных ID
        for i, p in enumerate(particles):
            p.id = start_idx + i
            self.particles.append(p)

        # Регистрируем связи, преобразуя локальные индексы в глобальные
        for i, j in bonds:
            gi = start_idx + i
            gj = start_idx + j
            # Добавляем в список bonds
            self.bonds.append((gi, gj))
            # Добавляем в bond_set с правильным порядком (меньший индекс первым)
            if gi < gj:
                self.bond_set.add((gi, gj))
            else:
                self.bond_set.add((gj, gi))
            # Заполняем поле bonded_to у частиц для быстрого доступа
            self.particles[gi].bonded_to.append(gj)
            self.particles[gj].bonded_to.append(gi)

    def relax(self, steps=50, dt=None, gamma=0.1):
        """
        Выполняет несколько шагов MD для релаксации системы.
        steps - количество шагов
        dt - шаг по времени (если None, используется текущий из основного цикла)
        gamma - коэффициент трения для термостата
        """
        if dt is None:
            dt = 0.001  # можно взять из основного, но лучше задать
        for _ in range(steps):
            self.velocity_verlet_step(dt, gamma=gamma)

    def react(self, dt):
        n_init = try_initiation(self, dt)
        if n_init > 0:
            print(f"Произошло {n_init} актов инициирования")
            self.relax(steps=10000, dt=dt*0.1, gamma=0.5)  # усиленная релаксация
        n_prop = try_propagation(self, dt)
        if n_prop > 0:
            print(f"Произошло {n_prop} реакций роста")
            self.relax(steps=10000, dt=dt*0.1, gamma=0.5)

    def plot(self, step=None, show_connections=True, save_path=None):
        """
        3D-визуализация текущего состояния реактора.
        
        Параметры:
            step: номер шага (для заголовка)
            show_connections: рисовать ли связи
            save_path: если указан, сохранить изображение в файл
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Определяем цвета и размеры для разных типов частиц
        style_map = {
            ParticleType.INITIATOR:      {'color': 'red',    'size': 100, 'marker': 'o'},
            ParticleType.SIM_BACKBONE:    {'color': 'gray',   'size': 50,  'marker': 's'},
            ParticleType.SIM_VINYL:       {'color': 'green',  'size': 60,  'marker': '^'},
            ParticleType.SIM_RADICAL:     {'color': 'blue',   'size': 80,  'marker': 'o'},
            ParticleType.SIM_RADICAL_SLOW:{'color': 'cyan',   'size': 80,  'marker': 'o'},
            ParticleType.SIM_INERT:       {'color': 'black',  'size': 40,  'marker': 'x'},
            # Если появятся другие типы, можно добавить
        }
        
        # Группируем частицы по типу для легенды
        type_to_particles = defaultdict(list)
        for p in self.particles:
            type_to_particles[p.ptype].append(p.position)
        
        # Рисуем частицы каждого типа
        for ptype, positions in type_to_particles.items():
            if not positions:
                continue
            positions = np.array(positions)
            style = style_map.get(ptype, {'color': 'purple', 'size': 50, 'marker': 'o'})
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    c=style['color'], s=style['size'], marker=style['marker'],
                    label=ptype.value, alpha=0.8)
        
        # Рисуем связи (опционально)
        if show_connections and self.bonds:
            for i, j in self.bonds:
                if i >= len(self.particles) or j >= len(self.particles):
                    continue  # защита от старых индексов
                pos_i = self.particles[i].position
                pos_j = self.particles[j].position
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]],
                        'k-', linewidth=0.5, alpha=0.3)
        
        # Настройка осей
        ax.set_xlim(0, self.config.size[0])
        ax.set_ylim(0, self.config.size[1])
        ax.set_zlim(0, self.config.size[2])
        ax.set_xlabel('X (нм)')
        ax.set_ylabel('Y (нм)')
        ax.set_zlabel('Z (нм)')
        
        if step is not None:
            ax.set_title(f'Состояние реактора, шаг {step}')
        else:
            ax.set_title('Состояние реактора')
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Изображение сохранено: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
        return fig

    def summary(self) -> str:
        """Краткая информация о реакторе."""
        return (f"Reactor '{self.config.name}': {len(self.particles)} particles, "
                f"size={self.config.size}, BC={self.config.boundary_conditions}")