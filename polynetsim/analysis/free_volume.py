"""
Модуль для анализа свободного объёма в полимерных системах.
Реализует методы:
- геометрический свободный объём (метод пробной сферы)
- (позже) флуктуационный свободный объём
"""

import numpy as np
from typing import Optional, List
from scipy.spatial import KDTree

class FreeVolumeAnalyzer:
    """
    Анализатор свободного объёма.
    Использует метод пробной сферы для оценки геометрического свободного объёма.
    """
    
    def __init__(self, probe_radius: float = 0.2):
        """
        probe_radius: радиус пробной сферы (в тех же единицах, что и радиусы частиц)
        """
        self.probe_radius = probe_radius
    
    def geometric_free_volume(self, reactor, n_samples: int = 10000, use_cache: bool = False) -> float:
        """
        Оценивает долю геометрического свободного объёма в реакторе.
        
        Параметры:
        reactor - объект Reactor с частицами (каждая частица имеет атрибут .position и .radius)
        n_samples - количество случайных пробных точек
        use_cache - если True, сохраняет KD-дерево для ускорения (полезно при многократных вызовах)
        
        Возвращает:
        доля точек, не перекрывающихся ни с одной частицей (0..1)
        """
        particles = reactor.particles
        if not particles:
            return 1.0  # пустой реактор
        
        # Извлекаем позиции и радиусы
        positions = np.array([p.position for p in particles])
        radii = np.array([p.radius for p in particles])
        box_size = np.array(reactor.config.size)
        
        # Генерируем случайные точки в объёме реактора
        test_points = np.random.rand(n_samples, 3) * box_size
        
        # Строим KD-дерево для быстрого поиска ближайших соседей
        tree = KDTree(positions)
        
        # Для каждой точки находим расстояние до ближайшей частицы
        distances, indices = tree.query(test_points, k=1)
        
        # Точка считается доступной, если расстояние до ближайшей частицы превышает сумму радиусов + probe_radius
        accessible = distances > (radii[indices] + self.probe_radius)
        
        free_fraction = np.mean(accessible)
        return free_fraction
    
    def free_volume_map(self, reactor, grid: tuple = (20, 20, 20)):
        """
        Строит 3D-карту свободного/занятого пространства на регулярной сетке.
        Возвращает трёхмерный булев массив: True = свободно, False = занято.
        """
        particles = reactor.particles
        if not particles:
            return np.ones(grid, dtype=bool)
        
        positions = np.array([p.position for p in particles])
        radii = np.array([p.radius for p in particles])
        box_size = np.array(reactor.config.size)
        
        # Создаём координатную сетку
        x = np.linspace(0, box_size[0], grid[0])
        y = np.linspace(0, box_size[1], grid[1])
        z = np.linspace(0, box_size[2], grid[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        # Для каждой точки сетки проверяем, перекрывается ли она с какой-либо частицей
        tree = KDTree(positions)
        distances, indices = tree.query(points, k=1)
        occupied = distances <= (radii[indices] + self.probe_radius)
        
        free_map = ~occupied.reshape(grid)
        return free_map
    
    def fluctuational_free_volume(self, positions_history: List[np.ndarray], 
                                radii: Optional[np.ndarray] = None) -> dict:
        """
        Оценивает флуктуационный свободный объём на основе траектории частиц.
        
        Параметры:
        positions_history - список массивов позиций для разных моментов времени.
                            Каждый элемент должен быть массивом формы (N, 3), где N - число частиц.
        radii - массив радиусов частиц (если None, берутся из reactor, но здесь reactor не передаётся)
        
        Возвращает словарь:
            'mean_squared_displacement' - среднеквадратичное смещение от среднего положения
            'fluctuation_volume_per_particle' - средний объём флуктуаций (как сфера с радиусом = sqrt(MSD))
            'relative_fluctuation' - отношение fluctuation_volume_per_particle к среднему объёму частицы
            'msd_per_component' - массив среднеквадратичных отклонений по координатам (усреднённый)
        """
        if len(positions_history) < 2:
            raise ValueError("Need at least two frames to compute fluctuations.")
        
        n_particles = positions_history[0].shape[0]
        # Проверяем, что все кадры имеют одинаковое количество частиц
        for pos in positions_history:
            assert pos.shape[0] == n_particles, "Inconsistent number of particles across frames."
        
        # Стекируем все кадры в массив (n_frames, n_particles, 3)
        all_positions = np.stack(positions_history, axis=0)  # (T, N, 3)
        
        # Вычисляем среднее положение каждой частицы по времени
        mean_positions = np.mean(all_positions, axis=0)  # (N, 3)
        
        # Вычисляем среднеквадратичное отклонение от среднего (по всем измерениям)
        # Для каждой частицы и каждого кадра считаем квадрат отклонения, затем усредняем по времени
        squared_displacements = np.mean(np.sum((all_positions - mean_positions[None, :, :])**2, axis=2), axis=0)  # (N,)
        msd_per_particle = squared_displacements  # среднеквадратичное смещение для каждой частицы
        
        # Среднее MSD по всем частицам
        mean_msd = np.mean(msd_per_particle)
        
        # Оценка радиуса флуктуаций (среднеквадратичное смещение)
        rms = np.sqrt(mean_msd)
        
        # Объём сферы радиуса rms
        fluctuation_volume = (4.0/3.0) * np.pi * rms**3
        
        # Если переданы радиусы частиц, можно вычислить относительный объём
        if radii is not None:
            mean_radius = np.mean(radii)
            particle_volume = (4.0/3.0) * np.pi * mean_radius**3
            relative_fluctuation = fluctuation_volume / particle_volume
        else:
            relative_fluctuation = None
        
        # Дополнительно: среднеквадратичные отклонения по каждой координате (усреднённые по частицам)
        msd_components = np.mean(np.mean((all_positions - mean_positions[None, :, :])**2, axis=0), axis=0)  # (3,)
        
        return {
            'mean_squared_displacement': mean_msd,
            'fluctuation_volume': fluctuation_volume,
            'relative_fluctuation': relative_fluctuation,
            'msd_components': msd_components,
            'rms': rms,
            'msd_per_particle': msd_per_particle  # можно использовать для анализа неоднородности
        }    