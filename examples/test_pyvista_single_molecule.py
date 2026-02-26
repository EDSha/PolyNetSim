import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

# Попытка импорта PyVista с установкой при отсутствии
try:
    import pyvista as pv
except ImportError:
    print("PyVista не установлен. Установите его командой: pip install pyvista")
    sys.exit(1)

def create_triple_monomer(center, chain_id, orientation='x'):
    """Создаёт мономер из трёх частиц: остов и две винильные группы."""
    d = 0.5
    if orientation == 'x':
        pos0 = center
        pos1 = center + np.array([ d, 0, 0])
        pos2 = center + np.array([-d, 0, 0])
    elif orientation == 'y':
        pos0 = center
        pos1 = center + np.array([0,  d, 0])
        pos2 = center + np.array([0, -d, 0])
    else:
        pos0 = center
        pos1 = center + np.array([0, 0,  d])
        pos2 = center + np.array([0, 0, -d])
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
    # Конфигурация реактора: небольшой куб
    box_size = 5.0
    config = ReactorConfig(size=(box_size, box_size, box_size), temperature=1.0)
    config.params.lj.epsilon = 0.0  # отключаем Леннард-Джонс для чистоты
    config.params.bond.stiffness = 1000.0
    config.params.bond.length = 0.5

    reactor = Reactor(config)

    # Создаём одну молекулу в центре
    center = np.array([box_size/2, box_size/2, box_size/2])
    particles, bonds = create_triple_monomer(center, chain_id=1, orientation='x')
    reactor.add_molecule(particles, bonds)

    # Обновляем сетку (занятые ячейки)
    reactor.update_grid()

    print(f"Частиц: {len(reactor.particles)}")
    print(f"Размер сетки: {reactor.grid_shape}")
    print(f"Всего ячеек: {np.prod(reactor.grid_shape)}")

    # --- Визуализация с PyVista ---
    plotter = pv.Plotter()

    # 1. Рисуем контур реактора (прозрачный куб)
    bounds = [0, box_size, 0, box_size, 0, box_size]
    outline = pv.Box(bounds=bounds)
    plotter.add_mesh(outline, color='black', style='wireframe', line_width=2)

    # 2. Рисуем частицы как сферы с реальными радиусами
    for p in reactor.particles:
        sphere = pv.Sphere(radius=p.radius, center=p.position)
        # Выбираем цвет по типу (упрощённо)
        if p.ptype == ParticleType.SIM_BACKBONE:
            color = 'gray'
        elif p.ptype == ParticleType.SIM_VINYL:
            color = 'green'
        else:
            color = 'blue'
        plotter.add_mesh(sphere, color=color, opacity=0.8, smooth_shading=True)

    # 3. Рисуем связи (цилиндры)
    for i, j in reactor.bonds:
        p1 = reactor.particles[i].position
        p2 = reactor.particles[j].position
        # Создаём цилиндр между точками
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length > 0:
            cylinder = pv.Cylinder(center=(p1 + p2)/2, direction=direction, radius=0.05, height=length)
            plotter.add_mesh(cylinder, color='black')

    # 4. (Опционально) Показать занятые ячейки сетки как маленькие кубики
    #    Чтобы не перегружать сцену, покажем только для одного слоя (z=середина)
    #    или можно добавить полупрозрачные воксели, но для мелкой сетки это может быть медленно.
    #    Для демонстрации выведем информацию о занятости в консоль.
    occupied_cells = np.sum(reactor.grid)
    print(f"Занятых ячеек сетки: {occupied_cells}")

    # Запускаем интерактивное окно
    plotter.show()

if __name__ == "__main__":
    main()