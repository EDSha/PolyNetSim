import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import numpy as np
import pyvista as pv
from polynetsim.core.particle import Particle, ParticleType
from polynetsim.core.reactor import Reactor, ReactorConfig

def main():
    # Конфигурация реактора
    box_size = 5.0
    config = ReactorConfig(size=(box_size, box_size, box_size), temperature=1.0)
    config.params.reaction.avg_monomer_radius = 0.3
    config.params.lj.epsilon = 0.0
    config.params.bond.stiffness = 1000.0
    config.params.bond.length = 0.5
    config.params.reaction.overlap_tolerance = 0.05  # допуск на перекрытие 5%
    
    # Расчёт микроскопических констант
    V_reactor_l = box_size**3 * 1e-24
    NA = 6.022e23
    kp_macro = 6.4e3
    kp_micro = kp_macro / (NA * V_reactor_l)
    print(f"kp_micro = {kp_micro:.2e} 1/s")
    config.params.reaction.kp_micro = kp_micro
    config.params.reaction.kd_micro = 7.1e-5

    reactor = Reactor(config)

    # Создаём один радикал в центре
    rad = Particle(id=0, ptype=ParticleType.SIM_RADICAL,
                   position=np.array([box_size/2, box_size/2, box_size/2]),
                   velocity=np.zeros(3), mass=1.0, radius=0.3,
                   chain_id=1, is_free=False)
    reactor.add_particle(rad)
    
    # Обновляем сетку после добавления радикала
    reactor.update_grid()
    
    # Вычисляем N_max_global для идеального радикала (без экранирующих частиц)
    N_max_global, _ = reactor.estimate_available_particles(0, return_positions=False)
    print(f"N_max_global (идеальный) = {N_max_global}")
    config.params.reaction.N_max_global = N_max_global

    # Добавляем экранирующие частицы (только один раз!)
    offsets = [
        [0.6, 0.0, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.0, 0.6],
        [-0.6, 0.0, 0.0],
    ]
    for i, off in enumerate(offsets, start=1):
        p = Particle(id=i, ptype=ParticleType.SIM_INERT,
                     position=rad.position + np.array(off),
                     velocity=np.zeros(3), mass=1.0, radius=0.3,
                     chain_id=1, is_free=False)
        reactor.add_particle(p)

    # Обновляем сетку после добавления экранирующих частиц
    reactor.update_grid()

    # Получаем доступные позиции для текущего радикала (с учётом экранирования)
    N_i, w_i, avail_pos = reactor.estimate_available_particles(0, return_positions=True)
    print(f"Доступных направлений: {len(avail_pos)}")
    print(f"N_i = {N_i}, w_i = {w_i:.2f}")

    # Визуализация с PyVista
    plotter = pv.Plotter()

    # Контур реактора
    bounds = [0, box_size, 0, box_size, 0, box_size]
    outline = pv.Box(bounds=bounds)
    plotter.add_mesh(outline, color='black', style='wireframe', line_width=2)

    # Рисуем радикал (красный)
    sphere_rad = pv.Sphere(radius=rad.radius, center=rad.position)
    plotter.add_mesh(sphere_rad, color='red', opacity=0.8, smooth_shading=True)

    # Рисуем экранирующие частицы (серые)
    for p in reactor.particles[1:]:
        sphere = pv.Sphere(radius=p.radius, center=p.position)
        plotter.add_mesh(sphere, color='gray', opacity=0.5, smooth_shading=True)

    # Рисуем сферу зондирования (полупрозрачная сетка)
    probe_sphere = pv.Sphere(radius=rad.radius + config.params.reaction.avg_monomer_radius,
                             center=rad.position)
    plotter.add_mesh(probe_sphere, color='blue', opacity=0.1, style='wireframe')

    # Отображаем доступные позиции как маленькие зелёные сферы
    # Отображаем размещённые блуждающие частицы (зелёные)
    if len(avail_pos) > 0:
        for pos in avail_pos:
            sphere = pv.Sphere(radius=config.params.reaction.avg_monomer_radius, center=pos)
            plotter.add_mesh(sphere, color='green', opacity=0.5, smooth_shading=True)

    # Генерируем случайные направления и показываем доступные/заблокированные точки (для контраста)
    n_test = 500
    directions = np.random.randn(n_test, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    R_s = rad.radius + config.params.reaction.avg_monomer_radius
    available_points = []
    blocked_points = []
    for d in directions:
        point = rad.position + R_s * d
        if not reactor.is_cell_occupied(point):
            available_points.append(point)
        else:
            blocked_points.append(point)

    if available_points:
        avail = np.array(available_points)
        # Добавляем полупрозрачные точки для контекста
        plotter.add_points(avail, color='green', point_size=3, opacity=0.3, label='Доступно (тест)')
    if blocked_points:
        blocked = np.array(blocked_points)
        plotter.add_points(blocked, color='red', point_size=3, opacity=0.3, label='Занято (тест)')

    plotter.add_legend()
    plotter.show()

    # Расчёт скоростей
    total_rate, rates, rad_indices = reactor.compute_rates()
    print(f"Total rate: {total_rate:.2e} 1/s")
    for i, rate in zip(rad_indices, rates):
        print(f"Radical {i}: rate = {rate:.2e} 1/s")

if __name__ == "__main__":
    main()