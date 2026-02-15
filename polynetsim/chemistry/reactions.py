"""
Модуль с функциями для обработки химических реакций.
Каждая функция принимает reactor и dt, и возвращает количество выполненных реакций.
"""

import numpy as np
import random
from typing import List, Tuple
from polynetsim.core.particle import Particle, ParticleType

def try_propagation(reactor, dt: float):
    """
    Обрабатывает реакции роста для трёхчастичного мономера:
    - радикал (обычный или медленный) атакует винильную группу (SIM_VINYL)
    - атакованная винильная группа становится обычным радикалом (SIM_RADICAL)
    - атакующий радикал становится инертным (SIM_INERT)
    - создаётся связь между ними
    Для медленных радикалов (SIM_RADICAL_SLOW) константа скорости уменьшена.
    """
    params = reactor.config.params.reaction
    kp = params.k_propagation
    r_cut = params.reaction_radius
    particles = reactor.particles
    reactions = 0

    # Индексы радикалов обоих типов
    radicals = [i for i, p in enumerate(particles) if p.ptype in (ParticleType.SIM_RADICAL, ParticleType.SIM_RADICAL_SLOW)]
    # Индексы винильных групп, которые ещё не прореагировали (имеют только внутримолекулярные связи)
    # В трёхчастичном мономере у винила одна связь (с остовом). Если связей больше, значит, он уже участвовал в реакции.
    vinyls = [i for i, p in enumerate(particles) if p.ptype == ParticleType.SIM_VINYL and len(p.bonded_to) <= 1]

    for i_rad in radicals:
        # Определяем тип радикала и эффективную константу
        is_slow = (particles[i_rad].ptype == ParticleType.SIM_RADICAL_SLOW)
        k_eff = kp * (0.1 if is_slow else 1.0)

        for i_vin in vinyls:
            if i_rad == i_vin:
                continue
            # Расстояние с учётом PBC
            delta = particles[i_vin].position - particles[i_rad].position
            if reactor.config.boundary_conditions == "periodic":
                box = np.array(reactor.config.size)
                delta = delta - box * np.round(delta / box)
            r = np.linalg.norm(delta)
            if r > r_cut:
                continue

            prob = k_eff * dt
            if prob > 1.0:
                prob = 1.0
            if random.random() < prob:
                # Выполняем реакцию
                # 1. Создаём связь
                if i_rad < i_vin:
                    reactor.bond_set.add((i_rad, i_vin))
                else:
                    reactor.bond_set.add((i_vin, i_rad))
                reactor.bonds.append((i_rad, i_vin))
                # 2. Обновляем bonded_to
                particles[i_rad].bonded_to.append(i_vin)
                particles[i_vin].bonded_to.append(i_rad)

                # 3. Меняем типы:
                # Атакованная винильная группа становится обычным радикалом
                particles[i_vin].ptype = ParticleType.SIM_RADICAL
                # Атакующий радикал (любой) становится инертным
                particles[i_rad].ptype = ParticleType.SIM_INERT

                reactions += 1
                break  # один радикал – одна реакция за шаг
    return reactions

def try_initiation(reactor, dt: float) -> int:
    """
    Обрабатывает распад инициатора: частица INITIATOR → два обычных радикала (SIM_RADICAL).
    Возвращает число распавшихся молекул инициатора.
    """
    params = reactor.config.params.reaction
    k_i = params.k_initiation
    if k_i <= 0:
        return 0

    to_remove = []      # индексы инициаторов, которые распадутся
    to_add = []         # новые радикалы для добавления

    for i, p in enumerate(reactor.particles):
        if p.ptype != ParticleType.INITIATOR:
            continue
        prob = k_i * dt
        if prob > 1.0:
            prob = 1.0
        if random.random() < prob:
            to_remove.append(i)
            pos = p.position.copy()
            for _ in range(2):
                offset = np.random.normal(0, 0.2, 3)
                new_pos = pos + offset
                new_pos = reactor.apply_boundary_conditions(new_pos)
                new_rad = Particle(
                    id=-1,
                    ptype=ParticleType.SIM_RADICAL,  # обычный радикал
                    position=new_pos,
                    velocity=np.zeros(3),
                    mass=1.0,
                    radius=0.3
                )
                to_add.append(new_rad)

    for idx in sorted(to_remove, reverse=True):
        del reactor.particles[idx]

    for rad in to_add:
        reactor.add_particle(rad)

    return len(to_remove)