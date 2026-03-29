    def estimate_available_particles(self, rad_idx, vinyl_fraction=0.6667, return_positions=False):
        """
        Оценивает для радикала с индексом rad_idx:
        - N_i: максимальное число блуждающих частиц (после геометрической упаковки)
        - w_i: ожидаемое число винильных групп
        - (опционально) positions: координаты размещённых частиц
        """
        rad = self.particles[rad_idx]
        R_s = rad.radius + self.config.params.reaction.avg_monomer_radius
        part_radius = self.config.params.reaction.avg_monomer_radius
        
        print(f"DEBUG: R_s = {R_s:.3f}, part_radius = {part_radius:.3f}")
        
        theta_green = np.arcsin(part_radius / R_s)
        overlap_tolerance = getattr(self.config.params.reaction, 'overlap_tolerance', 0.05)
        theta_green_eff = theta_green * (1 - overlap_tolerance)
        
        print(f"DEBUG: theta_green = {theta_green:.3f} rad, theta_green_eff = {theta_green_eff:.3f} rad")
        
        # --- Шаг 1: Серые круги (запрещённые зоны) ---
        gray_circles = []
        for p in self.particles:
            if p.id == rad_idx:
                continue
            vec = p.position - rad.position
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                continue
            dir_gray = vec / dist
            # Точная формула для углового радиуса с учётом размера зелёной частицы
            Rg = p.radius
            Rb = part_radius
            cos_theta = (R_s**2 + dist**2 - (Rg + Rb)**2) / (2 * R_s * dist)
            cos_theta = np.clip(cos_theta, -1, 1)
            theta_gray = np.arccos(cos_theta)
            gray_circles.append((dir_gray, theta_gray))
            print(f"DEBUG: Gray circle at dir {dir_gray}, theta = {theta_gray:.3f} rad, dist = {dist:.3f}")
        
        print(f"DEBUG: Всего серых кругов: {len(gray_circles)}")
        
        # --- Вспомогательные функции ---
        def angle_between(v1, v2):
            cos = np.clip(np.dot(v1, v2), -1, 1)
            return np.arccos(cos)
        
        def intersection_points(dir1, theta1, dir2, theta2):
            """Возвращает точки пересечения двух кругов на сфере."""
            delta = angle_between(dir1, dir2)
            
            if delta > theta1 + theta2 + 1e-8:
                return []
            if delta < abs(theta1 - theta2) - 1e-8:
                return []
            
            cos_alpha = (np.cos(theta2) - np.cos(theta1) * np.cos(delta)) / (np.sin(theta1) * np.sin(delta))
            cos_alpha = np.clip(cos_alpha, -1, 1)
            alpha = np.arccos(cos_alpha)
            
            e1 = dir1
            e2 = dir2 - np.dot(dir2, e1) * e1
            e2_norm = np.linalg.norm(e2)
            if e2_norm < 1e-8:
                return [e1]
            e2 = e2 / e2_norm
            e3 = np.cross(e1, e2)
            
            cand1 = np.cos(theta1) * e1 + np.sin(theta1) * (np.cos(alpha) * e2 + np.sin(alpha) * e3)
            cand2 = np.cos(theta1) * e1 + np.sin(theta1) * (np.cos(alpha) * e2 - np.sin(alpha) * e3)
            
            cand1 = cand1 / np.linalg.norm(cand1)
            cand2 = cand2 / np.linalg.norm(cand2)
            
            if np.allclose(cand1, cand2):
                return [cand1]
            return [cand1, cand2]
        
        def is_valid(candidate_dir, placed_green):
            """Проверяет, не перекрывается ли кандидат с серыми и уже размещёнными зелёными кругами."""
            # Проверка с серыми
            for g_dir, g_theta in gray_circles:
                angle = angle_between(candidate_dir, g_dir)
                if angle < g_theta - 1e-4:
                    return False
            # Проверка с зелёными
            for p_dir in placed_green:
                angle = angle_between(candidate_dir, p_dir)
                if angle < 2 * theta_green_eff - 1e-4:
                    return False
            return True
        
        # --- Шаг 2: Основной цикл упаковки ---
        placed_dirs = []  # уже размещённые зелёные круги
        all_forbidden = gray_circles.copy()  # все запрещённые зоны (серые + зелёные)
        
        iteration = 0
        while True:
            iteration += 1
            print(f"\n=== ИТЕРАЦИЯ {iteration} ===")
            
            # Поиск кандидатов
            candidates = []
            
            # 2.1 Пересечения всех запрещённых кругов между собой
            print("Поиск пересечений между запрещёнными кругами...")
            for i, (f1, t1) in enumerate(all_forbidden):
                for j, (f2, t2) in enumerate(all_forbidden[i+1:], i+1):
                    pts = intersection_points(f1, t1, f2, t2)
                    for pt in pts:
                        if is_valid(pt, placed_dirs):
                            candidates.append(pt)
            
            # 2.2 Если пересечений нет, ищем точки на границах
            if len(candidates) == 0:
                print("Пересечений нет, ищем точки на границах...")
                
                # Сначала пробуем ближайшую пару
                if len(all_forbidden) >= 2:
                    # Находим ближайшую пару по угловому расстоянию
                    min_angle = float('inf')
                    closest_pair = None
                    for i, (f1, t1) in enumerate(all_forbidden):
                        for j, (f2, t2) in enumerate(all_forbidden[i+1:], i+1):
                            angle = angle_between(f1, f2)
                            if angle < min_angle:
                                min_angle = angle
                                closest_pair = ((f1, t1), (f2, t2))
                    
                    if closest_pair is not None:
                        (f1, t1), (f2, t2) = closest_pair
                        # Точки на границах вдоль дуги
                        perp = f2 - np.dot(f2, f1) * f1
                        perp_norm = np.linalg.norm(perp)
                        if perp_norm > 1e-8:
                            perp = perp / perp_norm
                            cand1 = np.cos(t1) * f1 + np.sin(t1) * perp
                            cand1 = cand1 / np.linalg.norm(cand1)
                            cand2 = np.cos(t2) * f2 - np.sin(t2) * perp
                            cand2 = cand2 / np.linalg.norm(cand2)
                            
                            for cand in [cand1, cand2]:
                                if is_valid(cand, placed_dirs):
                                    candidates.append(cand)
                
                # Если всё ещё нет кандидатов, добавляем случайное направление
                if len(candidates) == 0:
                    print("Добавляем случайное направление")
                    random_dir = np.random.randn(3)
                    random_dir = random_dir / np.linalg.norm(random_dir)
                    if is_valid(random_dir, placed_dirs):
                        candidates.append(random_dir)
            
            # 2.3 Убираем дубликаты кандидатов
            unique_candidates = []
            for cand in candidates:
                duplicate = False
                for u in unique_candidates:
                    if angle_between(cand, u) < 1e-4:
                        duplicate = True
                        break
                if not duplicate:
                    unique_candidates.append(cand)
            candidates = unique_candidates
            
            print(f"Найдено кандидатов: {len(candidates)}")
            
            if len(candidates) == 0:
                print("Новых кандидатов нет, завершаем упаковку.")
                break
            
            # 2.4 Выбираем первого кандидата (можно улучшить выбор)
            best = candidates[0]
            placed_dirs.append(best)
            all_forbidden.append((best, theta_green_eff))
            print(f"Размещена частица {len(placed_dirs)} в направлении {best}")
            
            # Генерируем новых кандидатов от пересечений с новой частицей
            # (они будут найдены на следующей итерации)
        
        # --- Шаг 3: Преобразование в координаты ---
        placed_points = [rad.position + R_s * d for d in placed_dirs]
        N_i = len(placed_points)
        w_i = N_i * vinyl_fraction
        
        print(f"\nИТОГО: Размещено частиц: {N_i}")
        
        if return_positions:
            return N_i, w_i, np.array(placed_points)
        else:
            return N_i, w_i