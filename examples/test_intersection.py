import numpy as np

def intersection_points(dir1, theta1, dir2, theta2):
    """
    Возвращает точки пересечения двух окружностей на сфере.
    
    Параметры:
        dir1, dir2: единичные векторы центров кругов
        theta1, theta2: угловые радиусы кругов (в радианах)
    
    Возвращает:
        Список единичных векторов (0, 1 или 2 точки)
    """
    # Угол между центрами
    cos_delta = np.clip(np.dot(dir1, dir2), -1, 1)
    delta = np.arccos(cos_delta)
    
    # Проверка существования решения
    if delta > theta1 + theta2 + 1e-8:
        print(f"Круги слишком далеко: delta={delta:.3f} > {theta1:.3f}+{theta2:.3f}")
        return []
    if delta < abs(theta1 - theta2) - 1e-8:
        print(f"Один круг внутри другого: delta={delta:.3f} < |{theta1:.3f}-{theta2:.3f}|")
        return []
    
    # Вычисляем косинус угла phi
    cos_phi = (np.cos(theta2) - np.cos(theta1) * cos_delta) / (np.sin(theta1) * np.sin(delta))
    cos_phi = np.clip(cos_phi, -1, 1)
    print(f"cos_phi = {cos_phi:.3f}")
    
    # Строим базис
    e1 = dir1
    # e2 - единичный вектор, перпендикулярный e1 в плоскости dir1-dir2
    e2 = dir2 - cos_delta * e1
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-8:
        # Круги концентрические - бесконечно много решений, возвращаем одно
        print("Концентрические круги")
        return [e1]
    e2 = e2 / e2_norm
    # e3 - перпендикуляр к плоскости
    e3 = np.cross(e1, e2)
    
    # Два возможных угла
    phi1 = np.arccos(cos_phi)
    phi2 = -phi1
    
    points = []
    for phi in [phi1, phi2]:
        x = np.cos(theta1) * e1 + np.sin(theta1) * (np.cos(phi) * e2 + np.sin(phi) * e3)
        x = x / np.linalg.norm(x)  # нормализация для численной стабильности
        points.append(x)
    
    # Убираем дубликаты (если phi1 и phi2 совпадают)
    if len(points) == 2 and np.allclose(points[0], points[1]):
        return [points[0]]
    
    return points

def test_intersection():
    print("=== Тест 1: два круга радиусом 60° на осях X и Y ===")
    dir1 = np.array([1, 0, 0])
    dir2 = np.array([0, 1, 0])
    theta = 1.047  # 60° в радианах
    
    points = intersection_points(dir1, theta, dir2, theta)
    print(f"Найдено точек: {len(points)}")
    for i, p in enumerate(points):
        print(f"Точка {i}: {p}")
        # Проверяем расстояния
        dist1 = np.arccos(np.clip(np.dot(p, dir1), -1, 1))
        dist2 = np.arccos(np.clip(np.dot(p, dir2), -1, 1))
        print(f"  до dir1: {dist1:.3f} рад (должно быть {theta:.3f})")
        print(f"  до dir2: {dist2:.3f} рад (должно быть {theta:.3f})")
    
    print("\n=== Тест 2: два круга радиусом 30° на осях X и Y ===")
    theta_small = 0.524  # 30°
    points = intersection_points(dir1, theta_small, dir2, theta_small)
    print(f"Найдено точек: {len(points)}")
    for i, p in enumerate(points):
        print(f"Точка {i}: {p}")
        dist1 = np.arccos(np.clip(np.dot(p, dir1), -1, 1))
        dist2 = np.arccos(np.clip(np.dot(p, dir2), -1, 1))
        print(f"  до dir1: {dist1:.3f} рад (должно быть {theta_small:.3f})")
        print(f"  до dir2: {dist2:.3f} рад (должно быть {theta_small:.3f})")
    
    print("\n=== Тест 3: касающиеся круги (60° и 30° на расстоянии 90°) ===")
    # Круги должны касаться, если 60°+30° = 90°
    points = intersection_points(dir1, theta, dir2, theta_small)
    print(f"Найдено точек: {len(points)}")
    for i, p in enumerate(points):
        print(f"Точка {i}: {p}")
        dist1 = np.arccos(np.clip(np.dot(p, dir1), -1, 1))
        dist2 = np.arccos(np.clip(np.dot(p, dir2), -1, 1))
        print(f"  до dir1: {dist1:.3f} рад (должно быть {theta:.3f})")
        print(f"  до dir2: {dist2:.3f} рад (должно быть {theta_small:.3f})")

if __name__ == "__main__":
    test_intersection()