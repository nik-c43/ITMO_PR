"""То, что в симуляции известно точно: столы сцены и истинные положения кубиков.

Столы на настоящем роботе придут из карты или параметров. Истинных положений кубиков
там нет вовсе: в симуляции они нужны только для проверки результата."""
import numpy as np

from sortbot_task.world import Table

TABLE_GEOMS = {
    "table_source": ("table_source_top", None),  # стол, с которого забираем
    "table_red": ("table_red_top", "red"),
    "table_green": ("table_green_top", "green"),
    "table_blue": ("table_blue_top", "blue"),
}


def tables_from_sim(sim):
    """Список столов сцены. Робот подъезжает к каждому со стороны начала координат."""
    tables = []
    for name, (geom_name, color) in TABLE_GEOMS.items():
        geom = sim.model.geom(geom_name)
        center = sim.data.geom(geom.id).xpos
        tables.append(Table(
            name=name,
            center=center[:2].copy(),
            half_size=geom.size[:2].copy(),
            surface_z=float(center[2] + geom.size[2]),
            heading=float(np.arctan2(center[1], center[0])),
            color=color,
        ))
    return tables


def cube_positions(sim):
    """Истинные положения кубиков: {имя: (x, y, z)}. Робот их не знает, это только для проверок."""
    return {name: sim.body_pos(name) for name in sim.cube_names}


def placement_report(tables, positions):
    """Сверить истинные положения кубиков со столами.

    Возвращает (строки отчёта, сколько кубиков стоит на столах своего цвета, сколько всего).
    """
    lines, correct = [], 0
    for name, position in sorted(positions.items()):
        position = np.asarray(position)
        color = name.split("_")[1]  # имена кубиков в сцене: cube_<цвет>_<номер>
        table = next((t for t in tables if t.holds_cube(position)), None)
        right = table is not None and table.color == color
        correct += right
        where = f"({position[0]:6.3f}, {position[1]:6.3f}, {position[2]:5.3f})"
        lines.append(f"  {name:13s} {where} {'на ' + table.name if table else 'не на столе':16s} "
                     f"{'верно' if right else 'НЕВЕРНО'}")
    return lines, correct, len(positions)
