"""Описание столов сцены для логики сортировки. На роботе эти данные придут из карты или параметров."""
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
