"""Описание столов из параметров ROS 2 (см. config/sorting.yaml)."""
import numpy as np

from sortbot_task import Table


def read_tables(node):
    """Собрать столы из параметров вида tables.<имя>.<поле>."""
    params = node.get_parameters_by_prefix("tables")
    names = sorted({key.split(".")[0] for key in params})
    tables = []
    for name in names:
        get = lambda field: params[f"{name}.{field}"].value  # noqa: E731
        tables.append(Table(
            name=name,
            center=np.array(get("center"), dtype=float),
            half_size=np.array(get("half_size"), dtype=float),
            surface_z=float(get("surface_z")),
            heading=float(get("heading")),
            color=get("color") or None,  # пустая строка: стол, с которого забираем
        ))
    return tables
