"""Логика сортировки: развезти кубики со стола-источника по столам их цвета.

Ход работы:

    осмотреть стол-источник  ->  кубиков нет: конец
            ^                        |
            |                        v
            +---- кубики кончились: выбрать ближайший кубик
                                     |
              подъехать -> посмотреть вблизи -> взять -> отвезти -> поставить

Перед концом робот осматривает стол ещё раз: если кубик уронили или не заметили
с первого раза, он найдётся на новом кадре.

Здесь нет ни симулятора, ни ROS. Нужны:
    skills    движения (sortbot_motion.Skills)
    camera    объект с методом capture(), возвращающим RgbdFrame
    detector  объект с методом detect(frame, surface_z), возвращающим находки
    tables    список столов (sortbot_task.Table)
"""
from dataclasses import dataclass, field

import numpy as np

from sortbot_task.memory import CubeMemory
from sortbot_vision.geometry import in_view


@dataclass
class SorterConfig:
    cube_size: float = 0.04  # ребро кубика, м
    release_height: float = 0.005  # с какой высоты над столом кубик отпускается, м
    slot_radius: float = 0.08  # место занято, если ближе этого уже стоит отвезённый кубик, м
    view_margin: int = 40  # кубик ближе этого к краю кадра может быть обрезан, пикс
    table_margin: float = 0.01  # находки ближе этого к краю стола не считаются лежащими на нём, м
    max_rounds: int = 4  # сколько раз самое большее осматривать стол-источник


@dataclass
class SortResult:
    delivered: list = field(default_factory=list)  # отвезённые кубики (TrackedCube)
    left: list = field(default_factory=list)  # кубики, которые остались на столе-источнике
    rounds: int = 0  # сколько раз осматривали стол

    @property
    def ok(self):
        return not self.left


class Sorter:
    def __init__(self, skills, camera, detector, tables, memory=None, config=None, log=print):
        self.skills = skills
        self.camera = camera
        self.detector = detector
        self.memory = memory or CubeMemory()
        self.config = config or SorterConfig()
        self.log = log
        self.source = next(t for t in tables if t.color is None)
        self.destinations = {t.color: t for t in tables if t.color is not None}
        self.arm_folded = skills.robot.arm_target  # в этой позе рука сложена для езды

    # ---- главный цикл ----

    def run(self):
        result = SortResult()
        for _ in range(self.config.max_rounds):
            result.rounds += 1
            self.observe_source()
            todo = self.deliverable()
            self.log(f"Осмотр {result.rounds}: на столе кубиков к перевозке: {len(todo)}")
            if not todo:
                break
            while todo:
                base = self.skills.robot.base_pose[:2]
                cube = min(todo, key=lambda c: np.linalg.norm(c.position[:2] - base))
                self.deliver(cube)
                todo = self.deliverable()
        result.delivered = self.memory.delivered()
        result.left = self.memory.waiting()
        return result

    def deliverable(self):
        """Ожидающие кубики, для которых есть стол их цвета и свободное место на нём."""
        return [c for c in self.memory.waiting() if self.free_slot(c.color) is not None]

    # ---- зрение ----

    def observe_source(self):
        """Общий взгляд на стол-источник: обновить в памяти всё, что на нём лежит."""
        table = self.source
        near_edge = table.center - table.depth * table.direction
        self.skills.look_over_table(near_edge, table.heading, table.surface_z)
        return self.look(forget_missing=True)

    def look(self, forget_missing):
        """Снять кадр и занести находки на столе-источнике в память. Возвращает увиденные кубики."""
        cfg, table = self.config, self.source
        frame = self.camera.capture()
        detections = [d for d in self.detector.detect(frame, surface_z=table.surface_z)
                      if table.contains(d.position, cfg.table_margin)]
        visible = (lambda p: in_view(frame, p, cfg.view_margin)) if forget_missing else None
        return self.memory.observe(detections, frame.stamp, in_view=visible)

    # ---- один кубик ----

    def free_slot(self, color):
        """Свободное место на столе цвета color или None."""
        table = self.destinations.get(color)
        if table is None:
            return None
        taken = [c.position[:2] for c in self.memory.delivered(table.name)]
        for slot in table.place_slots():
            if all(np.linalg.norm(slot - p) > self.config.slot_radius for p in taken):
                return slot
        return None

    def deliver(self, cube):
        """Взять кубик и отвезти на стол его цвета. Возвращает True, если получилось."""
        try:
            return self._deliver(cube)
        except RuntimeError as error:  # рука не дотягивается до нужной точки
            self.log(f"  не получилось: {error}")
            self.skills.robot.set_gripper(0.0)
            self._back_off()
            self.memory.forget(cube)
            return False

    def _deliver(self, cube):
        cfg, skills, memory = self.config, self.skills, self.memory
        source, table = self.source, self.destinations[cube.color]
        self.log(f"Кубик {cube.id} ({cube.color}): еду к нему")

        # 1. Подъехать и уточнить положение взглядом сверху вблизи.
        if not skills.approach(cube.position, source.heading):
            self.log("  не смог подъехать")
            return self._abandon(cube)
        skills.look_down_at(cube.position, yaw=source.heading)
        # Соседние кубики в этом кадре могут быть обрезаны или закрыты захватом,
        # поэтому отсутствие находки здесь ни о чём не говорит: ничего не забываем.
        if cube not in self.look(forget_missing=False):
            self.log("  вблизи кубика нет на месте, забываю его")
            return self._abandon(cube)

        # 2. Взять. Пустой захват сжимается до упора: так видно, что кубик не взят.
        skills.pick(cube.position, cube.yaw)
        if not skills.holding():
            self.log("  кубик не взят")
            skills.robot.set_gripper(0.0)
            return self._abandon(cube)
        memory.mark_carried(cube)  # на столе-источнике его больше нет
        self._back_off()

        # 3. Отвезти и поставить на свободное место.
        slot = self.free_slot(cube.color)
        place = np.array([slot[0], slot[1], table.surface_z + 0.5 * cfg.cube_size])
        self.log(f"  везу на {table.name}")
        reached = skills.approach(place, table.heading)
        if not reached or not skills.holding():
            self.log("  кубик потерян по дороге" if reached else "  не смог подъехать к столу")
            skills.robot.set_gripper(0.0)
            return self._abandon(cube)
        skills.place(place + [0.0, 0.0, cfg.release_height], table.heading)
        self._back_off()
        memory.mark_delivered(cube, place, table.name)
        return True

    def _abandon(self, cube):
        self._back_off()
        self.memory.forget(cube)
        return False

    def _back_off(self):
        """Сложить руку и отъехать от стола."""
        self.skills.move_arm(self.arm_folded)
        self.skills.retreat()
