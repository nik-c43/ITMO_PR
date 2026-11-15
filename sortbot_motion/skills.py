"""Движения робота: подъехать, навести руку, взять кубик, поставить кубик.

Код не знает, симулятор под ним или настоящий робот. Ему нужен объект robot с таким интерфейсом:

    dt                           период управления, с
    time                         часы робота, с
    base_pose                    положение базы в мире: (x, y, угол)
    arm_target                   углы, заданные сейчас приводам руки
    set_cmd_vel(vx, vy, wz)      скорость базы в осях робота
    set_arm_target(q)            цель приводов руки
    set_gripper(closed)          0: открыт, 1: закрыт
    gripper_closure              насколько захват сжат на самом деле: 0 открыт, 1 сжат до упора
    solve_ik(pos, mat, q_init)   обратная кинематика, возвращает (углы, сошлось ли)
    step()                       прожить один период управления

В симуляторе step() делает шаг физики, в узле ROS 2 ждёт один период управления.
Длительности движений отсчитываются по часам робота, а не по числу шагов: если цикл
управления не успевает за заданным периодом, движения всё равно длятся столько, сколько задано.
"""
from dataclasses import dataclass, field

import numpy as np


def top_down(yaw):
    """Ориентация захвата «пальцами вниз», линия между пальцами повёрнута на yaw вокруг вертикали.

    Столбцы матрицы: оси захвата в мире. X: линия между пальцами, Z: направление подхода (вниз).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0],
                     [s, -c, 0.0],
                     [0.0, 0.0, -1.0]])


def tilted_down(tilt, heading):
    """Ориентация захвата «вперёд и вниз»: ось подхода отклонена от вертикали на tilt в сторону heading.

    Так камера на запястье смотрит на стол перед роботом сверху под углом.
    """
    forward_down = np.column_stack([[0, 1, 0], [np.cos(tilt), 0, np.sin(tilt)], [np.sin(tilt), 0, -np.cos(tilt)]])
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ forward_down


@dataclass
class MotionConfig:
    """Числа, зависящие от конкретного робота. Значения по умолчанию подобраны для TidyBot."""
    standoff: float = 0.55  # расстояние от центра базы до кубика при захвате, м
    approach_height: float = 0.12  # высота над кубиком, с которой захват опускается, м
    # Высота точки захвата над кубиком при осмотре вблизи. Камера стоит на 12 см выше
    # точки захвата, поэтому видит кубик с 0.34 м: это внутри рабочей зоны D435 (от 0.28 м).
    look_height: float = 0.20
    # Точка захвата находится у основания подушечек пальцев: при сжатии они уходят
    # на 4 см ниже неё. Поэтому целимся выше центра кубика, иначе пальцы упрутся в стол.
    grasp_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.035]))
    # Поза руки «захват смотрит вниз перед роботом»: начальное приближение для обратной кинематики.
    arm_seed: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.6, 3.14, -1.9, 0.0, -0.65, 1.57]))
    # К столу робот подъезжает в два приёма: встаёт напротив на этом отдалении, потом едет прямо.
    # Так он не задевает ножки стола углом базы, пока разворачивается.
    approach_backoff: float = 0.25
    # Общий взгляд на стол: база в overview_distance от края стола, захват вынесен вперёд
    # на overview_reach и поднят на overview_height над столешницей, наклон от вертикали overview_tilt.
    overview_distance: float = 0.65
    overview_reach: float = 0.30
    overview_height: float = 0.50
    overview_tilt: float = np.radians(50)
    # Сжатый пустой захват даёт closure около 1, с кубиком 4 см около 0.57.
    holding_closure: tuple = (0.3, 0.85)
    drive_gain: float = 2.0  # усиление регулятора подъезда, 1/с
    max_speed: float = 0.5  # м/с
    max_turn: float = 1.0  # рад/с
    position_tolerance: float = 0.005  # м
    angle_tolerance: float = 0.01  # рад


class Skills:
    def __init__(self, robot, config=None, on_step=None):
        """on_step вызывается после каждого периода управления (например, обновить окно)."""
        self.robot = robot
        self.config = config or MotionConfig()
        self.on_step = on_step

    # ---- базовые действия ----

    def tick(self):
        self.robot.step()
        if self.on_step is not None:
            self.on_step()

    def wait(self, seconds):
        # Половина периода в запасе: иначе из-за округления выходил бы лишний шаг.
        end = self.robot.time + seconds - 0.5 * self.robot.dt
        while self.robot.time < end:
            self.tick()

    def drive_to(self, x, y, th, timeout=20.0):
        """Доехать до позы (x, y, th) в мире. Возвращает True, если доехал за timeout секунд."""
        cfg, robot = self.config, self.robot
        goal = np.array([x, y, th])
        reached = False
        deadline = robot.time + timeout
        while robot.time < deadline:
            pose = robot.base_pose
            err = goal - pose
            err[2] = (err[2] + np.pi) % (2 * np.pi) - np.pi
            if np.linalg.norm(err[:2]) < cfg.position_tolerance and abs(err[2]) < cfg.angle_tolerance:
                reached = True
                break
            # Ошибка известна в мировых осях, а скорость задаётся в осях робота.
            c, s = np.cos(pose[2]), np.sin(pose[2])
            vx = np.clip(cfg.drive_gain * (c * err[0] + s * err[1]), -cfg.max_speed, cfg.max_speed)
            vy = np.clip(cfg.drive_gain * (-s * err[0] + c * err[1]), -cfg.max_speed, cfg.max_speed)
            wz = np.clip(cfg.drive_gain * err[2], -cfg.max_turn, cfg.max_turn)
            robot.set_cmd_vel(vx, vy, wz)
            self.tick()
        robot.set_cmd_vel(0.0, 0.0, 0.0)
        self.wait(0.3)
        return reached

    def move_arm(self, q_goal, duration=2.0):
        """Плавно перевести цель приводов руки в q_goal за duration секунд."""
        robot = self.robot
        q_start = robot.arm_target
        start = robot.time
        progress = 0.0
        while progress < 1.0:
            # Цель задаётся на конец наступающего периода управления.
            progress = min(1.0, (robot.time - start + robot.dt) / duration)
            s = 3 * progress**2 - 2 * progress**3  # плавный разгон и торможение
            robot.set_arm_target(q_start + s * (q_goal - q_start))
            self.tick()
        self.wait(0.3)

    def move_tool(self, pos, mat, duration=2.0, q_init=None):
        """Поставить точку захвата в pos (мир) с ориентацией mat."""
        q, ok = self.robot.solve_ik(np.asarray(pos, dtype=float), mat, q_init)
        if not ok:
            raise RuntimeError(f"Обратная кинематика не сошлась для точки {np.round(pos, 3)}")
        self.move_arm(q, duration)

    # ---- действия с кубиком ----

    def grasp_base_pose(self, target, heading):
        """Где встать базе, чтобы дотянуться до точки target, подъезжая к ней по направлению heading."""
        d = self.config.standoff
        return target[0] - d * np.cos(heading), target[1] - d * np.sin(heading), heading

    def approach(self, target, heading):
        """Подъехать к точке target по направлению heading: встать напротив, затем прямо вперёд."""
        x, y, th = self.grasp_base_pose(target, heading)
        back = self.config.approach_backoff
        return (self.drive_to(x - back * np.cos(th), y - back * np.sin(th), th)
                and self.drive_to(x, y, th))

    def retreat(self):
        """Отъехать от стола прямо назад."""
        x, y, th = self.robot.base_pose
        back = self.config.approach_backoff
        return self.drive_to(x - back * np.cos(th), y - back * np.sin(th), th)

    def look_over_table(self, edge_point, heading, surface_z):
        """Встать перед столом и навести камеру на столешницу сверху под углом.

        edge_point: середина ближнего к роботу края стола (x, y).
        """
        cfg = self.config
        forward = np.array([np.cos(heading), np.sin(heading)])
        base = np.asarray(edge_point) - cfg.overview_distance * forward
        reached = self.drive_to(base[0], base[1], heading)
        pose = self.robot.base_pose
        tool = np.array([*(pose[:2] + cfg.overview_reach * forward), surface_z + cfg.overview_height])
        self.move_tool(tool, tilted_down(cfg.overview_tilt, heading), q_init=cfg.arm_seed)
        self.wait(0.5)  # дать руке успокоиться перед кадром
        return reached

    def holding(self):
        """Держит ли захват предмет. Вызывать после сжатия: пустой захват сжимается до упора."""
        low, high = self.config.holding_closure
        return low < self.robot.gripper_closure < high

    def look_down_at(self, point, yaw=0.0):
        """Поднять захват над точкой так, чтобы камера на запястье видела её сверху вблизи."""
        cfg = self.config
        target = np.asarray(point) + cfg.grasp_offset + [0.0, 0.0, cfg.look_height]
        self.move_tool(target, top_down(yaw), q_init=cfg.arm_seed)
        self.wait(0.5)  # дать руке успокоиться перед кадром

    def pick(self, cube_position, cube_yaw):
        """Взять кубик с центром в cube_position, повёрнутый на cube_yaw. База уже стоит рядом."""
        cfg = self.config
        grasp = np.asarray(cube_position) + cfg.grasp_offset
        above = grasp + [0.0, 0.0, cfg.approach_height]
        # У куба симметрия 90 градусов: из четырёх равноценных углов берём ближайший
        # к направлению базы, чтобы запястье поворачивалось не больше чем на 45 градусов.
        heading = self.robot.base_pose[2]
        mat = top_down(heading + (cube_yaw - heading + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        self.robot.set_gripper(0.0)
        self.move_tool(above, mat, q_init=cfg.arm_seed)
        self.move_tool(grasp, mat, duration=1.5)
        self.robot.set_gripper(1.0)
        self.wait(1.0)
        self.move_tool(above, mat, duration=1.5)

    def place(self, cube_position, yaw):
        """Поставить взятый кубик так, чтобы его центр оказался в cube_position."""
        cfg = self.config
        release = np.asarray(cube_position) + cfg.grasp_offset
        above = release + [0.0, 0.0, cfg.approach_height]
        mat = top_down(yaw)
        self.move_tool(above, mat, q_init=cfg.arm_seed)
        self.move_tool(release, mat, duration=1.5)
        self.robot.set_gripper(0.0)
        self.wait(1.0)
        self.move_tool(above, mat, duration=1.5)
