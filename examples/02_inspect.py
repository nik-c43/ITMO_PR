"""Шаг 2. Заглянуть внутрь модели: из чего состоит мир с точки зрения MuJoCo.

Запуск: python examples/02_inspect.py
"""
import mujoco
import numpy as np

from sortbot_sim import SortingSim

JOINT_TYPES = {0: "free (6 степеней свободы)", 1: "ball", 2: "slide (сдвиг)", 3: "hinge (поворот)"}

sim = SortingSim()
m, d = sim.model, sim.data

print("=== Размеры модели ===")
print(f"шаг времени      : {m.opt.timestep} с")
print(f"тел (nbody)      : {m.nbody}")
print(f"сочленений (njnt): {m.njnt}")
print(f"координат (nq)   : {m.nq}   <- длина вектора data.qpos")
print(f"скоростей (nv)   : {m.nv}   <- длина вектора data.qvel")
print(f"приводов (nu)    : {m.nu}   <- длина вектора data.ctrl")
print(f"камер (ncam)     : {m.ncam}")

print("\n=== Сочленения: какая ячейка qpos за что отвечает ===")
for j in range(m.njnt):
    body = m.body(m.jnt_bodyid[j]).name
    name = m.joint(j).name or f"(без имени, тело {body})"
    print(f"qpos[{m.jnt_qposadr[j]:2d}]  {name:32s} {JOINT_TYPES[int(m.jnt_type[j])]}")

print("\n=== Приводы: какая ячейка ctrl чем управляет ===")
for a in range(m.nu):
    rng = m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else "без ограничений"
    print(f"ctrl[{a:2d}]  {m.actuator(a).name:26s} диапазон: {rng}")

print("\n=== Камеры ===")
for c in range(m.ncam):
    print(f"{c}: {m.camera(c).name}")

print("\n=== Ключевые кадры (заранее сохранённые позы) ===")
for k in range(m.nkey):
    print(f"{k}: {m.key(k).name}")

print("\n=== Состояние прямо сейчас ===")
np.set_printoptions(precision=3, suppress=True)
print("время         :", d.time)
print("база (x,y,th) :", sim.base_pose)
print("углы руки     :", sim.arm_q)
print("точка захвата :", d.site_xpos[sim.pinch_site])
for b in range(m.nbody):
    name = m.body(b).name
    if name.startswith("cube_"):
        print(f"{name:13s} :", d.body(b).xpos)

print("\n=== Один шаг физики ===")
sim.set_cmd_vel(0.3, 0.0, 0.0)  # едем вперёд со скоростью 0.3 м/с
for _ in range(500):
    sim.step()
print(f"прошло {d.time:.2f} с, база теперь в", sim.base_pose)
print("контактов в мире:", d.ncon)
