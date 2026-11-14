"""Обратная кинематика руки: «куда поставить захват» -> «какие углы сочленений».

Метод: итерации с демпфированной псевдообратной матрицей Якоби (damped least squares).
"""
import mujoco
import numpy as np


def solve_ik(sim, target_pos, target_mat, q_init=None, iters=300, tol=1e-4, damping=0.05, max_step=0.2):
    """Найти углы руки, при которых захват окажется в заданной позе.

    target_pos: точка в мире, куда должна попасть точка захвата (pinch_site).
    target_mat: матрица 3x3, ориентация захвата в мире.
    q_init: начальное приближение (по умолчанию текущие углы руки).

    Возвращает (q, ok): углы семи сочленений и признак, что решение сошлось.
    База считается неподвижной, решаем только для руки.
    """
    model = sim.model
    # Считаем на копии состояния, чтобы не трогать саму симуляцию.
    data = mujoco.MjData(model)
    data.qpos[:] = sim.data.qpos
    if q_init is not None:
        data.qpos[sim.arm_qadr] = q_init

    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, np.asarray(target_mat, dtype=float).flatten())

    arm_joint_ids = model.dof_jntid[sim.arm_dofadr]
    limited = model.jnt_limited[arm_joint_ids].astype(bool)
    lower, upper = model.jnt_range[arm_joint_ids].T

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    site_quat, site_quat_inv, err_quat = np.zeros(4), np.zeros(4), np.zeros(4)
    err = np.zeros(6)

    ok = False
    for _ in range(iters):
        # Прямая кинематика: по углам вычислить, где сейчас захват.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # Ошибка положения и ошибка ориентации (вектор поворота в мировых осях).
        err[:3] = target_pos - data.site_xpos[sim.pinch_site]
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[sim.pinch_site])
        mujoco.mju_negQuat(site_quat_inv, site_quat)
        mujoco.mju_mulQuat(err_quat, target_quat, site_quat_inv)
        mujoco.mju_quat2Vel(err[3:], err_quat, 1.0)

        if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 10 * tol:
            ok = True
            break

        # Якобиан: как смещается и поворачивается захват при малом изменении углов.
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, sim.pinch_site)
        jac = np.vstack([jac_pos, jac_rot])[:, sim.arm_dofadr]

        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
        dq *= min(1.0, max_step / (np.abs(dq).max() + 1e-12))

        q = data.qpos[sim.arm_qadr] + dq
        q[limited] = np.clip(q[limited], lower[limited], upper[limited])
        data.qpos[sim.arm_qadr] = q

    return data.qpos[sim.arm_qadr].copy(), ok
