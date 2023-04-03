#!/usr/bin/env python3

import numpy as np
import rerun as rr
import pinocchio as pin

VISCOUS_KD = 5.0
IMPEDANCE_KP, IMPEDANCE_KD = 50.0, 5.0


# Apply damping opposing the motion.
def viscous_damping(model, data, q, qd):
    cmd = -VISCOUS_KD * qd
    for i in range(0, model.nv):
        rr.log_scalar(f"viscous_damping/tau/{i}", cmd[i])

    return cmd


def joint_impedance(model, data, t, q, qd):
    # Make up a fake sinewave trajectory for one of the joints.
    # TODO: This is not good code.
    amp = 0.5
    freq = 0.5 * 2.0 * np.pi

    # Generate a reference position trajectory
    q_ref = pin.neutral(model)
    q_ref[1] = amp * np.sin(freq * t)

    # Generate a reference velocity trajectory (differentiate q_ref)
    qd_ref = np.zeros(model.nv)
    qd_ref[1] = amp * freq * np.cos(freq * t)

    # Generate a reference acceleration trajectory (differentiate qd_ref)
    qdd_ref = np.zeros(model.nv)
    qdd_ref[1] = -amp * freq**2 * np.sin(freq * t)

    # Compute feedforward torques based on the reference trajectory.
    tau_ff = pin.rnea(model, data, q_ref, qd_ref, qdd_ref)

    cmd = (
        IMPEDANCE_KP * pin.difference(model, q, q_ref)
        + IMPEDANCE_KD * (qd_ref - qd)
        + tau_ff
    )

    for i in range(0, model.nv):
        rr.log_scalar(f"joint_impedance/q_ref/{i}", q_ref[i])
        rr.log_scalar(f"joint_impedance/qd_ref/{i}", qd_ref[i])

        rr.log_scalar(f"joint_impedance/q_meas/{i}", q[i])
        rr.log_scalar(f"joint_impedance/qd_meas/{i}", qd[i])

        rr.log_scalar(f"joint_impedance/tau_ff/{i}", tau_ff[i])
        rr.log_scalar(f"joint_impedance/cmd/{i}", cmd[i])

    return cmd
