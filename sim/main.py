#!/usr/bin/env python3

import rerun as rr
import rerun_urdf as rrurdf

import argparse
import time
import numpy as np
import pinocchio as pin

import control
import contact
import log

# The frames which are capable of colliding with the floor.
CONTACT_FRAMES = ["wrist_3_link", "forearm_link"]


def simulate(model, data, q0, duration=1.0, dt=0.0025):
    # Initial conditions
    q = q0
    qd = np.zeros(model.nv)

    # Loop the simulation.
    for t in np.arange(0.0, duration, step=dt):
        rr.set_time_seconds("sim_time", t)

        # Make sure all of our frame kinematics are up-to-date before performing
        # calculations on the data.
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # Switch these for different control methods
        tau = control.viscous_damping(model, data, q, qd)
        # tau = control.joint_impedance(model, data, t, q, qd)

        # Compute the contact constraints for the various frames in contact.
        frames_in_contact, Jc, gamma = contact.contact_constraint(
            model, data, q, CONTACT_FRAMES
        )

        # Integrate the forward dynamics.
        qdd = pin.forwardDynamics(model, data, q, qd, tau, Jc, gamma)
        qd += qdd * dt
        q = pin.integrate(model, q, qd * dt)  # semi-implicit euler

        # Do some logging.
        log.log_robot_state(model, data)
        log.log_contact_forces(model, data, CONTACT_FRAMES, frames_in_contact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "simulation")

    model_path = "ur5e.urdf"
    print(f"Loading model from '{model_path}'")

    # Set up the logging environment
    log.log_robot_model(model_path)

    # Build and run the simulation
    model = pin.buildModelFromUrdf(model_path)
    data = model.createData()
    q0 = pin.neutral(model)
    simulate(model, data, q0, duration=5.0)

    # Disconnect from rerun
    rr.script_teardown(args)
