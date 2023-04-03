#!/usr/bin/env python3

import numpy as np
import pinocchio as pin

CONTACT_KP, CONTACT_KD = 500.0, 50.0


def contact_constraint(model, data, q, contact_frames):
    frames_in_contact = []
    J_c = np.zeros((0, model.nv))
    gamma_c = np.zeros(0)

    for frame in contact_frames:
        contact_frame_id = model.getFrameId(frame)

        # Compute the contact frame position and velocity
        z = data.oMf[contact_frame_id].translation[2]
        zd = pin.getFrameVelocity(
            model, data, contact_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).linear[2]

        # Identify if this link is in contact with the ground by it's
        # global-frame Z position.
        #
        # If in contact, then compute the contact Jacobian and a restoring
        # force, per the ground sprint stiffness and damping.
        #
        # TODO: This is not good code.
        in_contact = z < 0.0
        if in_contact:
            J_contact = pin.computeFrameJacobian(
                model, data, q, contact_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_contact = J_contact[2:3, :]  # Use only the translation z-axis row
            gamma_c0 = np.array([CONTACT_KP * z + CONTACT_KD * zd])
            frames_in_contact.append(frame)
        else:
            J_contact = np.zeros((0, model.nv))
            gamma_c0 = np.zeros(0)

        # Stack this contact constraint with the others.
        J_c = np.vstack((J_c, J_contact))
        gamma_c = np.hstack((gamma_c, gamma_c0))

    return frames_in_contact, J_c, gamma_c
