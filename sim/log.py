#!/usr/bin/env python3

import rerun as rr
import rerun_urdf
from scipy.spatial.transform import Rotation


def log_robot_model(urdf_path):
    urdf = rerun_urdf.load_urdf_from_file(urdf_path)
    rerun_urdf.log_scene(
        scene=urdf.scene, node=urdf.base_link, path="robot", timeless=True
    )
    rr.log_view_coordinates("/", xyz="FLU", timeless=True)
    rr.log_rect(
        "floor",
        color=(129 / 255, 212 / 255, 250 / 255),
        rect=(-1.0, -1.0, 2.0, 2.0),
        timeless=True,
    )


def log_contact_forces(model, data, contact_frames, frames_in_contact):
    for i, frame in enumerate(contact_frames):
        contact_frame_id = model.getFrameId(frame)
        oMc = data.oMf[contact_frame_id]

        if frame in frames_in_contact:
            cidx = frames_in_contact.index(frame)

            rr.log_scalar(f"contact_force[{i}]", data.lambda_c[cidx])
            rr.log_arrow(
                f"robot/world/contact_force[{i}]",
                origin=oMc.translation,
                vector=[0.0, 0.0, 0.002 * data.lambda_c[cidx]],
                width_scale=0.025,
            )
        else:
            rr.log_scalar(f"contact_force[{i}]", 0.0)
            rr.log_arrow(
                f"robot/world/contact_force[{i}]",
                origin=oMc.translation,
                vector=[0.0, 0.0, 0.0],
                width_scale=0.0,
            )


# Compute a transform which, when applied to a point in the child frame,
# transforms that point into the parent frame.
def compute_rel_tfm(model, data, parent, child):
    oMp = data.oMf[model.getFrameId(parent)]
    oMc = data.oMf[model.getFrameId(child)]
    pMc = oMp.actInv(oMc)
    return pMc


def log_robot_state(model, data, q=None, qd=None):
    # TODO: Link <-> scene graph names are hardcoded. This is not good code.
    frames = [
        "world",
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]

    path = "robot/" + frames[0]
    for i, frame_name in list(enumerate(frames))[1:]:
        parent_name = frames[i - 1]
        path = path + "/" + frame_name

        if model.existFrame(frame_name):
            pMc = compute_rel_tfm(model, data, parent_name, frame_name)
            R = Rotation.from_matrix(pMc.rotation)
            rr.log_rigid3(path, parent_from_child=(pMc.translation, R.as_quat()))

            if frame_name == "base_link":
                rr.log_point(
                    path + "/point",
                    position=[0.0, 0.0, 0.0],
                    radius=0.01,
                    label=frame_name,
                )

    if q is not None and qd is not None:
        for i, joint in enumerate(model.joints):
            rr.log_scalar("q/" + model.names[i], q[model.idx_qs[i]])
            rr.log_scalar("qd/" + model.names[i], qd[model.idx_vs[i]])
