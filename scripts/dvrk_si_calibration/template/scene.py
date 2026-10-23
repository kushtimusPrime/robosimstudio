# SPDX-License-Identifier: Apache-2.0
"""SutureBot inspection with a scope-mounted camera and controller tool frames."""

from dataset_browser import install
from multisession_preview import build, replay

ACTIVE_PROFILE = "setup_017"
APPEARANCE = False  # Optional instrument materials, visual bevels and scope lighting.
scene, sim, profile = build(ACTIVE_PROFILE, appearance=APPEARANCE)


def show_frame(sim, session, image_index=0, episode=None):
    replay(sim, session, profile, image_index, episode)


def browse_suturebot(session):
    install(session)
    session.set_paused(True)


def inspect_ik(session):
    replay(sim, session, profile, 0)
    session.set_paused(False)


routines = {"Inspect IK": inspect_ik, "Browse SutureBot": browse_suturebot}
