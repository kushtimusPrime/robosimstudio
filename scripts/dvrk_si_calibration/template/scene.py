# SPDX-License-Identifier: Apache-2.0
"""SutureBot inspection with a scope-mounted camera and controller tool frames."""

from dataset_browser import install
from multisession_preview import build, replay

ACTIVE_PROFILE = "setup_017"
APPEARANCE = False  # Optional instrument materials, visual bevels and scope lighting.
EXTERIOR = False  # Optional scan-informed exterior; preserves calibrated distal visuals.
scene, sim, profile = build(ACTIVE_PROFILE, appearance=APPEARANCE, exterior=EXTERIOR)


def show_frame(sim, session, image_index=0, episode=None):
    replay(sim, session, profile, image_index, episode)


def browse_suturebot(session):
    install(session)
    session.set_paused(True)


def inspect_ik(session):
    replay(sim, session, profile, 0)
    session.set_paused(False)


def inspect_exterior(session):
    from exterior_capture import set_browser_view

    show_frame(sim, session, 94)
    set_browser_view(sim, session, profile, "robot")


def inspect_housing(session):
    from exterior_capture import set_browser_view

    show_frame(sim, session, 94)
    set_browser_view(sim, session, profile, "tool")


def inspect_forceps(session):
    from exterior_capture import set_browser_view

    show_frame(sim, session, 94)
    set_browser_view(sim, session, profile, "forceps")


def inspect_scope(session):
    from exterior_capture import set_browser_view

    show_frame(sim, session, 94)
    set_browser_view(sim, session, profile, "scope_collar")


def inspect_carriage(session):
    from exterior_capture import set_browser_view

    show_frame(sim, session, 94)
    set_browser_view(sim, session, profile, "mount")


def page_action(action):
    """Adapt an immediate view action to Studio's (sim, session) generator API."""

    def run(sim, session):
        action(session)
        yield from ()

    return run


routines = {
    "Inspect IK": inspect_ik,
    "Browse SutureBot": browse_suturebot,
    "Inspect exterior": inspect_exterior,
    "Needle driver housing": inspect_housing,
    "DeBakey housing": inspect_forceps,
    "Scope collar": inspect_scope,
    "Instrument carriage": inspect_carriage,
}

routines = {name: page_action(action) for name, action in routines.items()}
