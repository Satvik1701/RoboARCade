"""Tests of the gamepad task."""


import json

import pytest

from core.tasks import TASKS
from helpers import check_mating_insertion, check_mating_pickup, check_mating_surface


MATING = json.loads((TASKS / "gamepad" / "manifest.json").read_text())["mjfit"]["parameters"]


def test_interface_surface_matches_visual_with_configured_clearance():
    check_mating_surface("gamepad", MATING)


def test_scripted_pickup_and_lift():
    check_mating_pickup("gamepad", MATING["target"])


@pytest.mark.parametrize("seed", range(1000000, 1000005))
def test_insertion_path_and_free_release(seed):
    check_mating_insertion("gamepad", MATING, seed)
