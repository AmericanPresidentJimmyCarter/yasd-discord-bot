from discord import app_commands

from constants import (
    DEFAULT_IMAGE_HEIGHT_WIDTH,
    MIN_IMAGE_HEIGHT_WIDTH,
)

CUTOFF_32 = 768
CUTOFF_64 = 1280

def generate_height_and_width_choices(
    max_size: int,
) -> list[app_commands.Choice]:
    '''
    Generate a list of choices in pixel sizes based on the user specified
    maximum height.
    '''
    step = 32
    if max_size > CUTOFF_32:
        step = 64
    if max_size > CUTOFF_64:
        step = 128

    valid = sorted(list(set(map(
        lambda x: MIN_IMAGE_HEIGHT_WIDTH + (step * x),
        range(int((max_size - MIN_IMAGE_HEIGHT_WIDTH) / step) + 1))) - \
            set([DEFAULT_IMAGE_HEIGHT_WIDTH])))
    if max_size not in valid:
        valid.append(max_size)
    choices = [
        app_commands.Choice(name="512 (Default)", value=512),
    ] + [
        app_commands.Choice(name=str(val), value=val)
        for val in valid
    ]

    return choices
