# adapted from pettingzoo.classic.connect_four
import glob
import os
import re
from pathlib import Path
from typing import Union

import pygame


def get_image(path):

    cwd = os.path.dirname(__file__)
    image = pygame.image.load(os.path.join(cwd, path))
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def load_chip(tile_size, filename, scale):

    chip = get_image(os.path.join("img", filename))
    chip = pygame.transform.scale(
        chip, (int(tile_size * (scale)), int(tile_size * (scale)))
    )
    return chip


def load_chip_preview(tile_size, filename, scale):

    chip = get_image(os.path.join(os.path.join("img", "preview"), filename))
    chip = pygame.transform.scale(
        chip, (int(tile_size * (scale)), int(tile_size * (scale)))
    )
    return chip

# from https://github.com/michaelfeil/skyjo_rl/blob/dev/rlskyjo/utils.py


def get_project_root() -> Path:

    return Path(__file__).parent.parent.parent.resolve()


def find_file_in_subdir(
    parent_dir: Union[Path, str], file_str: Union[Path, str], regex_match: str = None
) -> Union[str, None]:

    files = glob.glob(os.path.join(parent_dir, "**", file_str), recursive=True)
    if regex_match is not None:
        p = re.compile(regex_match)
        files = [s for s in files if p.match(s)]
    return sorted(files)[-1] if len(files) else None

