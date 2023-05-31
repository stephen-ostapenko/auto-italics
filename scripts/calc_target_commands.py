#!/bin/python3

import sys
import os
from os import path
import shutil

import numpy as np

from recover_svg import calc_commands_cnt


def train_val_test_split():
    np.random.seed(24)
    
    fonts = list(filter(lambda f: "i" not in f, os.listdir("fonts-src")))
    val_cnt = len(fonts) // 15
    test_cnt = len(fonts) // 15
    
    fonts = np.array(fonts)
    np.random.shuffle(fonts)
    return list(fonts[: -(val_cnt + test_cnt)]), list(fonts[-(val_cnt + test_cnt) : -test_cnt]), list(fonts[-test_cnt :])


def process_font(font_name, cmd_counts):
    os.mkdir("./tmp-svg")
    os.system(f"fonts2svg ./fonts-src/{font_name}i.otf -c 000000 -av -o tmp-svg")

    for it in os.walk("./tmp-svg"):
        for glyph_file in it[2]:
            if (glyph_file.startswith("uni")):
                continue
            
            cmd_counts.append(calc_commands_cnt(path.join(it[0], glyph_file)))

    shutil.rmtree("./tmp-svg")


if (__name__ == "__main__"):
    train_fonts, val_fonts, test_fonts = train_val_test_split()
    print(len(train_fonts), len(val_fonts), len(test_fonts))
    
    cmd_counts = []
    for font_name in test_fonts:
        process_font(font_name.replace(".otf", ""), cmd_counts)

    mean = np.mean(cmd_counts)
    std = np.std(cmd_counts)
    q_0_2 = np.quantile(cmd_counts, 0.2)
    q_0_8 = np.quantile(cmd_counts, 0.8)
    print(
        f"mean: {'{:.3f}'.format(mean)}; " + \
        f"std: {'{:.3f}'.format(std)}; " + \
        f"0.2 quantile: {'{:.3f}'.format(q_0_2)}; " + \
        f"0.8 quantile: {'{:.3f}'.format(q_0_8)}; ",
        end = "\n", flush = True
    )
