#!/bin/python3

import sys
import os
import numpy as np
import shutil
from svgpathtools import svg2paths
from tqdm import tqdm


def calc_commands_cnt(path_to_img):
    paths, attrs = svg2paths(path_to_img)
    
    command_cnt = 0
    for p in paths:
        command_cnt += len(p)

    return command_cnt


def recover_svg(input_path, output_path, img_name):
    bmp_img = img_name.replace("png", "bmp")
    svg_img = img_name.replace("png", "svg")
    
    os.system(f"magick {input_path}/{img_name} {output_path}/{bmp_img}")
    os.system(f"potrace --svg {output_path}/{bmp_img} -o {output_path}/{svg_img}")

    return calc_commands_cnt(f"{output_path}/{svg_img}")


def process_dir(input_path, output_path):
    cmd_counts = []
    
    for img_name in tqdm(sorted(os.listdir(input_path))):
        cmd_counts.append(recover_svg(input_path, output_path, img_name))

    return (
        np.mean(cmd_counts),
        np.std(cmd_counts),
        np.quantile(cmd_counts, 0.2),
        np.quantile(cmd_counts, 0.8)
    )


if (__name__ == "__main__"):
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    shutil.rmtree(output_path)
    os.mkdir(output_path)

    print(process_dir(input_path, output_path))
