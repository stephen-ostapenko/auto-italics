#!/bin/python3

import sys
import os
import shutil

import numpy as np
np.random.seed(24)

from tqdm import tqdm

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageChops
from PIL.Image import Resampling


def get_glyph_image(glyph, font_name, glyph_size, img_size):
    font = ImageFont.truetype(f"fonts-src/{font_name}.otf", glyph_size - glyph_size // 10)

    image = Image.new("L", (img_size, img_size), "white")
    draw = ImageDraw.Draw(image)
    
    offset_w, offset_h = font.getoffset(glyph)
    w, h = draw.textsize(glyph, font = font)
    pos = ((img_size - w - offset_w) / 2, (img_size - h - offset_h) / 2)

    draw.text(pos, glyph, "black", font = font)

    return image


def draw_glyph_PNG(image, glyph_name, font_name, output_folder):
    if (not os.path.exists(f"{output_folder}")):
        os.mkdir(f"{output_folder}")
    
    if (not os.path.exists(f"{output_folder}/{font_name}")):
        os.mkdir(f"{output_folder}/{font_name}")
    
    image.save(f"{output_folder}/{font_name}/{glyph_name}.png")


def get_all_chars_from_font(font_name):
    with TTFont(f"fonts-src/{font_name}.otf") as font:
        characters = []
        for t in font["cmap"].tables:
            if (not t.isUnicode()):
                continue
            
            for c in t.cmap.items():
                characters.append((str(chr(c[0])), c[1]))
                
        return set(characters)


def get_glyph_size(glyph, font_name, img_size):
    image = Image.new("L", (img_size, img_size), "white")
    draw = ImageDraw.Draw(image)
    
    l, r = 1, img_size * 4
    while (l + 1 < r):
        m = (l + r) // 2
        
        re_font = ImageFont.truetype(f"fonts-src/{font_name}.otf", m)
        it_font = ImageFont.truetype(f"fonts-src/{font_name}i.otf", m)
        
        re_w, re_h = draw.textsize(glyph, font = re_font)
        it_w, it_h = draw.textsize(glyph, font = it_font)
        
        if (re_w > img_size or re_h > img_size or it_w > img_size or it_h > img_size):
            r = m
        else:
            l = m
    
    return l


def check_glyph_equality(glyph, font_name, img_size):
    re_font = ImageFont.truetype(f"fonts-src/{font_name}.otf", img_size // 2)
    it_font = ImageFont.truetype(f"fonts-src/{font_name}i.otf", img_size // 2)

    re_image = Image.new("L", (img_size, img_size), "white")
    re_draw = ImageDraw.Draw(re_image)
    it_image = Image.new("L", (img_size, img_size), "white")
    it_draw = ImageDraw.Draw(it_image)

    re_draw.text((img_size // 8, img_size // 8), glyph, "black", font = re_font)
    it_draw.text((img_size // 8, img_size // 8), glyph, "black", font = it_font)

    diff = ImageChops.difference(re_image, it_image)
    return (diff.getbbox() is None)


def draw_font_set_PNG(font_name, re_output, it_output, img_size):
    re_chars = get_all_chars_from_font(font_name)
    it_chars = get_all_chars_from_font(font_name + "i")
    
    chars = re_chars.intersection(it_chars)
    
    for glyph, glyph_name in chars:
        if (glyph.isspace()):
            continue

        if (glyph_name == ".null"):
            continue
            
        if (ord(glyph[0]) > 0x2116):
            continue
            
        glyph_size = get_glyph_size(glyph, font_name, img_size)

        re_img = get_glyph_image(glyph, font_name, glyph_size, img_size)
        it_img = get_glyph_image(glyph, font_name + "i", glyph_size, img_size)
        
        draw_glyph_PNG(re_img, glyph_name, font_name, re_output)
        draw_glyph_PNG(it_img, glyph_name, font_name + "i", it_output)


def draw_fonts_PNG(re_output, it_output, img_size):
    if (os.path.exists(re_output)):
        shutil.rmtree(re_output)
        os.mkdir(re_output)
        
    if (os.path.exists(it_output)):
        shutil.rmtree(it_output)
        os.mkdir(it_output)
    
    re_fonts = sorted(list(filter(lambda f: "i" not in f, os.listdir("fonts-src"))))
    it_fonts = sorted(list(filter(lambda f: "i" in f, os.listdir("fonts-src"))))
    assert(re_fonts == list(map(lambda s: s.replace("i", ""), it_fonts)))
    
    for font_name in tqdm(re_fonts):
        draw_font_set_PNG(font_name.replace(".otf", ""), re_output, it_output, img_size)


IMAGE_SIZE = int(sys.argv[1])
CHANNELS_CNT = 1

def draw_all_fonts(img_size):
    draw_fonts_PNG(f"fonts-re-{img_size}", f"fonts-it-{img_size}", img_size)
    
    pass


draw_all_fonts(img_size = IMAGE_SIZE)
