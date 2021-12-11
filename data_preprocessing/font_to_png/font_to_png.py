import os
import sys
import argparse
import pathlib

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Obtaining characters from .ttf/.otf')
parser.add_argument('--font_root', type=str, default='./fonts', help='font directory')
parser.add_argument('--charset', type=str, default='charset.txt', help='characters')
parser.add_argument('--save_root', type=str, default='./saves', help='images directory')
parser.add_argument('--img_size', type=int, help='The size of generated images')
parser.add_argument('--font_size', type=int, help='The font size of generated characters')

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

if __name__ == '__main__':

    args = parser.parse_args()

    ## Open charset txt file
    charset_file = open(args.charset, 'r', encoding='utf-8')
    characters = charset_file.read()


    # Find all font files in font_path
    font_root = pathlib.Path(args.font_root)
    font_paths = font_root.glob('*.[ot]t[fc]')
    font_paths = [str(path) for path in font_paths]
    print(f'Total fonts found: {len(font_paths)}')
    for font_path in font_paths:
        print(font_path)

    # Iterate through fonts
    for font_label, font_path in zip(range(len(font_paths)), font_paths):
        src_font = ImageFont.truetype(font_path, size=args.font_size)

        font_dir = pathlib.Path(args.save_root + '/font_' + str(font_label))
        if not font_dir.exists():
            font_dir.mkdir(parents=True)

        # Iterate through characters in charset
        for char, num in zip(characters, range(len(characters))):
            img = draw_example(char, src_font, args.img_size,
                               (args.img_size - args.font_size) / 2, (args.img_size - args.font_size) / 2)
            img.save(font_dir.joinpath("%04d.png" % num))
