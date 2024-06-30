import os
import re
import shutil
import tempfile
import tkinter
from tkinter import filedialog

from src.sheettopng import SHEETtoPNG
from src.pngtosvg import PNGtoSVG
from src.svgtottf import SVGtoTTF


def run(sheets_dir, output_dir, characters_dir, config):
    SHEETtoPNG().convert(sheets_dir, characters_dir, cols=8, rows=8)
    PNGtoSVG().convert(characters_dir)
    SVGtoTTF().convert(characters_dir, output_dir, config)


def converters(sheets_dir, output_dir=None, characters_dir=None, config=None):
    if not characters_dir:
        characters_dir = tempfile.mkdtemp()
        isTempdir = True
    else:
        isTempdir = False

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"output/{os.path.basename(sheets_dir)}"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if config is None:
        config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "config/default.json"
        )

    run(sheets_dir, output_dir, characters_dir, config)

    if isTempdir:
        shutil.rmtree(characters_dir)

def main(data_dir_path):
    sheets_dirs_name = os.listdir(data_dir_path)
    sheets_dirs_name = sorted(sheets_dirs_name, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for sheets_dir_name in sheets_dirs_name:
        sheets_dir = os.path.join(data_dir_path, sheets_dir_name)

        try:
            if len(sheets_dir) == 0:
                print("You haven't chosen the sheets directory!")
            else:
                # characters_dir = filedialog.askdirectory(title='Choose characters directory')
                output_parent_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "output"
                )
                if not os.path.exists(output_parent_dir):
                    os.makedirs(output_parent_dir)

                metadata = {"filename": None, "family": None, "style": None}
                # converters(sheets_dir=sheets_dir, characters_dir=characters_dir)
                converters(sheets_dir=sheets_dir)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # run a file
    # select images' folder
    tkinter.Tk().withdraw() # prevent an empty window from appearing
    sheets_dir = filedialog.askdirectory(title='Choose sheets directory')

    if len(sheets_dir) == 0:
        print("You haven't chosen the sheets directory!")
    else:
        characters_dir = filedialog.askdirectory(title='Choose characters directory')
        output_parent_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "output"
        )
        if not os.path.exists(output_parent_dir):
            os.makedirs(output_parent_dir)

        metadata = {"filename": None, "family": None, "style": None}
        # converters(sheets_dir=sheets_dir, characters_dir=characters_dir)
        converters(sheets_dir=sheets_dir)

    # run multiple files
    # data_dir_path = 'sheets'
    # main(data_dir_path)
