import os
import sys
import json


class SVGtoTTF:
    def convert(self, characters_dir, output_dir, config):
        """Convert a directory with SVG images to TrueType Font.

        Calls a subprocess to the run this script with Fontforge Python
        environment.

        Parameters
        ----------
        characters_dir : str
            Path to directory with SVGs to be converted.
        output_dir : str
            Path to output directory.
        config : str
            Path to config file.
        """
        import subprocess
        import platform

        subprocess.run(
            (
                ["ffpython"]
                if platform.system() == "Windows"
                else ["fontforge", "-script"]
            )
            + [
                os.path.abspath(__file__),
                characters_dir,
                output_dir,
                config,
            ]
        )


    def add_glyphs(self, characters_dir):
        """Read and add SVG images as glyphs to the font.

        Walks through the provided directory and uses each ord(character).svg file
        as glyph for the character. Then using the provided config, set the font
        parameters and export TTF file to outdir.

        Parameters
        ----------
        characters_dir : str
            Path to directory with SVGs to be converted.
        """
        space = self.font.createMappedChar(ord(" "))
        space.width = 300

        for k in self.config["glyphs"]:
            # Create character glyph
            g = self.font.createChar(k)
            self.unicode_mapping.setdefault(k, g.glyphname)
            # Get outlines
            src = "{}/{}.svg".format(k, k)
            src = characters_dir + os.sep + src
            g.importOutlines(src, ("removeoverlap", "correctdir"))
            g.removeOverlap()


    def generate_font_file(self, filename, output_dir):
        """Output TTF file.

        Additionally checks for multiple outputs and duplicates.

        Parameters
        ----------
        filename : str
            Output filename.
        output_dir : str
            Path to output directory.
        """
        if filename is None:
            raise NameError("filename not found in config file.")

        outfile = str(
            output_dir
            + os.sep
            + (filename + ".ttf" if not filename.endswith(".ttf") else filename)
        )

        while os.path.exists(outfile):
            outfile = os.path.splitext(outfile)[0] + " (1).ttf"

        sys.stderr.write("\nGenerating %s...\n" % outfile)
        self.font.generate(outfile)


    def convert_main(self, characters_dir, output_dir, config_file):
        try:
            self.font = fontforge.font()
        except:
            import fontforge

        with open(config_file) as f:
            self.config = json.load(f)

        self.font = fontforge.font()
        self.unicode_mapping = {}
        self.add_glyphs(characters_dir)

        # Generate font and save as a .ttf file
        filename = os.path.basename(output_dir)
        self.generate_font_file(filename, output_dir)


if __name__ == "__main__":
    # characters_dir = "characters/Quan"
    # output_dir = "output/Quan"
    # config = "config/Quan_font_config.json"

    # SVGtoTTF().convert_main(characters_dir, output_dir, config)

    if len(sys.argv) != 4:
        raise ValueError("Incorrect call to SVGtoTTF")
    SVGtoTTF().convert_main(sys.argv[1], sys.argv[2], sys.argv[3])
