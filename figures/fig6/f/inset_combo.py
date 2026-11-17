"""Place wallprune_G20_combo.png as an inset in the top right of wallprune_G20.png.

Writes wallprune_G20_inset.png next to the inputs (the base plot is left untouched,
since plot_wallprune.py regenerates it).
"""

import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "wallprune_G20.png")
COMBO = os.path.join(HERE, "wallprune_G20_combo.png")
OUT = os.path.join(HERE, "wallprune_G20_inset.png")

INSET_FRAC = 0.46    # inset width as a fraction of the base figure width
MARGIN_X = 0.075     # gap from the right edge (fraction of base width); keeps it inside the axes
MARGIN_Y = 0.115     # gap from the top edge (fraction of base height), clears the title
BORDER_PX = 4

base = Image.open(BASE).convert("RGB")
combo = Image.open(COMBO).convert("RGB")

side = round(base.width * INSET_FRAC)
combo = combo.resize((side, side), Image.LANCZOS)

x0 = base.width - side - round(base.width * MARGIN_X)
y0 = round(base.height * MARGIN_Y)
base.paste(combo, (x0, y0))
ImageDraw.Draw(base).rectangle([x0, y0, x0 + side - 1, y0 + side - 1],
                               outline="#282828", width=BORDER_PX)

base.save(OUT)
print(f"Saved {OUT}")
