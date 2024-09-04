"""
This script checks each image's validity and format.
"""

from PIL import Image
import glob

max_height = 1400

for img in glob.glob("./data/unsafe_images/*"):
    orig_img = Image.open(img)
    to_save = False

    if orig_img.mode != "RGB":
        print("Converting", img, "from", orig_img.mode, "to RGB")
        orig_img = orig_img.convert("RGB")
        to_save = True

    if orig_img.size[1] > max_height:
        ratio = max_height / orig_img.size[1]
        new_img = orig_img.resize(
            (int(ratio * orig_img.size[0]), int(ratio * orig_img.size[1]))
        )
        print("Resized img from", orig_img.size, "to", new_img.size)
        orig_img = new_img
        to_save = True

    if to_save:
        orig_img.save(img)
    # print(img, i.size)
