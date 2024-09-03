from PIL import Image
import glob

max_height = 2000

for img in glob.glob("./data/unsafe_images/*"):
    i = Image.open(img)
    if i.size[1] > max_height:
        ratio = max_height/i.size[1]
        new_img = i.resize((int(ratio * i.size[0]), int(ratio * i.size[1])))
        print("Resized img from", i.size, "to", new_img.size)
        new_img.save(img)

    # print(img, i.size)