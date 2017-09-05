from features.addimg import add2Db, saveme
from progress_bar.progress import progress
from image.pre_process import images
from features.info import inlocal  # , debug
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--add", required=True,
                help="python script.py -i <IMAGE_PATH>")

args = vars(ap.parse_args())

if inlocal:
    img_path = "/home/smacar/Desktop/data/full/0" + args["add"] + ".jpg"
else:
    img_path = args["add"]

img = images(img_path)

features = []

for i in range(len(img.des)):
    features.append((img, i))

bar = progress("adding", len(features))

add2Db(0, features, bar)
saveme()

bar.finish()
