from features.addimg import add2Db, saveme
from progress_bar.progress import progress
from image.pre_process import images
from features.info import inlocal
from view.out import jasonDump
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--add", required=True,
                help="python script.py -i <IMAGE_PATH>")

ap.add_argument('-d', "--debug", action="store_true",
                help="to debug the program", default=False)

args = vars(ap.parse_args())

debug = args["debug"]

if inlocal:
    img_path = "/home/smacar/Desktop/data/full/0" + args["add"] + ".jpg"
else:
    img_path = args["add"]

try:
    img = images(img_path)

    features = []

    for i in range(len(img.des)):
        features.append((img, i))

    bar = progress("adding", len(features))

    add2Db(0, features, bar)
    saveme()

    bar.finish()

    if not inlocal:
        jasonDump(1)

except Exception as e:
    if debug:
        print e

    if not inlocal:
        jasonDump(0)
