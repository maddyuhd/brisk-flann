from features.addimg import add2Db, saveme
from progress_bar.progress import progress
from image.pre_process import images
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-a", "--add", required=True,
                help="add2db i/p image path")

args = vars(ap.parse_args())

add = args["add"]

img_path = "/home/smacar/Desktop/data/full/0155.jpg"
img = images(img_path)

features = []

for i in range(len(img.des)):
    features.append((img, i))

bar = progress("adding", len(features))

add2Db(0, features, bar)
saveme()
bar.finish()
