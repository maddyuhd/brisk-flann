'''

'''
import glob
from image.pre_process import images
from tensor_flow.pre_cluster import tfInit
from features.construct import constructMe
from features.info import inlocal
import argparse

ap = argparse.ArgumentParser()

if __name__ == "__main__":

    ap.add_argument('-d', "--debug", action="store_true",
                    help="to debug the program", default=False)

    if inlocal:
        ap.add_argument("-n", "--branch", required=True,
                        help="branch factor")
        ap.add_argument("-l", "--leafSize", required=True,
                        help="leaf size factor")
    else:
        ap.add_argument("-i", "--build", required=True,
                        help="path to source images")

    args = vars(ap.parse_args())

    debug = args["debug"]

    if (inlocal):
        rootDir = '../data/1/*.jpg'
        n_clusters = int(args["branch"])
        max_size_lev = int(args["leafSize"])
    else:
        from features.info import n_clusters, max_size_lev
        rootDir = str(args["build"]) + "*.jpg"

    features = []
    imagList = glob.glob(rootDir)

    for img_path in imagList:
        img = images(img_path)

        for i in range(len(img.des)):
            features.append((img, i))

    tfObj = tfInit(n_clusters, max_size_lev)
    tfObj.clusterVar()
    tfObj.finalVariable()

    tree = constructMe(0, features, tfObj, debug)
    tree.saveDb(inlocal)

if debug:
    print("[INFO] indexed {} images, {} vectors".format(
        len(imagList), len(features)))


# remove try and catch
# += 1
# for i in array # not with index
