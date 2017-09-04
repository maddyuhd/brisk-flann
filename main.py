import glob
from image.pre_process import images
from tensor_flow.pre_cluster import tfInit
from db.pick import saveFile
from features.construct import constructTree
from features.info import inlocal  # , debug
import argparse

ap = argparse.ArgumentParser()
if __name__ == "__main__":

    if inlocal:
        ap.add_argument("-n", "--branch", required=True,
                        help="branch factor")
        ap.add_argument("-l", "--leafSize", required=True,
                        help="leaf size factor")
    else:
        ap.add_argument("-b", "--build", required=True,
                        help="path to source images")

    args = vars(ap.parse_args())

    if (inlocal):
        rootDir = '../data/1/*.jpg'
        n_clusters = int(args["branch"])
        max_size_lev = int(args["leafSize"])
    else:
        from features.info import n_clusters, max_size_lev
        rootDir = args["build"]
    features = []
    fileList = glob.glob(rootDir)

    for img_path in fileList:
        img = images(img_path)

        for i in range(len(img.des)):
            features.append((img, i))

    tfObj = tfInit(n_clusters, max_size_lev)
    tfObj.clusterVar()
    tfObj.finalVariable()

    obj = constructTree(0, features, tfObj)

    saveFile(obj.tree, "tree", inlocal)
    saveFile(obj.imagesInLeaves, "imagesInLeaves", inlocal)
    saveFile(obj.nodes, "nodes", inlocal)
    saveFile(obj.nodeIndex, "nodeIndex", inlocal)

# print("[INFO] indexed {} images, {} vectors".format(
#     len(fileList), len(features)))


# remove try and catch
# += 1
# proper logging for each action
# for i in array # not with index
