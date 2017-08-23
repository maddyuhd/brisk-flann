import sys
import os
from image.pre_process import images
from features.construct import constructTree
from tensor_flow.pre_cluster import tfInit
from db.pick import saveFile


inlocal = True
# inlocal = False

if __name__ == "__main__":

    if (inlocal):
        n_clusters = int(sys.argv[1])
        max_size_lev = int(sys.argv[2])
    else:
        n_clusters = 8
        max_size_lev = 500

    features = []
    rootDir = '../data/1'
    fileList = sorted(os.listdir(rootDir))

    for imgname in fileList:
        img_path = rootDir + '/' + str(imgname)

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
    # print("[INFO] indexed {} images, {} vectors".format(
    #     len(fileList), len(features)))
