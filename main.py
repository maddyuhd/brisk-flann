import sys
import os
from construct import inlocal, progress, images, constructTree
from construct import tree, imagesInLeaves, nodes, tfInit, saveFile

if (inlocal):
    n_clusters = int(sys.argv[1])
    max_size_lev = int(sys.argv[2])
else:
    n_clusters = 8
    max_size_lev = 500


features = []
rootDir = 'data/1'
fileList = sorted(os.listdir(rootDir))

for imgname in fileList:
    img_path = rootDir + '/' + str(imgname)

    img = images(img_path)

    for i in range(len(img.des)):
        features.append((img, i))

tfObj = tfInit(n_clusters, max_size_lev)
tfObj.clusterVar()
tfObj.finalVariable()

bar = progress("Constructing", len(features))

constructTree(0, features, tfObj, bar)

bar.finish()

saveFile(tree, "tree", inlocal)
saveFile(imagesInLeaves, "imagesInLeaves", inlocal)
saveFile(nodes, "nodes", inlocal)
# print("[INFO] indexed {} images, {} vectors".format(
#     len(fileList), len(features)))
