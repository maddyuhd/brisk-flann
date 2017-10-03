'''
To Do:
    -- When 2 Construct ?
    -- Final check - better method?
    -- Features
        - Replace
        - Duplicate
    -- Hdf5 implement
    -- Redis implement
    -- Threading
        - add.py
        - construct.py
    -- Tweak parameter
    -- Bug - del tree[node]
    -- CleanUp
        -search2.py
        -search.py
        -searcher.py
    -- Indexer Class :)

Change Log:
-- V1.2.0
    -- Handle multiple users (add.py)
    --- Logging
    --- cleanup clear.py
-- V1.1.2
    --- search2.py with app (feature points)
    --- cleanup tf.cluser
'''
import glob
from image.pre_process import images
from tensor_flow.pre_cluster import tfInit
from features.construct import constructMe
from features.info import inlocal, success, failed
from log.log import logInfo
import argparse

log = logInfo("[RECONSTRUCT]")

ap = argparse.ArgumentParser()

# if __name__ == "__main__":

ap.add_argument('-d', "--debug", action="store_true", help="Debug",
                default=False)

if inlocal:
    ap.add_argument("-n", "--branch", required=True, help="branch")
    ap.add_argument("-l", "--leafSize", required=True, help="leaf size")
else:
    ap.add_argument("-i", "--build", required=True, help="path src images")

args = vars(ap.parse_args())

debug = args["debug"]

try:
    if (inlocal):
        rootDir = '/home/smacar/Desktop/data/full/*.jpg'
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

    log.dump(1, success)

except Exception as e:
    if debug:
        print e

    if not inlocal:
        log.dump(3, failed + str(e))
