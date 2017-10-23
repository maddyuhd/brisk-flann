'''
To Do:
    -- When 2 Construct ?
    -- Final check - better method?
    -- Features
        - Replace
        - Duplicate
    -- Threading
        - add.py
        - construct.py
    -- Tweak parameter
    -- Bug - del tree[node]
    -- CleanUp
        -search.py

Change Log:
-- V2.0.0
    --- CleanUp
        --searcher.py
        --search2.py
    --- Indexer Class
    # -- Hdf5 implement
    # -- Redis implement
    # -- Handle multiple users (add.py)
-- V1.2.0
    --- Logging
    --- cleanup clear.py
-- V1.1.2
    --- search2.py with app (feature points)
    --- cleanup tf.cluser
'''
import glob
import argparse
from tensor_flow.pre_cluster import tfInit
from tree.construct import constructMe
from tree.indexer import index
from features.info import inlocal, success, failed
from log.log import logInfo

log = logInfo("[RECONSTRUCT]")

ap = argparse.ArgumentParser()

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
    if inlocal:
        rootDir = '/home/smacar/Desktop/data/10/*.jpg'
        n_clusters = int(args["branch"])
        max_size_lev = int(args["leafSize"])
    else:
        from features.info import n_clusters, max_size_lev
        rootDir = str(args["build"]) + "*.jpg"

    imagList = glob.glob(rootDir)

    features = index(imagList)

    tfObj = tfInit(n_clusters, max_size_lev)

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
