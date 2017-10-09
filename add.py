import argparse
from features.addimg import add2Db
from features.info import inlocal, success, failed
from tensor_flow.pre_cluster import tfInit
from tree.indexer import index
from view.out import jsonDump
from log.log import logInfo

log = logInfo("[ADD]")

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--add", required=True,
                help="python script.py -i <IMAGE_PATH>")

ap.add_argument('-d', "--debug", action="store_true",
                help="to debug the program", default=False)

if inlocal:
    ap.add_argument("-n", "--branch", required=True,
                    help="branch factor")
    ap.add_argument("-l", "--leafSize", required=True,
                    help="leaf size factor")

args = vars(ap.parse_args())

debug = args["debug"]

try:
    if inlocal:
        img_path = ["/home/smacar/Desktop/data/full/0" + args["add"] + ".jpg"]
        n_clusters, max_size_lev = int(args["branch"]), int(args["leafSize"])

    else:
        from features.info import n_clusters, max_size_lev
        img_path = [args["add"]]

    features = index(img_path)

    tfObj = tfInit(n_clusters, max_size_lev)

    add2Db(0, features, tfObj, debug)

    if not inlocal:
        log.dump(1, success)
        jsonDump(1)

except Exception as e:
    if debug:
        print e

    if not inlocal:
        jsonDump()
        log.dump(3, failed + str(e))
