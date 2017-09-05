from image.pre_process import images
from tensor_flow.pre_cluster import tfInit
from progress_bar.progress import progress
from features.searcher import searchTree, similarImages, loaddb
from features.info import debug, inlocal
import argparse

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ap = argparse.ArgumentParser()

if inlocal:
    ap.add_argument("-n", "--branch", required=True,
                    help="branch factor")

    ap.add_argument("-i", "--name", required=True,
                    help="image name")

else:
    ap.add_argument("-p", "--path", required=True,
                    help="python script.py <IMAGE_PATH>")

args = vars(ap.parse_args())

timer = True

if args["name"] == "batch":
    batchMode = True
else:
    batchMode = False


def accuracy(name, y, batch):
    if batch:
        global count

        for i in range(len(y)):
            if name == y[i][0]:
                count += 1
                break
    else:
        print "[RESULT] {}: {}".format(name, y)


if timer:
    from time import time
    start = time()

loaddb()

if timer:
    newt = time()
    print "[INFO] T2l :{0:.2f} sec".format(time() - start)

if (inlocal):
    n_clusters = int(args["branch"])
    if batchMode:
        import glob
        global count
        count = 0
        img_path = glob.glob("../data/full1/*.jpg")

    else:
        img_path = ["/home/smacar/Desktop/data/full1/0" +
                    args["name"] + ".jpg"]
else:
    from features.info import n_clusters
    img_path = [args["path"]]

if debug:
    bar = progress("Traversing", len(img_path))

tfObj = tfInit(n_clusters)
tfObj.finalVariable()

for img in img_path:
    imgObj = images(img, 400, False)
    finalObj = similarImages()

    for d in imgObj.des:
        tree_obj = searchTree(tfObj, 0, d)
        finalObj.add_results(tree_obj.result)

    y = finalObj.similar_result()
    if debug:
        bar.update()

    if (debug):
        accuracy(imgObj.name, y, batchMode)

    if not inlocal:
        import json
        # d = dict(status=0, id0=y[0][0])
        d = {'id0': y[0][0], 'id1': y[1][0],
             'id2': y[2][0], 'id3': y[3][0],
             't': str(time() - start) + "sec"}
        print json.dumps(d)

if debug:
    bar.finish()

if (debug and batchMode):
    a = count / float(len(img_path)) * 100
    print "[INFO] Final accuracy - {}%".format(a)
    if time:
        print "[INFO] Avg time - {0:.2f} sec".format(
            (time() - newt) / float(len(img_path)))
