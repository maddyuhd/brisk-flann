from main import inlocal
from image.pre_process import images
from tensor_flow.pre_cluster import tfInit
from progress_bar.progress import progress
from features.searcher import searchTree, similarImages
import sys
from time import time
import os

start = time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
debug = True


def accuracy(name, y):
    global count
    for i in range(len(y)):
        if name == y[i][0]:
            count += 1
            break


if debug:
    global count
    count = 0


if (inlocal):
    import glob
    img_path = glob.glob("../data/2/*.jpg")

    n_clusters = int(sys.argv[1])
    # imgname = str(sys.argv[1])
    # img_path = "data/2/00"+imgname+".jpg"

else:
    img_path = [str(sys.argv[1])]
    from construct import n_clusters


bar = progress("Traversing", len(img_path))

tfObj = tfInit(n_clusters)
tfObj.finalVariable()

for img in img_path:
    imgObj = images(img, 380, False)
    finalObj = similarImages()

    for d in imgObj.des:
        tree_obj = searchTree(tfObj, 0, d)
        finalObj.add_results(tree_obj.result)

    y = finalObj.similar_result()
    bar.update()

    if (debug):
        accuracy(imgObj.name, y)

    if not inlocal:
        import json
        d = {'id0': y[0][0], 'id1': y[1][0],
             'id2': y[2][0], 'id3': y[3][0],
             't': str(time() - start) + "sec"}
        print json.dumps(d)

bar.finish()

if (debug):
    a = count / float(len(img_path)) * 100
    print a
    # print "[INFO] final accuracy {}%".format(a * 100)
