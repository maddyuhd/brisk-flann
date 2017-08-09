from time import time
import numpy as np
import collections
import os
import tensorflow as tf
from heapq import heappush, heappop
from construct import vecVal, openFile, images, tfInit, progress, inlocal

import sys
import cv2

start = time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
Lmax = 50


def hamming2(s1, s2):
    # assert len(s1) == len(s2)
    r = (1 << np.arange(8))[:, None]
    # return np.count_nonzero((s1 & r) != (s2 & r))
    return np.count_nonzero((np.bitwise_xor(s1, s2) & r) != 0)


def traverseTree(obj, N, PQ, Q):
    global imagesInLeaves, tree, nodes, length, result

    try:
        # result.append(imagesInLeaves[N])
        for i in imagesInLeaves[N]:
            val = vecVal(i)
            cost = hamming2(val, Q)
            heappush(result, (cost, i))
            length += 1

    except KeyError:

        C = tree[N]

        if len(C) > 0:
            with tf.Session(graph=obj.graph) as sess:

                with tf.device("/gpu:0"):
                    data = []
                    for i in C:
                        vect = vecVal(nodes[i])
                        distances = sess.run(obj.euclid_dist, feed_dict={
                            obj.v1: vect, obj.v2: Q})
                        heappush(data, (distances, i))

            Cq = heappop(data)[1]

            for i in data:
                heappush(PQ, i)

            traverseTree(obj, Cq, PQ, Q)

    except Exception as e:
        raise e


def searchTree(obj, T, Q, i):
    global Lmax, length, result

    length = 0
    PQ, result = [], []

    # for ti in T:
    traverseTree(obj, T, PQ, Q)  # ll here

    while (len(PQ) > 0 and length < Lmax):
        # print "inside",length
        try:
            N = heappop(PQ)[1]
        except Exception as e:
            raise e

        traverseTree(obj, N, PQ, Q)
    # pbar.update(i)


def accuracy(name, y):
    global count
    for i in range(len(y)):
        if name == y[i][0]:
            count += 1
            break


if __name__ == "__main__":

    if (inlocal):
        global count
        count = 0

    tree = openFile("tree", inlocal)
    imagesInLeaves = openFile("imagesInLeaves", inlocal)
    nodes = openFile("nodes", inlocal)

    if (inlocal):
        import glob
        img_path = glob.glob("data/2/*.jpg")
        # imgname = str(sys.argv[1])
        # img_path = "data/2/00"+imgname+".jpg"
    else:
        img_path = [str(sys.argv[1])]

    bar = progress("Traversing", len(img_path))
    # star = cv2.xfeatures2d.StarDetector_create(15, 30, 10, 8, 5)
    star = cv2.xfeatures2d.StarDetector_create(10, 30, 15)
    # (border, threshold, ,)

    for imgp in img_path:
        img = images(imgp, 400, star)

        hello = []
        if (inlocal):
            n_clusters = int(sys.argv[1])
        else:
            from construct import n_clusters

        tfObj = tfInit(n_clusters)
        tfObj.finalVariable()

    # if (inlocal):
        # print "[INFO] total items : ", len(img.des)
        # widgets = ["Traversing: ", progressbar.Percentage(),
        #             " ", progressbar.Bar(), " ", progressbar.ETA()]
    #     pbar = progressbar.ProgressBar(maxval=len(img.des), widgets=widgets)
    #     pbar.start()

        for i, d in enumerate(img.des):

            searchTree(tfObj, 0, d, i)

            for _ in range(5):
                help = heappop(result)
                hello.append(help[1][0].name)

        # pbar.finish()

        y = collections.Counter(hello).most_common(5)
        if (inlocal):
            # print "{} : {}".format(img.name,y)
            accuracy(img.name, y)
            bar.update()
        else:
            import json
            d = {'id0': y[0][0], 'id1': y[1][0],
                 'id2': y[2][0], 'id3': y[3][0],
                 't': str(time() - start) + "sec"}
            print json.dumps(d)
    bar.finish()

    if (inlocal):
        a = count / float(len(img_path)) * 100
        print a
        # print "[INFO] final accuracy {}%".format(a * 100)
