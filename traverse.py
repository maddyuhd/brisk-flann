import numpy as np
import collections
import os
import sys
import progressbar
import tensorflow as tf
from heapq import heappush, heappop
from construct import vecVal, openFile, images, tfInit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
Lmax = 50

def hamming2(s1, s2):
    assert len(s1) == len(s2)
    # print s1,s2
    r = (1 << np.arange(8))[:,None]
    # print r
    # cv2.norm(s1, s2[, normType[, mask]])
    # return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return np.count_nonzero((np.bitwise_xor(s1,s2) & r) != 0)


def traverseTree(obj, N, PQ, Q):
    global imagesInLeaves, tree, nodes, length, result

    try:
        # result.append(imagesInLeaves[N])
        for i in imagesInLeaves[N]:
            val = vecVal(i)
            cost = hamming2(val, Q)
            heappush(result, (cost, i))
            length += 1

    except:
        
        C = tree[N]

        if len(C) > 0:
            # dim = len(vecVal(nodes[C[0]]))
            # graph = tf.Graph()
            # with graph.as_default():
            #     sess = tf.Session()
            with tf.Session(graph=obj.graph) as sess:
            #     with tf.device("/cpu:0"):

            #         v1 = tf.placeholder("float", [dim])
            #         v2 = tf.placeholder("float", [dim])
            #         euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))
            #         # centroid = tf.Variable((Q))
     
            #         init_op = tf.global_variables_initializer()
            #         sess.run(init_op)

                with tf.device("/gpu:0"):
                    data = []
                    for i in C:
                        vect = vecVal(nodes[i])
                        distances = sess.run(obj.euclid_dist, feed_dict={
                            obj.v1: vect, obj.v2: Q})
                        heappush(data, (distances, i))

            Cq = heappop(data)[1]

            for i in data:
                heappush(PQ,i)

            traverseTree(obj, Cq, PQ, Q)


def searchTree(obj, T, Q, i):
    global Lmax, length, result

    length = 0 
    PQ, result = [], []

    # for ti in T:
    traverseTree(obj, T, PQ, Q)
    # print length
    # print PQ

    while (len(PQ) > 0 and length < Lmax):
        # print "inside",length
        try:
            N = heappop(PQ)[1]
        except:
            raise

        traverseTree(obj, N, PQ, Q)
    pbar.update(i)


if __name__ == "__main__":
    debug = False

    tree = openFile("tree")
    imagesInLeaves = openFile("imagesInLeaves")
    nodes = openFile("nodes")

    # imgname = str(sys.argv[1])

    # img_path = "data/2/00"+imgname+".jpg"
    img_path = str(sys.argv[1])
    img = images(img_path, 320)

    hello=[]
    tfObj = tfInit(32, 16)
    tfObj.finalVariable()

    if (debug):
        print "[INFO] total items : ", len(img.des)
    widgets = ["Traversing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(img.des), widgets=widgets)
    pbar.start()

    for i,d in enumerate(img.des):

        searchTree(tfObj, 0, d, i)

        for _ in range(5):
            help = heappop(result)
            hello.append(help[1][0].name)
    
    pbar.finish()

    y = collections.Counter(hello).most_common(5) 
    # print y[0][0]
    return y[0][0]
    # print "{} : {}".format(img.name,y)

