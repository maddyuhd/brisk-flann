import numpy as np
import collections
import os
import sys
import tensorflow as tf
from heapq import heappush, heappop
from construct import vecVal, timeCal, openFile, images

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


def traverseTree(N, PQ, Q):
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
            dim = len(vecVal(nodes[C[0]]))
            graph = tf.Graph()
            # with graph.as_default():
            #     sess = tf.Session()
            with tf.Session(graph=graph) as sess:
                with tf.device("/cpu:0"):

                    v1 = tf.placeholder("float", [dim])
                    v2 = tf.placeholder("float", [dim])
                    euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))
                    centroid = tf.Variable((Q))
     
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)

                with tf.device("/gpu:0"):
                    data = []
                    for i in C:
                        vect = vecVal(nodes[i])
                        distances = sess.run(euclid_dist, feed_dict={
                            v1: vect, v2: Q})
                        heappush(data, (distances, i))

                Cq = heappop(data)[1]

                for i in data:
                    heappush(PQ,i)

                traverseTree(Cq, PQ, Q)


def searchTree(T, Q):
    global Lmax, length, result

    length = 0 
    PQ, result = [], []

    # for ti in T:
    traverseTree(T, PQ, Q)
    # print length
    # print PQ

    while (len(PQ) > 0 and length < Lmax):
        # print "inside",length
        try:
            N = heappop(PQ)[1]
        except:
            raise

        traverseTree(N, PQ, Q)

    # K = 4
    # result = []

    # for i in range(K):
        # result.append(heappop(result)[1])
    # if len(result)>1:
    #     return result
    # else:
    #     return 0


if __name__ == "__main__":

    tree = openFile("tree")
    imagesInLeaves = openFile("imagesInLeaves")
    nodes = openFile("nodes")

    imgname = str(sys.argv[1])

    img_path = "/home/smacar/Desktop/dev/online/tree/binary_brief/data/2/00"+imgname+".jpg"
    img = images(img_path)

    start = timeCal()
    hello=[]

    print "Total items : ", len(img.des)

    for d in img.des:
        # print "Q",d
        searchTree(0,d)
        # print len(res)
        for _ in range(5):
            help = heappop(result)
            hello.append(help[1][0].name)
    
    start.result("taken")
    y = collections.Counter(hello).most_common(5) 
    print "{} : {}".format(img.name,y)

