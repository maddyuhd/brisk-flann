from construct import vecVal, openFile, images, max_size_lev, n_clusters, progress
import tensorflow as tf
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(filename="log/log_add.log", level=logging.WARNING)
# debug, info, warning, error, critical

try:
    tree = openFile("tree")
    imagesInLeaves = openFile("imagesInLeaves")
    nodes = openFile("nodes")
except Exception as e:
    logging.warning("unable to load files: %s " %str(e))

def add2Db(node, vector, bar):
    global tree, imagesInLeaves, nodes, max_size_lev

    #checking for leafnode 
    try:

        if (len(imagesInLeaves[node]) + len(vector) >= max_size_lev):

            #clean up
            try:
                C = tree[node]
                # print "{} node -> {}".format(node,C) #clean up
                dim = len(vecVal(nodes[C[0]]))

                graph = tf.Graph()
                with tf.Session(graph=graph) as sess:
                    childIDs = [[] for i in range(n_clusters)]
                    centroidsVal = [tf.Variable(vecVal(nodes[i])) for i in C]

                    v1 = tf.placeholder("float", [dim])
                    v2 = tf.placeholder("float", [dim])
                    euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))

                    centroidPh = tf.placeholder("float", [n_clusters])
                    cluster_assignment = tf.argmin(centroidPh, 0)

                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)

                    with tf.device("/gpu:0"):

                        for vector_n in range(len(vector)):

                            vect = vecVal(vector[vector_n])

                            distances = [sess.run(euclid_dist, feed_dict={
                                v1: vect, v2: sess.run(centroid)})
                                for centroid in centroidsVal]

                            assignments_val = sess.run(cluster_assignment, feed_dict={
                                            centroidPh: distances})

                            childIDs[assignments_val].append(vector[vector_n])
            except Exception as e:
                logging.critical("inside keyerror  %s" %str(e))

            #need changes
            for idx, _ in enumerate(childIDs):
                if childIDs[idx]!=[]:
                    # print node,idx #clean up
                    nodeIndex = tree[node][idx]
                    add2Db(nodeIndex, childIDs[idx], bar)

        # except Exception as e:
        #     logging.critical("reconstruct the cluser %s "%str(e))

        else:
            l = len(imagesInLeaves[node])

            for idx, v in enumerate(vectors):
                imagesInLeaves[node].append(v)
                v[0].updateLeaf((node, l + idx))
                bar.update()
            #save the update file

    except KeyError:

        try:

            C = tree[node]
            # print "{} node -> {}".format(node,C) #clean up
            dim = len(vecVal(nodes[C[0]]))

            graph = tf.Graph()
            with tf.Session(graph=graph) as sess:
                childIDs = [[] for i in range(n_clusters)]
                centroidsVal = [tf.Variable(vecVal(nodes[i])) for i in C]

                v1 = tf.placeholder("float", [dim])
                v2 = tf.placeholder("float", [dim])
                euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))

                centroidPh = tf.placeholder("float", [n_clusters])
                cluster_assignment = tf.argmin(centroidPh, 0)

                init_op = tf.global_variables_initializer()
                sess.run(init_op)

                with tf.device("/gpu:0"):

                    for vector_n in range(len(vector)):

                        vect = vecVal(vector[vector_n])

                        distances = [sess.run(euclid_dist, feed_dict={
                            v1: vect, v2: sess.run(centroid)})
                            for centroid in centroidsVal]

                        assignments_val = sess.run(cluster_assignment, feed_dict={
                                        centroidPh: distances})

                        childIDs[assignments_val].append(vector[vector_n])
        except Exception as e:
            logging.critical("inside keyerror  %s" %str(e))

        for idx, _ in enumerate(childIDs):
            if childIDs[idx]!=[]:
                # print node,idx #clean up
                nodeIndex = tree[node][idx]
                add2Db(nodeIndex, childIDs[idx], bar)
    except Exception as e:
        logging.critical("outside %s "%str(e))


img_path = "data/2/0060.jpg"
img = images(img_path)

features = []

for i in range(len(img.des)):
    features.append((img,i))

bar = progress("adding", len(features))
add2Db(0, features, bar)
bar.finish()
