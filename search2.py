from parallel.searchmodel import llProcess
# from view.progress_bar import progress
from features.searcher import analyse, loaddb
from features.info import inlocal
from view.out import jsonDump
import numpy as np
import argparse

ap = argparse.ArgumentParser()

# ap.add_argument('-b', "--batch", action="store_true",
#                 help="Process in batch the program", default=False)

# ap.add_argument('-t', "--time", action="store_true",
#                 help="time taken to process", default=False)

ap.add_argument('-d', "--debug", action="store_true",
                help="to debug the program", default=False)
if inlocal:
    ap.add_argument("-n", "--branch", required=True,
                    help="branch factor")

    ap.add_argument("-i", "--name", required=False,
                    help="image name")

else:
    ap.add_argument("-i", "--data", required=True,
                    help="python script.py <IMAGE_PATH>")

args = vars(ap.parse_args())

# batchMode, timer = args["batch"], args["time"]
debug = args["debug"]

if inlocal:
    n_clusters = args["branch"]
else:
    from features.info import n_clusters

try:
    data = args["data"]
    data = eval("[" + data[1:-2].replace("\;", "],[") + "]]")
    data = np.asarray(data)
    data = data.astype("uint8")

    loaddb()
    n_threads = 3

    resultObj = analyse()

    llProcess(data, n_threads, n_clusters, resultObj)

    result = resultObj.output(data)
    status, id = result

    if not inlocal:
        jsonDump(status, id)


except Exception as e:

    if debug:
        print e

    if not inlocal:
        jsonDump(0)

# class cleanup():
#     def __init__(self, batchMode, inlocal, timer):
#         self.batchMode = batchMode
#         self.inlocal = inlocal
#         self.timer = timer
#         self.newt, self.start, self.count = None, None, None
#         if inlocal and timer:
#             from time import time
#             self.start = time()
#         if batchMode:
#             self.count = 0

#     def timeTaken(self):
#         if self.timer:
#             from time import time
#             self.newt = time()
#             print "[INFO] T2l :{0:.2f} sec" \
#                 .format(time() - self.start)

#     def final(self, total):
#             acc = self.count / float(total) * 100
#             print "[INFO] Final accuracy - {}%".format(acc)

#     def avgTime(self, total):
#         from time import time
#         print "[INFO] Avg time - {0:.2f} sec".format(
#             (time() - self.newt) / float(total))

#     def accuracy(self, name, id, status):

#         if self.batchMode:
#             # for i in range(len(y)):
#             if name == id:  # [0][0]:
#                 self.count += 1
#                 # break
#         elif status:
#             print "[RESULT] {}: {}".format(name, id)
#         else:
#             print "[FAIL] {}: {}".format(name, id)


# clean = cleanup(batchMode, inlocal, timer)

# loaddb()

# clean.timeTaken()

# if batchMode:
#     import glob
#     from features.info import n_clusters
#     img_paths = glob.glob("../data/2/*.jpg")

# elif inlocal:
#     n_clusters = int(args["branch"])
#     img_paths = ["/home/smacar/Desktop/data/full1/0" +
#                  args["name"] + ".jpg"]
# else:
#     from features.info import n_clusters
#     img_paths = [args["path"]]


# if debug:
#     bar = progress("Traversing", len(img_paths))

# n_threads = 3

# for img in img_paths:
#     imgObj = images(img, 400, False)
#     resultObj = analyse()

#     llProcess(imgObj.des, n_threads, n_clusters, resultObj)

#     result = resultObj.output(imgObj.des)
#     status, id = result

#     if debug:
#         bar.update()

#         clean.accuracy(imgObj.name, id, status)

#     if not inlocal:
#         jsonDump(status, id)

# if debug:
#     bar.finish()

# if (debug and batchMode):
#     clean.final(len(img_paths))
#     if timer:
#         clean.avgTime(len(img_paths))
