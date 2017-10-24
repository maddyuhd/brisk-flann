import argparse
from image.pre_process import images
from parallel.model import llProcess
from tree.searcher import loaddb
from features.info import inlocal, success, failed
from view.progress_bar import progress
from view.out import jsonDump
from log.log import logInfo

ap = argparse.ArgumentParser()

ap.add_argument('-b', "--batch", action="store_true", help="search in batch",
                default=False)
ap.add_argument('-d', "--debug", action="store_true", help="debug",
                default=False)

if inlocal:
    ap.add_argument("-n", "--branch", required=True, help="branch factor")
    ap.add_argument("-i", "--name", required=False, help="image name")

else:
    ap.add_argument("-i", "--path", required=False, help="image path")
    ap.add_argument("-f", "--feat", required=False, help="features 2D array")

args = vars(ap.parse_args())

batchMode, debug = args["batch"], args["debug"]


class cleanup():
    def __init__(self, batchMode, inlocal, debug):
        self.batchMode = batchMode
        self.inlocal = inlocal
        self.debug = debug
        self.newt, self.start, self.count = None, None, None
        self.recall, self.precision = None, None

        if debug:
        # if inlocal and debug:
            from time import time
            self.time = time
            self.start = self.time()
        if batchMode:
            self.count, self.recall, self.precision = 0, 0, 0

    def timeTaken(self):
        if self.debug:
            self.newt = self.time()
            print "[INFO] T2l :{0:.2f} sec" \
                .format(self.time() - self.start)

    def final(self, total):
        if self.batchMode:
            self.recall = self.recall / float(total) * 100
            self.precision = self.precision / float(total) * 100
            f1 = (2 * self.precision * self.recall) / float(
                self.precision + self.recall) / 100
            print "[INFO] Recall    : {} %".format(self.recall)
            print "[INFO] Precision : {} %".format(self.precision)
            print "[INFO] F1 score : {}".format(f1)
        # acc = self.count / float(total) * 100
        # print "[INFO] Final accuracy - {}%".format(acc)

        if self.debug and self.batchMode:
            print "[INFO] Avg time - {0:.2f} sec".format(
                (self.time() - self.newt) / float(total))

    def accuracy(self, name, imgId, status):

        if self.batchMode:
            if status:
                self.recall += 1
                if name == imgId:
                    self.precision += 1
            # # for i in range(len(y)):
            # if name == id:  # [0][0]:
            #     self.count += 1
            #     # break
        elif status:
            print "[RESULT] {}: {}".format(name, imgId)
        else:
            print "[FAIL] {}: {}".format(name, imgId)

try:
    clean = cleanup(batchMode, inlocal, debug)

    loaddb()

    clean.timeTaken()

    datas = []
    N_THREADS = 3

    log = logInfo("[SEARCH]")

    if inlocal:
        n_clusters = int(args["branch"])
        if batchMode:
            import glob
            img_paths = glob.glob("/home/smacar/Desktop/data/10s/*.jpg")

            for img in img_paths:
                imgObj = images(img, resize=400, src=False)
                datas.append(imgObj)

        else:
            img_paths = ["/home/smacar/Desktop/data/full1/0" +
                        args["name"] + ".jpg"]

            for img in img_paths:
                imgObj = images(img, resize=400, src=False)
                datas.append(imgObj)

    else:
        from features.info import n_clusters
        img_paths, imgFeat = [args["path"]], args["feat"]

        if img_paths:
            for img in img_paths:
                imgObj = images(img, resize=400, src=False)
                datas.append(imgObj)

        elif imgFeat:
            from tree.numpy_help import featureCleanup
            datas.append(featureCleanup(imgFeat))


    if debug:
        bar = progress("Traversing", len(datas))

    for data in datas:
        if img_paths:
            name = data.name
            data = data.des

        result = llProcess(data, N_THREADS, n_clusters)

        status, imgId = result

        if debug:
            bar.update()
            if img_paths:
                clean.accuracy(name, imgId, status)

        if not inlocal:
            jsonDump(status, imgId)
            log.dump(1, success)

    if debug:
        bar.finish()
        clean.final(len(datas))

except Exception as e:
    if debug:
        print e
    if not inlocal:
        jsonDump()
        log.dump(3, failed + str(e))
