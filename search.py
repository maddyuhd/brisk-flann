from image.pre_process import images
from parallel.model import myThread, loadTf
from progress_bar.progress import progress
from features.searcher import analyse, loaddb
from features.info import inlocal
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-b', "--batch", action="store_true",
                help="Process in batch the program", default=False)

ap.add_argument('-t', "--time", action="store_true",
                help="time taken to process", default=False)

ap.add_argument('-d', "--debug", action="store_true",
                help="to debug the program", default=False)
if inlocal:
    ap.add_argument("-n", "--branch", required=True,
                    help="branch factor")

    ap.add_argument("-i", "--name", required=False,
                    help="image name")

else:
    ap.add_argument("-i", "--path", required=True,
                    help="python script.py <IMAGE_PATH>")

args = vars(ap.parse_args())

batchMode, timer, debug = args["batch"], args["time"], args["debug"]


class cleanup():
    def __init__(self, batchMode, inlocal, timer):
        self.batchMode = batchMode
        self.inlocal = inlocal
        self.timer = timer
        self.newt, self.start, self.count = None, None, None
        if inlocal and timer:
            from time import time
            self.start = time()
        if batchMode:
            self.count = 0

    def timeTaken(self):
        if self.timer:
            from time import time
            self.newt = time()
            print "[INFO] T2l :{0:.2f} sec" \
                .format(time() - self.start)

    def final(self, total):
            acc = self.count / float(total) * 100
            print "[INFO] Final accuracy - {}%".format(acc)

    def avgTime(self, total):
        from time import time
        print "[INFO] Avg time - {0:.2f} sec".format(
            (time() - self.newt) / float(total))

    def accuracy(self, name, y):
        if self.batchMode:
            for i in range(len(y)):
                if name == y[i][0]:
                    self.count += 1
                    break
        else:
            print "[RESULT] {}: {}".format(name, y)


clean = cleanup(batchMode, inlocal, timer)

loaddb()

clean.timeTaken()

if batchMode:
    import glob
    from features.info import n_clusters
    img_paths = glob.glob("../data/2/*.jpg")

elif inlocal:
    n_clusters = int(args["branch"])
    img_paths = ["/home/smacar/Desktop/data/full1/0" +
                 args["name"] + ".jpg"]
else:
    from features.info import n_clusters
    img_paths = [args["path"]]

if debug:
    bar = progress("Traversing", len(img_paths))


def chunkData(l):
    return [l[0:(len(l) / 2)], l[(len(l) / 2):]]
    # for i in xrange(0, len(l), n):
    #     yield l[i:i + n]


loadTf(n_clusters)

for img in img_paths:
    imgObj = images(img, 400, False)
    resultObj = analyse()

    chunks = chunkData(imgObj.des)

    # for idx, i in enumerate(len(chunks)):

    thread1 = myThread(1, "Thread-1", chunks[0], resultObj)
    thread2 = myThread(2, "Thread-2", chunks[1], resultObj)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # for d in imgObj.des:
    #     tree = searchMe(tfObj, 0, d)
    #     resultObj.update(tree.result)

    y = resultObj.output()

    if debug:
        bar.update()
        clean.accuracy(imgObj.name, y)

    if not inlocal:
        import json
        d = dict(status=1, id=y[0][0])
        print json.dumps(d)

if debug:
    bar.finish()

if (debug and batchMode):
    clean.final(len(img_paths))
    if timer:
        clean.avgTime(len(img_paths))
