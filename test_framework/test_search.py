from image.pre_process import images
from core.model import llProcess
from tree.searcher import loaddb

import glob
img_paths = glob.glob("/home/smacar/Desktop/data/100s/*.jpg")
# img_paths = ["/home/smacar/Desktop/data/100s/0001.jpg"]
# print "[PASSED] {}: {}".format(name, imgId)
# print "[FAIL] {}: {}".format(name, imgId)


def test_initial_load(benchmark):
    benchmark.pedantic(loaddb, iterations=1, rounds=2)

datas = []
for img in img_paths:
    img = images(img, resize=400, src=False)
    datas.append(img)

N_THREADS = 3
n_clusters = 8

def process():
    for data in datas:
        name, data = data.name, data.des
        result = llProcess(data, N_THREADS, n_clusters)
        # obj.update(result, name)

class debug():
    def __init__(self):
        self.recall, self.precision = 0, 0
    
    def update(self, result, name):
        status, imgId = result
        if status:
            self.recall += 1
            if name == imgId:
                self.precision += 1
    
    def result(self, total):
        self.recall = self.recall / float(total) * 100
        self.precision = self.precision / float(total) * 100
        f1 = (2 * self.precision * self.recall) / float(
            self.precision + self.recall) / 100
        print "[INFO] Recall    : {} %".format(self.recall)
        print "[INFO] Precision : {} %".format(self.precision)
        print "[INFO] F1 score  : {}".format(f1)


# obj = debug()

def test_search(benchmark):
    benchmark.pedantic(process, iterations=1, rounds=2)


# def test_print_info():
    # obj.result(len(img_paths))
