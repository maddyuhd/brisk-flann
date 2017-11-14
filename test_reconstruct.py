import glob
from core.pre_cluster import tfInit
from tree.construct import ConstructMe
from tree.indexer import Indexer
from db.hdf5 import FileStructure
from image.pre_process import images

rootDir = '/home/smacar/Desktop/data/100/*.jpg'
imag_list = glob.glob(rootDir)

n_clusters, max_size_lev = 8, 500
TF = tfInit(n_clusters, max_size_lev)

HDF5 = FileStructure()
vectors = Indexer(imag_list, images, HDF5)

def test_reconstruct(benchmark):
    benchmark.pedantic(ConstructMe, args=(vectors.dump(), TF, False), iterations=1, rounds=1)

    # print "[INFO] indexed {} images".format(len(imag_list))
