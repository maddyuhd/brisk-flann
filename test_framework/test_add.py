from features.addimg import add2Db
from core.pre_cluster import tfInit
from tree.indexer import Indexer
from db.hdf5 import FileStructure
from image.pre_process import images


HDF5 = FileStructure()

img_path = ["/home/smacar/Desktop/data/full/0001.jpg"]
vectors = Indexer(img_path, images, HDF5)

n_clusters, max_size_lev = 8, 500
TF = tfInit(n_clusters, max_size_lev)

def test_add_img(benchmark):
    benchmark.pedantic(add2Db, args=(0, vectors.dump(), TF, False), iterations=1, rounds=1)
