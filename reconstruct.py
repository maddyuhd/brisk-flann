import glob
import argparse
from core.pre_cluster import tfInit
from tree.construct import ConstructMe
from tree.indexer import Indexer
from db.hdf5 import FileStructure
from image.pre_process import images
from features.info import success, failed
from features.info import n_clusters, max_size_lev
from log.log import logInfo


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--build", required=True, help="<src_folder>")

# ap.add_argument("-u", "--uuid", required=True,
#                 help="<DB_ID>")

args = vars(ap.parse_args())

try:
    LOG = logInfo("[RECONSTRUCT]")

    rootDir = str(args["build"]) + "*.jpg"
    imag_list = glob.glob(rootDir)

    HDF5 = FileStructure()
    vectors = Indexer(imag_list, images, HDF5)

    TF = tfInit(n_clusters, max_size_lev)
    ConstructMe(vectors.dump(), TF, debug=False)

    LOG.dump(1, success)

except Exception as e:
    LOG.dump(3, failed + str(e))
