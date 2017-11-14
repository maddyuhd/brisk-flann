import argparse
from features.addimg import add2Db
from features.info import n_clusters, max_size_lev
from features.info import success, failed
from core.pre_cluster import tfInit
from tree.indexer import Indexer
from db.hdf5 import FileStructure
from image.pre_process import images
from view.out import jsonDump
from log.log import logInfo

LOG = logInfo("[ADD]")

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--add", required=True,
                help="<IMAGE_PATH>")

# ap.add_argument("-u", "--uuid", required=True,
#                 help="<DB_ID>")

args = vars(ap.parse_args())

try:
    img_path = [args["add"]]

    HDF5 = FileStructure()
    vectors = Indexer(img_path, images, HDF5)

    TF = tfInit(n_clusters, max_size_lev)

    add2Db(0, vectors.dump(), TF, debug=False)

    LOG.dump(1, success)
    jsonDump(1)

except Exception as e:
    jsonDump()
    LOG.dump(3, failed + str(e))
