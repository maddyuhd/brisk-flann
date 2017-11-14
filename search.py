import argparse
from core.model import llProcess
from tree.searcher import loaddb
from tree.numpy_help import featureCleanup
from features.info import success, failed, n_clusters
from view.out import jsonDump
from log.log import logInfo


ap = argparse.ArgumentParser()

# ap.add_argument('-b', "--batch", action="store_true", help="search in batch",
#                 default=False)

# ap.add_argument("-u", "--uuid", required=True,
#                 help="<DB_ID>")

ap.add_argument("-f", "--feat", required=False, help="<Feaature_Vector>")

args = vars(ap.parse_args())

try:
    N_THREADS = 3
    LOG = logInfo("[SEARCH]")

    loaddb()

    data = featureCleanup(args["feat"])
    result = llProcess(data, N_THREADS, n_clusters)

    status, imgId = result
    jsonDump(status, imgId)
    LOG.dump(1, success)

except Exception as e:
    jsonDump()
    LOG.dump(3, failed + str(e))
