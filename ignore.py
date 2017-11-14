import argparse
from features.info import inlocal, success, failed
from log.log import logInfo
from db.pick import openFile, saveFile
from db.in_memory import Manage_del_dis

ap = argparse.ArgumentParser()

ap.add_argument("-r", "--remove", required=False,
                help="python script.py -r <IMAGE_ID> ")

ap.add_argument("-b", "--disable", required=False,
                help="python script.py -d <IMAGE_ID>")

# ap.add_argument("-u", "--uuid", required=True,
#                 help="<DB_ID>")

args = vars(ap.parse_args())

try:

    remove, disable = args["remove"], args["disable"]

    if remove:
        action = "[REMOVE]"
    elif disable:
        action = "[DISABLE]"

    log = logInfo(action)
    obj = Manage_del_dis()

    if remove:
        responds = obj.disable(remove)
    elif disable:
        responds = obj.remove(disable)

    if responds:
        log.dump(1, success)
    else:
        log.dump(1, success + "UNDO")

except Exception as e:
    log.dump(3, failed + str(e))
