from db.pick import openFile, saveFile
from features.info import inlocal, success, failed
import argparse
from log.log import logInfo

ap = argparse.ArgumentParser()

ap.add_argument("-r", "--remove", required=False,
                help="python script.py -r <IMAGE_ID> ")

ap.add_argument("-b", "--disable", required=False,
                help="python script.py -d <IMAGE_ID>")

ap.add_argument('-d', "--debug", action="store_true",
                help="to debug the program", default=False)

args = vars(ap.parse_args())

debug, remove, disable = args["debug"], args["remove"], args["disable"]

if remove and disable:
    msg = "Use either remove(-r) OR disable(-b)"
    ap.error(msg)

if remove:
    action = "[REMOVE]"
    id = remove
    file_name = "remove"
if disable:
    action = "[DISABLE]"
    id = disable
    file_name = "disable"

log = logInfo(action)

try:
    val = openFile(file_name, inlocal)
    val.add(id)
    saveFile(val, file_name, inlocal)

    log.dump(1, success)

    if debug:
        print "[INFO] {}d : {}".format(file_name, id)

except Exception as e:

    if debug:
        print e

    if not inlocal:
        log.dump(3, failed + str(e))


# if remove:
#     action = "[REMOVE]"
#     log = logInfo(action)

#     try:
#         setVal = openFile("remove", inlocal)
#         setVal.add(remove)
#         saveFile(setVal, "remove", inlocal)

#         log.dump(1, "SUCCESS")

#         if debug:
#             print "[INFO] removed %s" % remove

#     except Exception as e:
#         log.dump(3, e)


# if disable:
#     action = "[DISABLE]"
#     log = logInfo(action)

#     try:
#         setVal = openFile("disable", inlocal)
#         setVal.add(disable)
#         saveFile(setVal, "disable", inlocal)

#         log.dump(1, "SUCCESS")

#         if debug:
#             print "[INFO] disabled %s" % disable

#     except Exception as e:
#         log.dump(3, e)
