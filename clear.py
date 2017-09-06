from db.pick import openFile, saveFile
from features.info import inlocal
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-r", "--remove", required=False,
                help="python script.py -r <IMAGE_ID> ")

ap.add_argument("-d", "--disable", required=False,
                help="python script.py -d <IMAGE_ID>")

ap.add_argument('-b', "--debug", action="store_true",
                help="to debug the program", default=False)

args = vars(ap.parse_args())

debug, remove, disable = args["debug"], args["remove"], args["disable"]

# if delete and disable:
#     ap.error("Either remove OR disable")

if remove:
    setVal = openFile("remove", inlocal)
    setVal.add(remove)
    saveFile(setVal, "remove", inlocal)

    if debug:
        print "[INFO] removed %s" % remove

if disable:
    setVal = openFile("disable", inlocal)
    setVal.add(disable)
    saveFile(setVal, "disable", inlocal)

    if debug:
        print "[INFO] disabled %s" % disable
