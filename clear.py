# from db.pick import openFile
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-r", "--remove", required=False,
                help="python script.py -r <IMAGE_ID> ")

ap.add_argument("-d", "--disable", required=False,
                help="python script.py -d <IMAGE_ID>")

args = vars(ap.parse_args())

delete = args["remove"]
disable = args["disable"]

# if delete and disable:
#     ap.error("Either remove OR disable")

if args["remove"]:
    print "remove %s" % delete

if args["disable"]:
    print "disable %s" % disable
