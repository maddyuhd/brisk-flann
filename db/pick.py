import pickle


def saveFile(object, name, debug=False):
    if debug:
        f = open("module/" + name + '.pckl', 'wb')
    else:
        f = open("/home/ubuntu/brisk-flann/module/" + name + '.pckl', 'wb')

    pickle.dump(object, f)
    f.close()

    # if debug:
    #     print "[INFO] saved " + name


def openFile(object, debug=False):
    if debug:
        f = open("module/" + object + '.pckl', 'rb')
    else:
        f = open("/home/ubuntu/brisk-flann/module/" + object + '.pckl', 'rb')
    return pickle.load(f)
    # if debug:
    #     print "[INFO] opening " + object
