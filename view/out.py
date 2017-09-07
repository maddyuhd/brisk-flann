import json


def jasonDump(state=0, val=0):
    if state:
        d = dict(status=state, id=val)
    else:
        d = dict(status=state, id=val)  # , id=y[0][0])
    print json.dumps(d)
