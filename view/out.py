import json


def jsonDump(state=0, val=0):
    if state:
        d = dict(status=state, id=val)
    else:
        d = dict(status=state)

    print json.dumps(d)
