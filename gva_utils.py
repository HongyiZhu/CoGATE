from types          import SimpleNamespace
import json

def load_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def dict2dotdict(dic):
    if dic != None:
        return SimpleNamespace(**dic)
    else:
        return SimpleNamespace()


def dotdict2dict(dotdict):
    return vars(dotdict)