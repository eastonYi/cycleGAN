import yaml
import sys


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            if type(self[item]) is dict:
                self[item] = AttrDict(self[item])
            res = self[item]
        except:
            print('not found {}'.format(item))
            res = None
        return res

CONFIG_FILE = sys.argv[-1]
args = AttrDict(yaml.load(open(CONFIG_FILE), Loader=yaml.SafeLoader))

args.add_timing = True
