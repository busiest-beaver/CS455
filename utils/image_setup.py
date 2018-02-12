import pandas as pd
import numpy as np
import os

def split(inputs):
    outputs = {}
    for _type in inputs.keys():
        sets = inputs[_type]
        outputs[_type] = {}
        outputs[_type]['train'] = {}
        outputs[_type]['validate'] = {}

        for _class in sets.keys():
            _set = sets[_class]
            _len = len(_set)
            validate, train = np.split(_set, [int(.2*_len)])
            outputs[_type]['train'][_class] = list(train['userid'])
            outputs[_type]['validate'][_class] = list(validate['userid'])
    return outputs

def link(src, dst, outputs):
    for _type in outputs.keys():
        for _use in outputs[_type]:
            for _class in outputs[_type][_use]:
                _dir = "%s/%s/%s/%s" % (dst, _type, _use, _class)
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                for userid in outputs[_type][_use][_class]:
                    fname = "%s.jpg" % userid
                    fsrc = os.path.join(src, fname)
                    fdst = os.path.join(_dir, fname)
                    os.symlink(fsrc, fdst)

def main():
    PROFILE_CSV = os.path.expanduser('/data/training/profile/profile.csv')
    SRC_IMG_DIR = os.path.expanduser('/data/training/image')
    DST_IMG_DIR = os.path.expanduser('/data')
    df = pd.read_csv(PROFILE_CSV).loc[:, ['userid', 'gender', 'age']]

    inputs = {
        'gender' : {
            'male' : df[df.gender==0.0].sample(frac=1),
            'female' : df[df.gender==1.0].sample(frac=1)
            },
        'age' : {
            'xx-24' : df[(df.age <= 24)].sample(frac=1),
            '25-34' : df[(25 <= df.age) & (df.age <= 34)].sample(frac=1),
            '35-49' : df[(35 <= df.age) & (df.age <= 49)].sample(frac=1),
            '50-xx' : df[(50 <= df.age)].sample(frac=1)
            },
        'regression' : {
            'personality' : df.sample(frac=1)
            }
        }

    outputs = split(inputs)

    link(SRC_IMG_DIR, DST_IMG_DIR, outputs)

main()
