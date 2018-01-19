import os
from tcss455group7 import user as User
import pickle

class baseline:

    __PROFILE_FILE_NAME = 'profile/profile.csv'
    __TRAINED_MODEL_PATH = '/data/.baseline'

    def __init__(self):
        '''Constructor for this class.'''

    def train(self, **kwargs):
        input_dir = kwargs['input_dir']
        users = self.parse_profile_csv(input_dir)
        n_users = len(users)
        model = {
                  'age': [0,0,0,0],
                  'gender': 0.0,
                  'ope': 0.0,
                  'con': 0.0,
                  'ext': 0.0,
                  'agr': 0.0,
                  'neu': 0.0
                }
        prediction = User.user(userid=-1)
        for user in users:
            attrs = user.attrs()
            for k, v in model.items():
                if k == 'age':
                    age = float(attrs['age'])
                    if age < 25.0:
                        model[k][0] += 1
                    elif age < 35.0:
                        model[k][1] += 1
                    elif age < 50.0:
                        model[k][2] += 1
                    else:
                        model[k][3] += 1
                else:
                    model[k] = v + float(attrs[k])
        for k, v in model.items():
            if not k == 'age':
                model[k] = v / n_users

        age_groups = ['xx-24', '25-34', '35-49', '50-xx']
        age_group = age_groups[model['age'].index(max(model['age']))]
        prediction.set(age=age_group)
        prediction.set(gender="%.1f" % round(model['gender'], 0))
        prediction.set(ope="%.3f" % round(model['ope'], 3))
        prediction.set(con="%.3f" % round(model['con'], 3))
        prediction.set(ext="%.3f" % round(model['ext'], 3))
        prediction.set(agr="%.3f" % round(model['agr'], 3))
        prediction.set(neu="%.3f" % round(model['neu'], 3))

        with os.fdopen(os.open(self.__TRAINED_MODEL_PATH, os.O_WRONLY | os.O_CREAT, 0o666), 'wb') as model_file:
            pickle.dump(prediction, model_file)

    def parse_profile_csv(self, input_dir):
        file_name = os.path.join(input_dir, self.__PROFILE_FILE_NAME)
        users = []
        with open(file_name) as file_handle:
            lines = iter(file_handle)
            header = next(lines).strip().split(',')[1:]
            for line in lines:
                row = line.strip().split(',')[1:]
                attributes = {}
                for i in range(0, len(header)):
                    k = header[i]
                    v = row[i]
                    attributes[k] = v
                user = User.user(**attributes)
                users.append(user)
        return users


    def test(self, **kwargs):
        prediction = None
        input_dir = kwargs['input_dir']
        output_dir = kwargs['output_dir']

        file_name = os.path.expanduser(self.__TRAINED_MODEL_PATH)
        if not os.path.exists(file_name):
            print('The model must be trained before it can be used for testing.')
            exit()

        with open(file_name, 'rb') as model_file:
            prediction = pickle.load(model_file)

        users = self.parse_profile_csv(input_dir)
        for user in users:
            id = user.id()
            prediction.id(id=id)
            xml_str = prediction.to_xml()
            with open(os.path.join(output_dir, "%s.xml" % id), 'wb+') as xml_file_handle:
                    xml_file_handle.write(xml_str)

