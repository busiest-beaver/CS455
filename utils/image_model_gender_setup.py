import pandas as pd
import numpy as np
import os

PROFILE_CSV = '/data/training/profile/profile.csv'
SRC_IMG_DIR = '/data/training/image'
MALE_DIR = '/data/training/image/male'
MALE_TRAIN_DIR = '/data/training/image/male/train'
MALE_TEST_DIR = '/data/training/image/male/test'
MALE_VALIDATE_DIR = '/data/training/image/male/validate'

FEMALE_DIR = '/data/training/image/female'
FEMALE_TRAIN_DIR = '/data/training/image/female/train'
FEMALE_TEST_DIR = '/data/training/image/female/test'
FEMALE_VALIDATE_DIR = '/data/training/image/female/validate'

df = pd.read_csv(PROFILE_CSV).loc[:, ['userid', 'gender']]
f_set = df[df.gender==1.0].sample(frac=1)
m_set = df[df.gender==0.0].sample(frac=1)

f_test, f_validate, f_train = np.split(f_set, [int(.2*len(f_set)), int(.4*len(f_set))])
m_test, m_validate, m_train = np.split(m_set, [int(.2*len(m_set)), int(.4*len(m_set))])

if not os.path.exists(FEMALE_DIR):
    os.makedirs(FEMALE_DIR)
if not os.path.exists(FEMALE_TRAIN_DIR):
    os.makedirs(FEMALE_TRAIN_DIR)
if not os.path.exists(FEMALE_VALIDATE_DIR):
    os.makedirs(FEMALE_VALIDATE_DIR)
if not os.path.exists(FEMALE_TEST_DIR):
    os.makedirs(FEMALE_TEST_DIR)

if not os.path.exists(MALE_DIR):
    os.makedirs(MALE_DIR)
if not os.path.exists(MALE_TRAIN_DIR):
    os.makedirs(MALE_TRAIN_DIR)
if not os.path.exists(MALE_VALIDATE_DIR):
    os.makedirs(MALE_VALIDATE_DIR)
if not os.path.exists(MALE_TEST_DIR):
    os.makedirs(MALE_TEST_DIR)

for index, row in f_train.iterrows():
    fname = "%s.jpg" % row['userid']
    src = os.path.join(SRC_IMG_DIR, fname)
    dst = os.path.join(FEMALE_TRAIN_DIR, fname)
    os.symlink(src, dst)

for index, row in f_validate.iterrows():
    fname = "%s.jpg" % row['userid']
    src = os.path.join(SRC_IMG_DIR, fname)
    dst = os.path.join(FEMALE_VALIDATE_DIR, fname)
    os.symlink(src, dst)

for index, row in f_test.iterrows():
    fname = "%s.jpg" % row['userid']
    src = os.path.join(SRC_IMG_DIR, fname)
    dst = os.path.join(FEMALE_TEST_DIR, fname)
    os.symlink(src, dst)

for index, row in m_train.iterrows():
    fname = "%s.jpg" % row['userid']
    src = os.path.join(SRC_IMG_DIR, fname)
    dst = os.path.join(MALE_TRAIN_DIR, fname)
    os.symlink(src, dst)

for index, row in m_validate.iterrows():
    fname = "%s.jpg" % row['userid']
    src = os.path.join(SRC_IMG_DIR, fname)
    dst = os.path.join(MALE_VALIDATE_DIR, fname)
    os.symlink(src, dst)

for index, row in m_test.iterrows():
    fname = "%s.jpg" % row['userid']
    src = os.path.join(SRC_IMG_DIR, fname)
    dst = os.path.join(MALE_TEST_DIR, fname)
    os.symlink(src, dst)

