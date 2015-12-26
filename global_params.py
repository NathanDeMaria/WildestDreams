import os
import glob

_home_dir = os.path.dirname(os.path.realpath(__file__))
BATCH_SIZE = 128
DEPLOY_PROTOTXT = os.path.join(_home_dir, 'caffe/deploy.prototxt')
IMAGES = glob.glob(os.path.join(_home_dir, 'images', '*', '*'))
LMDB_NAME = '/home/datadatadata.lmdb'
PIXELS = 128
SOLVER_PROTOTXT = os.path.join(_home_dir, 'caffe/solver.prototxt')
TEAMS = ['Illinois', 'Indiana', 'Iowa', 'Maryland', 'Michigan',
         'MichiganState', 'Minnesota', 'Nebraska', 'Northwestern',
         'OhioState', 'PennState', 'Purdue', 'Rutgers', 'Wisconsin']
TRAIN_PROTOTXT = os.path.join(_home_dir, 'caffe/train.prototxt')
