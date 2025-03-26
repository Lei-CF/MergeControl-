
import socket
from absl import flags
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from envs.merge_env_v1 import *
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])


