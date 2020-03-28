import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

