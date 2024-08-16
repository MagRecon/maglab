import logging

# Suppress DEBUG messages from other packages
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('maglab').setLevel(logging.DEBUG)

from os.path import dirname

import sys

from . import const
from . import dataset
from . import preprocess
from . import geo
from . import metrics
from . import convert
from . import vtk
from . import saver
from . import microfields
from . import demag

from .utils import *
from .phasemapper import *
from .micro import *
from .ltem import *
from .tie import *
from .coloring import *
from .sd import *


__all__ = []
__all__.extend(utils.__all__)
__all__.extend(phasemapper.__all__)
__all__.extend(micro.__all__)
__all__.extend(ltem.__all__)
__all__.extend(tie.__all__)


# THIS_DIR = dirname(__file__)
# sys.path.append(THIS_DIR)
