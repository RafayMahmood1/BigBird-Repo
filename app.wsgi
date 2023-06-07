import sys
import os
import logging

logging.basicConfig(stream=sys.stderr)
print(os.path.dirname(__file__))

sys.path.insert(0, os.path.dirname(__file__))

from route import app as application
