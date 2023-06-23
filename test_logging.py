import logging
import pdb

logger_loudness = logging.getLogger('logger_loudness')
logger_loudness.setLevel(logging.INFO)
file_handler1 = logging.FileHandler('./logs/logfiletest.log')
logger_loudness.addHandler(file_handler1)
logger_loudness.info("This is a test message")

pdb.set_trace()
logger_loudness.info("This is after opening the file")
