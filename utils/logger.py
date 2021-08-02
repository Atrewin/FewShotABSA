# -*- coding: utf-8 -*-

import logging
import sys
import io

logger = logging.getLogger("Summarization logger")

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)




class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

tqdm_out = TqdmToLogger(logger,level=logging.INFO)