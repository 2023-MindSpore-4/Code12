import logging
import os
import sys
import time
from datetime import datetime


class Logger(object):
    def __init__(self, rank, save):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                # string = elapsed_time + string
                pass
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


# class Writer(object):
#     def __init__(self, rank, save):
#         self.rank = rank
#         if self.rank == 0:
#             self.writer = SummaryWriter(log_dir=save, flush_secs=20)
#
#     def add_scalar(self, *args, **kwargs):
#         if self.rank == 0:
#             self.writer.add_scalar(*args, **kwargs)
#
#     def add_figure(self, *args, **kwargs):
#         if self.rank == 0:
#             self.writer.add_figure(*args, **kwargs)
#
#     def add_image(self, *args, **kwargs):
#         if self.rank == 0:
#             self.writer.add_image(*args, **kwargs)
#
#     def add_histogram(self, *args, **kwargs):
#         if self.rank == 0:
#             self.writer.add_histogram(*args, **kwargs)
#
#     def add_histogram_if(self, write, *args, **kwargs):
#         if write and False:  # Used for debugging.
#             self.add_histogram(*args, **kwargs)
#
#     def close(self, *args, **kwargs):
#         if self.rank == 0:
#             self.writer.close()