import os
import logging

class SimpleLogger():
    def __init__(self, log_path, log_name):
        self.logger = logging.getLogger(log_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        hdlr = logging.FileHandler(log_path + log_name + '.log')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        hdlr.setFormatter(formatter)

        self.logger.handlers = []
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log_info(self, logs):
        log = ''
        for key, value in logs.items():
            log += key + ': ' + str(value) + '\n'
        self.logger.info(log + '\n')
