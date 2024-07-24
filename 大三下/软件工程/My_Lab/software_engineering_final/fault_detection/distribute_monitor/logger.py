'''
日志模块
'''

import logging
import os

class Logger:
    def __init__(self, log_file = 'distribute_monitor_system.log',format = '%(asctime)s - %(ipaddr)s - %(levelname)s - %(message)s', dateformat='%Y/%m/%d %H:%M:%S', level = logging.DEBUG , mode = 'w') -> None:
        self.logger = logging.getLogger('distribute monitor')
        formatter = logging.Formatter(fmt=format, datefmt= dateformat)
        self.logger.setLevel(level=level)
        self.handler = None
        
        if log_file is None or log_file == 'stdout':
            self.log_filepath = 'stdout'
            self.handler = logging.StreamHandler()
        else:
            self.log_filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), log_file)
            self.handler = logging.FileHandler(filename=self.log_filepath, mode=mode)
            
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        
    def info(self, ip, msg):
        self.logger.info(msg, extra={'ipaddr': ip})

    def debug(self, ip, msg):
        self.logger.debug(msg, extra={'ipaddr': ip})

    def warning(self, ip, msg):
        self.logger.warning(msg, extra={'ipaddr': ip})

    def error(self, ip, msg):
        self.logger.error(msg, extra={'ipaddr': ip})
        
    def __del__(self):
        self.handler.flush()
        self.handler.close()

if __name__ == '__main__':
    log = Logger()
    log.info('127.0.0.1','test')
    log.debug('127.0.0.1','test')
    log.warning('127.0.0.1','test')
    log.error('127.0.0.1','test')

LOGGER = Logger()