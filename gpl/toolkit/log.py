import logging
from logging import StreamHandler

def set_logger_format():
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    for handler in root_logger.handlers:
        if isinstance(handler, StreamHandler):
            handler.setFormatter(formatter)