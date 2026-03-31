import logging

LOG_BASE_FORMAT = "%(asctime)s  [%(name)s]  %(levelname)8s | %(message)s"


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = LOG_BASE_FORMAT

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name="WiCAT", log_level="debug"):
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    ch = logging.StreamHandler()
    ch.setLevel(log_level_map.get(log_level, logging.DEBUG))
    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    logger.propagate = False

    return logger
