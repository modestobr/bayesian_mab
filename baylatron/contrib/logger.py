import logging
import sys

LOG_LEVEL = logging.INFO


def structured_logging_info():
    """Define a clear structure to the logging
    Returns:
        logger: The logger for the module
    """
    # Create a format
    formatter = logging.Formatter(
        "Line %(lineno)d : [ %(asctime)s ] %(filename)s/%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a logger object
    logger = logging.getLogger("root")

    # Set the level of logger as DEBUG
    logger.setLevel(LOG_LEVEL)

    # Create a stream handler that logs info and above to the stdout
    log_console = logging.StreamHandler(sys.stdout)
    log_console.setLevel(LOG_LEVEL)
    log_console.setFormatter(formatter)

    log_console.setFormatter(formatter)

    # Add file and console handlers to logger
    logger.addHandler(log_console)

    return logger


logger = structured_logging_info()
