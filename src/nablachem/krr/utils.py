import structlog
import logging
import sys
import math


# Set root logger to WARNING to suppress other libs, but allow this module's logger to be DEBUG
logging.basicConfig(level=logging.WARNING, format="%(message)s")
# Create a module-specific logger that can show DEBUG messages
logging.getLogger(__name__).setLevel(logging.DEBUG)


def custom_level_processor(logger, method_name, event_dict):
    """Add custom level for result messages"""
    if event_dict.get("result_level"):
        event_dict["level"] = "result"
        event_dict.pop("result_level", None)
    return event_dict


# Configure structlog with colors and timestamps
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        custom_level_processor,
        structlog.dev.ConsoleRenderer(
            colors={
                "result": "\033[34m",  # blue
            }
        ),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Create logger instance
logger = structlog.get_logger()


# Wrapper functions
def info(message, **kwargs):
    """Log info level message in grey"""
    logger.info(message, **kwargs)


def debug(message, **kwargs):
    """Log debug level message in yellow"""
    logger.debug(message, **kwargs)


def warning(message, **kwargs):
    """Log warning level message in orange"""
    logger.warning(message, **kwargs)


def error(message, **kwargs):
    """Log error level message in red"""
    logger.error(message, **kwargs)
    sys.exit(1)


def result(message, **kwargs):
    """Log result level message in blue"""
    # Create a new logger with result level color
    result_logger = structlog.get_logger().bind(result_level=True)
    result_logger.info(message, **kwargs)


def get_training_sizes(mincount: int, maxcount: int) -> list[int]:
    training_sizes = []

    if mincount > 0:
        training_sizes.append(mincount)

    if mincount > 0 and maxcount > mincount:
        min_power = int(math.log2(mincount))
        max_power = int(math.log2(maxcount))

        start_power = min_power + 1 if 2**min_power == mincount else min_power

        for i in range(start_power, max_power + 1):
            power_of_2 = 2**i
            if mincount < power_of_2 <= maxcount:
                training_sizes.append(power_of_2)

    if maxcount != mincount and maxcount not in training_sizes:
        training_sizes.append(maxcount)

    return sorted(set(training_sizes))
