import inspect
import logging
from functools import wraps

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def set_logger_save_path(log_path):
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def log_function_call(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            params = inspect.signature(func).parameters
            param_dict = {param: value for param, value in zip(params, args)}
            param_dict.update(kwargs)

            logger.info(f"Calling function: {func.__name__}")
            result = func(*args, **kwargs)
            # logger.info(f"Function {func.__name__} returned: {result}")

            return result
        return wrapper
    return decorator
