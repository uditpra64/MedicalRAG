import os
import logging
import logging.handlers
from datetime import datetime

def setup_logging(
    log_level=logging.INFO,
    log_dir="logs",
    log_to_console=True,
    log_to_file=True,
    log_format=None
):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: logs)
        log_to_console: Whether to log to console (default: True)
        log_to_file: Whether to log to file (default: True)
        log_format: Custom log format (default: None, uses predefined format)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
        )
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"medical_search_{timestamp}.log")
        
        # Set up file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Log initialization
    logger.info("Logging initialized")
    return logger

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to log messages.
    
    Example:
    ```
    logger = get_logger(__name__)
    logger = LoggerAdapter(logger, {"module": "RAG"})
    logger.info("Message with context")  # Logs: Message with context [module=RAG]
    ```
    """
    
    def process(self, msg, kwargs):
        # Add context to the message
        context_str = " ".join(f"[{k}={v}]" for k, v in self.extra.items())
        if context_str:
            msg = f"{msg} {context_str}"
        return msg, kwargs

def track_performance(func):
    """
    Decorator to track and log function performance.
    
    Example:
    ```
    @track_performance
    def my_function(arg1, arg2):
        # Function body
    ```
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    return wrapper