"""
============================================================================
LOGGING UTILITIES
============================================================================
Purpose: Centralized logging configuration for entire pipeline
Features:
    - Colored console output (different colors per log level)
    - File logging with rotation
    - Structured logging with timestamps
    - Stage-specific loggers
    - Performance tracking decorator

Usage:
    from utils.logging_utils import setup_logger, log_stage
    
    logger = setup_logger("my_module")
    logger.info("Processing started")
    
    @log_stage("Stage 1: My Stage")
    def my_function():
        pass

Author: GATE AE SOTA Pipeline
============================================================================
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from functools import wraps
import time
from typing import Callable

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color codes for console output
    
    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red + Bold
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors"""
        try:
            # Add color to levelname
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )
            
            # Format message - handle Unicode encoding issues
            msg = super().format(record)
            # Replace problematic Unicode characters with ASCII equivalents
            msg = msg.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('⚠', '[WARN]')
            return msg
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # Fallback: remove special characters
            record.levelname = record.levelname.replace('\033[', '').replace('m', '').replace('[', '').replace(']', '')
            msg = super().format(record)
            return msg.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('⚠', '[WARN]')


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
    
    Returns:
        logging.Logger: Configured logger
    
    Example:
        logger = setup_logger("question_loader")
        logger.info("Loading question...")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    console_formatter = ColoredFormatter(
        fmt='%(levelname)-8s | %(name)-20s | %(message)s'
    )
    
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (colored)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        # Use simpler formatter on Windows to avoid encoding issues
        try:
            console_handler.setFormatter(console_formatter)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback to plain formatter if encoding fails
            plain_formatter = logging.Formatter(
                fmt='%(levelname)-8s | %(name)-20s | %(message)s'
            )
            console_handler.setFormatter(plain_formatter)
        logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if log_to_file:
        # Create date-based log file
        log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_stage(stage_name: str) -> Callable:
    """
    Decorator to log stage execution with timing
    
    Args:
        stage_name: Name of the stage (e.g., "Stage 1: Question Loading")
    
    Returns:
        Callable: Decorated function
    
    Example:
        @log_stage("Stage 1: Question Loading")
        def load_question(file_path):
            # ... implementation
            pass
        
        # Output:
        # ===== Stage 1: Question Loading - START =====
        # ... function execution ...
        # ===== Stage 1: Question Loading - COMPLETE (1.23s) =====
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from first arg (self) if exists, else create new
            if args and hasattr(args[0], '__class__'):
                logger = logging.getLogger(args[0].__class__.__name__)
            else:
                logger = logging.getLogger(func.__module__)
            
            # Log start
            logger.info("=" * 60)
            logger.info(f"{stage_name} - START")
            logger.info("=" * 60)
            
            # Execute function with timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                # Log completion
                logger.info("=" * 60)
                logger.info(f"{stage_name} - COMPLETE ({elapsed_time:.2f}s)")
                logger.info("=" * 60)
                
                return result
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                
                # Log failure
                logger.error("=" * 60)
                logger.error(f"{stage_name} - FAILED ({elapsed_time:.2f}s)")
                logger.error(f"Error: {e}")
                logger.error("=" * 60)
                
                raise
        
        return wrapper
    return decorator


class ProgressLogger:
    """
    Log progress for batch operations
    
    Example:
        progress = ProgressLogger("Processing questions", total=1300)
        
        for i, question in enumerate(questions):
            # ... process question ...
            progress.update(i + 1)
    """
    
    def __init__(self, task_name: str, total: int, update_every: int = 10):
        """
        Args:
            task_name: Name of the task
            total: Total items to process
            update_every: Log progress every N items
        """
        self.task_name = task_name
        self.total = total
        self.update_every = update_every
        self.current = 0
        self.start_time = time.time()
        self.logger = setup_logger("progress")
    
    def update(self, current: int):
        """
        Update progress
        
        Args:
            current: Current item number (1-indexed)
        """
        self.current = current
        
        # Log at intervals
        if current % self.update_every == 0 or current == self.total:
            elapsed = time.time() - self.start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (self.total - current) / rate if rate > 0 else 0
            
            progress_pct = (current / self.total) * 100
            
            self.logger.info(
                f"{self.task_name}: {current}/{self.total} ({progress_pct:.1f}%) | "
                f"Rate: {rate:.1f} items/s | "
                f"Remaining: {remaining/60:.1f} min"
            )
    
    def complete(self):
        """Log completion"""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"{self.task_name}: COMPLETE | "
            f"Total time: {elapsed/60:.1f} min | "
            f"Avg rate: {rate:.1f} items/s"
        )


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Log exception with full traceback
    
    Args:
        logger: Logger instance
        exception: Exception object
        context: Additional context string
    """
    import traceback
    
    logger.error("=" * 60)
    logger.error(f"EXCEPTION OCCURRED{': ' + context if context else ''}")
    logger.error("=" * 60)
    logger.error(f"Type: {type(exception).__name__}")
    logger.error(f"Message: {str(exception)}")
    logger.error("\nTraceback:")
    logger.error(traceback.format_exc())
    logger.error("=" * 60)


def setup_pipeline_logging():
    """
    Setup logging for entire pipeline
    
    Creates loggers for all modules with consistent configuration
    Call this once at pipeline startup
    """
    # Main pipeline logger
    pipeline_logger = setup_logger("pipeline", level=logging.INFO)
    
    # Stage loggers
    stage_modules = [
        "initialization",
        "question_loader",
        "question_classifier",
        "cache_manager",
        "retrieval_dense",
        "retrieval_sparse",
        "retrieval_merger",
        "image_consensus",
        "model_orchestrator",
        "voting_engine",
        "debate_orchestrator",
        "synthesis_engine",
        "output_manager",
        "pipeline_runner"
    ]
    
    for module in stage_modules:
        setup_logger(module, level=logging.INFO)
    
    # Utility loggers
    utility_modules = [
        "cost_tracker",
        "checkpoint_manager",
        "health_monitor",
        "api_wrappers",
        "similarity_utils"
    ]
    
    for module in utility_modules:
        setup_logger(module, level=logging.WARNING)  # Less verbose
    
    pipeline_logger.info("Pipeline logging initialized")
    pipeline_logger.info(f"Log directory: {LOGS_DIR}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_pipeline_logging()
    
    # Test logger
    logger = setup_logger("test_module")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test stage decorator
    @log_stage("Test Stage")
    def test_function():
        time.sleep(1)
        logger.info("Processing...")
        return "Done"
    
    result = test_function()
    
    # Test progress logger
    progress = ProgressLogger("Test Task", total=100)
    for i in range(1, 101):
        time.sleep(0.01)
        progress.update(i)
    progress.complete()
