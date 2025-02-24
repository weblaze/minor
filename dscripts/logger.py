import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Purple
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record):
        # Save original format
        format_orig = self._style._fmt
        
        # Add colors if not writing to file
        if not any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
            self._style._fmt = f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}" + format_orig + f"{self.COLORS['RESET']}"
        
        # Call the original formatter
        result = super().format(record)
        
        # Restore original format
        self._style._fmt = format_orig
        return result

def setup_logger(name: str, log_dir: Optional[Path] = None, log_to_file: bool = True) -> logging.Logger:
    """
    Set up a logger with both console and file output
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        log_to_file: Whether to save logs to file
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

class DataLogger:
    """Logger for tracking data processing metrics and statistics"""
    
    def __init__(self, name: str, log_dir: Path):
        self.name = name
        self.log_dir = log_dir
        self.metrics: Dict[str, Any] = {}
        self.logger = setup_logger(name, log_dir)
        
    def log_metric(self, metric_name: str, value: Any):
        """Log a metric value"""
        self.metrics[metric_name] = value
        self.logger.info(f"{metric_name}: {value}")
    
    def log_stats(self, data_array, name: str):
        """Log basic statistics for a numpy array"""
        stats = {
            'mean': float(data_array.mean()),
            'std': float(data_array.std()),
            'min': float(data_array.min()),
            'max': float(data_array.max()),
            'shape': data_array.shape
        }
        self.metrics[f"{name}_stats"] = stats
        self.logger.info(f"{name} stats: {json.dumps(stats, indent=2)}")
    
    def save_metrics(self):
        """Save all metrics to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = self.log_dir / f"{self.name}_metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Saved metrics to {metrics_file}")

def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Get or create a logger with the given name"""
    if log_dir is None:
        log_dir = Path("logs")
    return setup_logger(name, log_dir) 