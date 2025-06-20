"""
Logging configuration for Structured Documents Synthetic Data Generator
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from .config import get_config


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with additional context"""
        
        # Add standard fields
        if not hasattr(record, 'component'):
            record.component = record.name.split('.')[-1]
        
        if not hasattr(record, 'operation'):
            record.operation = getattr(record, 'funcName', 'unknown')
        
        # Format the message
        formatted = super().format(record)
        
        # Add extra context if available
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'component', 'operation']:
                extra_fields.append(f"{key}={value}")
        
        if extra_fields:
            formatted += f" | {' '.join(extra_fields)}"
        
        return formatted


class LoggerManager:
    """Centralized logger management"""
    
    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}
        self._configured = False
    
    def configure_logging(self, config_file: Optional[str] = None):
        """Configure logging system"""
        if self._configured:
            return
        
        config = get_config(config_file)
        log_config = config.logging
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = StructuredFormatter(log_config.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_config.level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if configured)
        if log_config.file_path:
            file_path = Path(log_config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_config.file_path,
                maxBytes=log_config.max_file_size,
                backupCount=log_config.backup_count
            )
            file_handler.setLevel(getattr(logging, log_config.level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        self._configured = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger with given name"""
        if not self._configured:
            self.configure_logging()
        
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        
        return self._loggers[name]
    
    def log_document_generation(self, 
                              logger: logging.Logger,
                              document_type: str,
                              document_id: str,
                              status: str,
                              generation_time: Optional[float] = None,
                              file_size: Optional[int] = None,
                              **kwargs):
        """Log document generation event"""
        extra_data = {
            'document_type': document_type,
            'document_id': document_id,
            'status': status,
            'operation': 'document_generation'
        }
        
        if generation_time is not None:
            extra_data['generation_time'] = f"{generation_time:.2f}s"
        
        if file_size is not None:
            extra_data['file_size'] = f"{file_size:,} bytes"
        
        extra_data.update(kwargs)
        
        message = f"Document generation {status.lower()}: {document_type} ({document_id})"
        
        if status.lower() == 'completed':
            logger.info(message, extra=extra_data)
        elif status.lower() == 'failed':
            logger.error(message, extra=extra_data)
        else:
            logger.info(message, extra=extra_data)
    
    def log_api_request(self, 
                       logger: logging.Logger,
                       method: str,
                       endpoint: str,
                       status_code: int,
                       response_time: float,
                       user_id: Optional[str] = None,
                       **kwargs):
        """Log API request"""
        extra_data = {
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time': f"{response_time:.3f}s",
            'operation': 'api_request'
        }
        
        if user_id:
            extra_data['user_id'] = user_id
        
        extra_data.update(kwargs)
        
        message = f"{method} {endpoint} - {status_code}"
        
        if 200 <= status_code < 400:
            logger.info(message, extra=extra_data)
        elif 400 <= status_code < 500:
            logger.warning(message, extra=extra_data)
        else:
            logger.error(message, extra=extra_data)
    
    def log_privacy_event(self,
                         logger: logging.Logger,
                         event_type: str,
                         document_id: str,
                         pii_detected: bool = False,
                         pii_types: Optional[list] = None,
                         **kwargs):
        """Log privacy-related events"""
        extra_data = {
            'event_type': event_type,
            'document_id': document_id,
            'pii_detected': pii_detected,
            'operation': 'privacy_processing'
        }
        
        if pii_types:
            extra_data['pii_types'] = ','.join(pii_types)
        
        extra_data.update(kwargs)
        
        message = f"Privacy event: {event_type} for document {document_id}"
        
        if pii_detected:
            logger.warning(f"{message} - PII detected and processed", extra=extra_data)
        else:
            logger.info(f"{message} - No PII detected", extra=extra_data)
    
    def log_validation_result(self,
                            logger: logging.Logger,
                            document_id: str,
                            validation_type: str,
                            result: bool,
                            score: Optional[float] = None,
                            errors: Optional[list] = None,
                            **kwargs):
        """Log validation results"""
        extra_data = {
            'document_id': document_id,
            'validation_type': validation_type,
            'result': 'passed' if result else 'failed',
            'operation': 'validation'
        }
        
        if score is not None:
            extra_data['score'] = f"{score:.3f}"
        
        if errors:
            extra_data['error_count'] = len(errors)
            extra_data['errors'] = '; '.join(errors)
        
        extra_data.update(kwargs)
        
        message = f"Validation {validation_type}: {'PASSED' if result else 'FAILED'} for {document_id}"
        
        if result:
            logger.info(message, extra=extra_data)
        else:
            logger.error(message, extra=extra_data)


# Global logger manager
_logger_manager: Optional[LoggerManager] = None


def get_logger_manager() -> LoggerManager:
    """Get global logger manager instance"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    return _logger_manager


def get_logger(name: str) -> logging.Logger:
    """Get logger for component"""
    return get_logger_manager().get_logger(name)


def configure_logging(config_file: Optional[str] = None):
    """Configure logging system"""
    get_logger_manager().configure_logging(config_file)


# Convenience functions for common logging patterns
def log_document_generation(document_type: str, 
                          document_id: str, 
                          status: str, 
                          **kwargs):
    """Log document generation event"""
    logger = get_logger('document_generation')
    get_logger_manager().log_document_generation(
        logger, document_type, document_id, status, **kwargs
    )


def log_api_request(method: str, 
                   endpoint: str, 
                   status_code: int, 
                   response_time: float, 
                   **kwargs):
    """Log API request"""
    logger = get_logger('api')
    get_logger_manager().log_api_request(
        logger, method, endpoint, status_code, response_time, **kwargs
    )


def log_privacy_event(event_type: str, 
                     document_id: str, 
                     **kwargs):
    """Log privacy event"""
    logger = get_logger('privacy')
    get_logger_manager().log_privacy_event(
        logger, event_type, document_id, **kwargs
    )


def log_validation_result(document_id: str, 
                        validation_type: str, 
                        result: bool, 
                        **kwargs):
    """Log validation result"""
    logger = get_logger('validation')
    get_logger_manager().log_validation_result(
        logger, document_id, validation_type, result, **kwargs
    )