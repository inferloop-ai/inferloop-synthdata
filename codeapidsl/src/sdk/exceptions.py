# src/sdk/exceptions.py
class SynthCodeException(Exception):
    """Base exception for SDK"""
    pass

class ValidationException(SynthCodeException):
    """Validation specific exception"""
    pass
