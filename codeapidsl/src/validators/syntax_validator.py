# src/validators/syntax_validator.py
import ast
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Tuple

class SyntaxValidator:
    """Validates code syntax across multiple languages"""
    
    SUPPORTED_LANGUAGES = {
        'python': {
            'extension': '.py',
            'validator': '_validate_python_syntax'
        },
        'javascript': {
            'extension': '.js',
            'validator': '_validate_javascript_syntax'
        },
        'typescript': {
            'extension': '.ts',
            'validator': '_validate_typescript_syntax'
        },
        'java': {
            'extension': '.java',
            'validator': '_validate_java_syntax'
        },
        'go': {
            'extension': '.go',
            'validator': '_validate_go_syntax'
        }
    }
    
    def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax for given language"""
        if language not in self.SUPPORTED_LANGUAGES:
            return {
                "valid": False,
                "errors": [f"Unsupported language: {language}"],
                "warnings": []
            }
        
        validator_method = getattr(self, self.SUPPORTED_LANGUAGES[language]['validator'])
        return validator_method(code)
    
    def _validate_python_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return {
                "valid": True,
                "errors": [],
                "warnings": self._check_python_style(code)
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Syntax error at line {e.lineno}: {e.msg}"],
                "warnings": []
            }
    
    def _validate_javascript_syntax(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript syntax using Node.js"""
        return self._validate_with_external_tool(code, '.js', ['node', '--check'])
    
    def _validate_typescript_syntax(self, code: str) -> Dict[str, Any]:
        """Validate TypeScript syntax"""
        return self._validate_with_external_tool(code, '.ts', ['tsc', '--noEmit'])
    
    def _validate_java_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Java syntax"""
        return self._validate_with_external_tool(code, '.java', ['javac', '-Xlint'])
    
    def _validate_go_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Go syntax"""
        return self._validate_with_external_tool(code, '.go', ['go', 'build'])
    
    def _validate_with_external_tool(self, code: str, extension: str, 
                                   command: List[str]) -> Dict[str, Any]:
        """Validate using external compilation tools"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Modify command to include temp file
            full_command = command + [temp_file]
            result = subprocess.run(
                full_command, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            return {
                "valid": result.returncode == 0,
                "errors": result.stderr.split('\n') if result.stderr else [],
                "warnings": []
            }
        except subprocess.TimeoutExpired:
            return {
                "valid": False,
                "errors": ["Validation timeout"],
                "warnings": []
            }
        except FileNotFoundError:
            return {
                "valid": False,
                "errors": [f"Validator not found: {command[0]}"],
                "warnings": []
            }
        finally:
            os.unlink(temp_file)
    
    def _check_python_style(self, code: str) -> List[str]:
        """Check Python style guidelines"""
        warnings = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 88:
                warnings.append(f"Line {i}: Line too long ({len(line)} > 88 characters)")
            if line.endswith(' '):
                warnings.append(f"Line {i}: Trailing whitespace")
        
        return warnings
