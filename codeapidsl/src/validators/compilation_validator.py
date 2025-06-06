# src/validators/compilation_validator.py
class CompilationValidator:
    """Validates that code compiles and runs basic tests"""
    
    def __init__(self):
        self.syntax_validator = SyntaxValidator()
    
    def validate_compilation(self, code: str, language: str) -> Dict[str, Any]:
        """Validate that code compiles without errors"""
        # First check syntax
        syntax_result = self.syntax_validator.validate_code(code, language)
        if not syntax_result["valid"]:
            return syntax_result
        
        # Then attempt compilation
        return self._compile_code(code, language)
    
    def _compile_code(self, code: str, language: str) -> Dict[str, Any]:
        """Attempt to compile code"""
        if language == 'python':
            return self._compile_python(code)
        elif language in ['javascript', 'typescript']:
            return self._compile_js_ts(code, language)
        elif language == 'java':
            return self._compile_java(code)
        elif language == 'go':
            return self._compile_go(code)
        else:
            return {"valid": False, "errors": [f"Compilation not supported for {language}"]}
    
    def _compile_python(self, code: str) -> Dict[str, Any]:
        """Python doesn't need compilation, but we can import test"""
        try:
            compile(code, '<string>', 'exec')
            return {"valid": True, "errors": [], "warnings": []}
        except Exception as e:
            return {"valid": False, "errors": [str(e)], "warnings": []}
    
    def _compile_js_ts(self, code: str, language: str) -> Dict[str, Any]:
        """Compile JavaScript/TypeScript"""
        extension = '.ts' if language == 'typescript' else '.js'
        command = ['tsc', '--noEmit'] if language == 'typescript' else ['node', '--check']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                command + [temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "valid": result.returncode == 0,
                "errors": result.stderr.split('\n') if result.stderr else [],
                "warnings": []
            }
        finally:
            os.unlink(temp_file)
