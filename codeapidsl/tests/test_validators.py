"""Unit tests for code validators"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.validators import SyntaxValidator, CompilationValidator, UnitTestValidator


class TestSyntaxValidator(unittest.TestCase):
    """Test cases for the SyntaxValidator class"""
    
    def setUp(self):
        self.validator = SyntaxValidator()
    
    def test_validate_python_valid(self):
        """Test validating valid Python code"""
        code = "def test_function():\n    return 42\n"
        result = self.validator.validate(code, language="python")
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
    
    def test_validate_python_invalid(self):
        """Test validating invalid Python code"""
        code = "def test_function()\n    return 42\n"
        result = self.validator.validate(code, language="python")
        
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)
    
    def test_validate_javascript_valid(self):
        """Test validating valid JavaScript code"""
        code = "function test() {\n  return 42;\n}\n"
        result = self.validator.validate(code, language="javascript")
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
    
    def test_validate_javascript_invalid(self):
        """Test validating invalid JavaScript code"""
        code = "function test() {\n  return 42\n"
        result = self.validator.validate(code, language="javascript")
        
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)


class TestCompilationValidator(unittest.TestCase):
    """Test cases for the CompilationValidator class"""
    
    def setUp(self):
        self.validator = CompilationValidator()
    
    def test_compile_python_valid(self):
        """Test compiling valid Python code"""
        code = "def test_function():\n    return 42\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.validator.validate(code, language="python", work_dir=tmpdir)
            self.assertTrue(result["valid"])
            self.assertEqual(len(result["errors"]), 0)
    
    def test_compile_python_invalid(self):
        """Test compiling invalid Python code"""
        code = "import nonexistentmodule\n\ndef test_function():\n    return 42\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.validator.validate(code, language="python", work_dir=tmpdir)
            self.assertFalse(result["valid"])
            self.assertGreater(len(result["errors"]), 0)


class TestUnitTestValidator(unittest.TestCase):
    """Test cases for the UnitTestValidator class"""
    
    def setUp(self):
        self.validator = UnitTestValidator()
    
    def test_validate_with_tests_passing(self):
        """Test validating code with passing tests"""
        code = "def add(a, b):\n    return a + b\n"
        tests = "def test_add():\n    assert add(1, 2) == 3\n    assert add(-1, 1) == 0\n"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "module.py")
            test_path = os.path.join(tmpdir, "test_module.py")
            
            with open(code_path, "w") as f:
                f.write(code)
            with open(test_path, "w") as f:
                f.write("from module import add\n" + tests)
            
            with patch("subprocess.run") as mock_run:
                mock_process = MagicMock()
                mock_process.returncode = 0
                mock_process.stdout = ""
                mock_run.return_value = mock_process
                
                result = self.validator.validate(
                    code, 
                    language="python",
                    test_code=tests,
                    work_dir=tmpdir
                )
                
                self.assertTrue(result["valid"])
                self.assertEqual(len(result["errors"]), 0)
    
    def test_validate_with_tests_failing(self):
        """Test validating code with failing tests"""
        code = "def add(a, b):\n    return a - b  # Bug: subtraction instead of addition\n"
        tests = "def test_add():\n    assert add(1, 2) == 3\n    assert add(-1, 1) == 0\n"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "module.py")
            test_path = os.path.join(tmpdir, "test_module.py")
            
            with open(code_path, "w") as f:
                f.write(code)
            with open(test_path, "w") as f:
                f.write("from module import add\n" + tests)
            
            with patch("subprocess.run") as mock_run:
                mock_process = MagicMock()
                mock_process.returncode = 1
                mock_process.stdout = "AssertionError: assert -1 == 3"
                mock_run.return_value = mock_process
                
                result = self.validator.validate(
                    code, 
                    language="python",
                    test_code=tests,
                    work_dir=tmpdir
                )
                
                self.assertFalse(result["valid"])
                self.assertGreater(len(result["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
