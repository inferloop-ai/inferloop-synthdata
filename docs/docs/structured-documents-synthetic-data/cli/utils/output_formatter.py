#!/usr/bin/env python3
"""
Output formatter for CLI commands.

Provides consistent formatting for CLI output including colors,
levels, and structured display of information.
"""

import sys
from enum import Enum
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Log levels for output formatting"""
    DEBUG = 'debug'
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    SUCCESS = 'success'


class Colors:
    """ANSI color codes for terminal output"""
    
    # Reset
    RESET = '\033[0m'
    
    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports colors"""
        return (
            hasattr(sys.stdout, 'isatty') and 
            sys.stdout.isatty() and 
            sys.platform != 'win32'
        )


class OutputFormatter:
    """Formatter for CLI output with colors and consistent styling"""
    
    def __init__(self, verbose: bool = False, quiet: bool = False, 
                 use_colors: bool = True):
        self.verbose = verbose
        self.quiet = quiet
        self.use_colors = use_colors and Colors.supports_color()
        
        # Define level colors
        self.level_colors = {
            LogLevel.DEBUG: Colors.GRAY,
            LogLevel.INFO: Colors.BLUE,
            LogLevel.WARNING: Colors.YELLOW,
            LogLevel.ERROR: Colors.RED,
            LogLevel.SUCCESS: Colors.GREEN
        }
        
        # Define level prefixes
        self.level_prefixes = {
            LogLevel.DEBUG: '[DEBUG]',
            LogLevel.INFO: '[INFO]',
            LogLevel.WARNING: '[WARN]',
            LogLevel.ERROR: '[ERROR]',
            LogLevel.SUCCESS: '[SUCCESS]'
        }
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled"""
        if not self.use_colors:
            return text
        return f"{color}{text}{Colors.RESET}"
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format message with level prefix and colors"""
        prefix = self.level_prefixes[level]
        color = self.level_colors[level]
        
        if self.use_colors:
            prefix = self._colorize(prefix, color)
        
        return f"{prefix} {message}"
    
    def debug(self, message: str):
        """Print debug message"""
        if self.verbose and not self.quiet:
            formatted = self._format_message(LogLevel.DEBUG, message)
            print(formatted, file=sys.stderr)
    
    def info(self, message: str):
        """Print info message"""
        if not self.quiet:
            formatted = self._format_message(LogLevel.INFO, message)
            print(formatted)
    
    def warning(self, message: str):
        """Print warning message"""
        formatted = self._format_message(LogLevel.WARNING, message)
        print(formatted, file=sys.stderr)
    
    def error(self, message: str):
        """Print error message"""
        formatted = self._format_message(LogLevel.ERROR, message)
        print(formatted, file=sys.stderr)
    
    def success(self, message: str):
        """Print success message"""
        if not self.quiet:
            formatted = self._format_message(LogLevel.SUCCESS, message)
            print(formatted)
    
    def output(self, message: str):
        """Print plain output message"""
        if not self.quiet:
            print(message)
    
    def print_header(self, title: str, width: int = 60):
        """Print formatted header"""
        if self.quiet:
            return
            
        border = "=" * width
        header = f"{border}\n{title.center(width)}\n{border}"
        
        if self.use_colors:
            header = self._colorize(header, Colors.BOLD + Colors.CYAN)
        
        print(header)
    
    def print_section(self, title: str, width: int = 40):
        """Print formatted section header"""
        if self.quiet:
            return
            
        border = "-" * width
        section = f"\n{title}\n{border}"
        
        if self.use_colors:
            section = self._colorize(section, Colors.BOLD + Colors.BLUE)
        
        print(section)
    
    def print_table(self, headers: List[str], rows: List[List[str]], 
                   min_width: int = 10):
        """Print formatted table"""
        if self.quiet:
            return
        
        if not headers or not rows:
            return
        
        # Calculate column widths
        col_widths = [max(min_width, len(h)) for h in headers]
        
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format header
        header_row = " | ".join(
            header.ljust(col_widths[i]) for i, header in enumerate(headers)
        )
        separator = "-+-".join("-" * width for width in col_widths)
        
        if self.use_colors:
            header_row = self._colorize(header_row, Colors.BOLD + Colors.CYAN)
        
        print(header_row)
        print(separator)
        
        # Format data rows
        for row in rows:
            formatted_row = " | ".join(
                str(cell).ljust(col_widths[i]) if i < len(col_widths) else str(cell)
                for i, cell in enumerate(row)
            )
            print(formatted_row)
    
    def print_list(self, items: List[str], bullet: str = """):
        """Print formatted list"""
        if self.quiet:
            return
        
        for item in items:
            if self.use_colors:
                bullet_colored = self._colorize(bullet, Colors.CYAN)
                print(f"  {bullet_colored} {item}")
            else:
                print(f"  {bullet} {item}")
    
    def print_key_value(self, data: Dict[str, Any], indent: int = 0):
        """Print key-value pairs"""
        if self.quiet:
            return
        
        spaces = " " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                if self.use_colors:
                    key_colored = self._colorize(f"{key}:", Colors.BOLD + Colors.BLUE)
                    print(f"{spaces}{key_colored}")
                else:
                    print(f"{spaces}{key}:")
                self.print_key_value(value, indent + 2)
            elif isinstance(value, list):
                if self.use_colors:
                    key_colored = self._colorize(f"{key}:", Colors.BOLD + Colors.BLUE)
                    print(f"{spaces}{key_colored}")
                else:
                    print(f"{spaces}{key}:")
                self.print_list([str(item) for item in value])
            else:
                if self.use_colors:
                    key_colored = self._colorize(f"{key}:", Colors.BOLD + Colors.BLUE)
                    print(f"{spaces}{key_colored} {value}")
                else:
                    print(f"{spaces}{key}: {value}")
    
    def print_json(self, data: Dict[str, Any], indent: int = 2):
        """Print JSON data with syntax highlighting"""
        if self.quiet:
            return
        
        import json
        
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        
        if self.use_colors:
            # Simple JSON syntax highlighting
            lines = json_str.split('\n')
            for line in lines:
                if ':' in line and not line.strip().startswith('"'):
                    # Key-value lines
                    key_part, value_part = line.split(':', 1)
                    key_colored = self._colorize(key_part, Colors.BLUE)
                    print(f"{key_colored}:{value_part}")
                else:
                    print(line)
        else:
            print(json_str)
    
    def print_progress_bar(self, current: int, total: int, width: int = 50, 
                          prefix: str = "", suffix: str = ""):
        """Print a progress bar"""
        if self.quiet:
            return
        
        percent = (current / total) if total > 0 else 0
        filled_width = int(width * percent)
        
        bar = "ˆ" * filled_width + "‘" * (width - filled_width)
        
        if self.use_colors:
            bar = self._colorize(bar, Colors.GREEN)
        
        progress_text = f"{prefix} [{bar}] {percent:.1%} {suffix}"
        
        # Use carriage return to overwrite previous line
        print(f"\r{progress_text}", end="", flush=True)
        
        # Print newline when complete
        if current >= total:
            print()
    
    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask for user confirmation"""
        if self.quiet:
            return default
        
        suffix = " [Y/n]" if default else " [y/N]"
        prompt = f"{message}{suffix}: "
        
        if self.use_colors:
            prompt = self._colorize(prompt, Colors.YELLOW)
        
        try:
            response = input(prompt).strip().lower()
            if not response:
                return default
            return response in ['y', 'yes', 'true', '1']
        except (KeyboardInterrupt, EOFError):
            print("\n")  # New line after interrupt
            return False


def format_output(message: str, level: LogLevel = LogLevel.INFO, 
                 use_colors: bool = True) -> str:
    """Format a single message with level and colors"""
    formatter = OutputFormatter(use_colors=use_colors)
    return formatter._format_message(level, message)


def create_output_formatter(verbose: bool = False, quiet: bool = False, 
                           use_colors: bool = True) -> OutputFormatter:
    """Factory function to create output formatter"""
    return OutputFormatter(
        verbose=verbose,
        quiet=quiet,
        use_colors=use_colors
    )


__all__ = [
    'OutputFormatter',
    'LogLevel',
    'Colors',
    'format_output',
    'create_output_formatter'
]