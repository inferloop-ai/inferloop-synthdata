# src/delivery/formatters.py
import json
import csv
import io
from typing import List, Dict, Any

class JSONLFormatter:
    """Format output as JSON Lines"""
    
    def format(self, generated_code: List[Dict[str, Any]], 
               validation_results: List[Dict[str, Any]] = None) -> str:
        """Format data as JSONL"""
        lines = []
        
        for i, code_sample in enumerate(generated_code):
            # Merge with validation results if available
            if validation_results and i < len(validation_results):
                code_sample["validation"] = validation_results[i]
            
            lines.append(json.dumps(code_sample, ensure_ascii=False))
        
        return '\n'.join(lines)

class CSVFormatter:
    """Format output as CSV"""
    
    def format(self, generated_code: List[Dict[str, Any]], 
               validation_results: List[Dict[str, Any]] = None) -> str:
        """Format data as CSV"""
        if not generated_code:
            return ""
        
        output = io.StringIO()
        fieldnames = ['id', 'prompt', 'code', 'language', 'lines_of_code', 
                     'complexity', 'dependencies', 'syntax_valid', 'compilation_valid']
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, code_sample in enumerate(generated_code):
            row = {
                'id': code_sample.get('id', ''),
                'prompt': code_sample.get('prompt', ''),
                'code': code_sample.get('code', '').replace('\n', '\\n'),
                'language': code_sample.get('language', ''),
                'lines_of_code': code_sample.get('metadata', {}).get('lines_of_code', 0),
                'complexity': code_sample.get('metadata', {}).get('estimated_complexity', ''),
                'dependencies': str(code_sample.get('metadata', {}).get('dependencies', [])),
                'syntax_valid': '',
                'compilation_valid': ''
            }
            
            # Add validation results if available
            if validation_results and i < len(validation_results):
                validation = validation_results[i]
                if 'syntax_validation' in validation:
                    row['syntax_valid'] = validation['syntax_validation'].get('valid', False)
                if 'compilation_validation' in validation:
                    row['compilation_valid'] = validation['compilation_validation'].get('valid', False)
            
            writer.writerow(row)
        
        return output.getvalue()
