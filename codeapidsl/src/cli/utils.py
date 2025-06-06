def naive_linechunk(file_path, start_line, end_line):
    """
    Read a chunk of lines from a file between start_line and end_line (inclusive).
    
    Args:
        file_path (str): Path to the file
        start_line (int): Starting line number (1-indexed)
        end_line (int): Ending line number (1-indexed)
        
    Returns:
        str: The content of the specified line range
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    if start_line < 1:
        start_line = 1
    
    if end_line > len(lines):
        end_line = len(lines)
        
    return ''.join(lines[start_line-1:end_line])
