#!/usr/bin/env python3
"""
Test utilities for creating mock data and test environments.

Provides functions for generating test data, creating mock documents,
setting up test environments, and test assertions.
"""

import json
import random
import string
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from faker import Faker
from PIL import Image, ImageDraw

from ..core import get_logger
from .file_utils import write_file, create_directory

logger = get_logger(__name__)
fake = Faker()


def create_mock_document(doc_type: str = 'general', 
                        include_annotations: bool = True,
                        include_metadata: bool = True) -> Dict[str, Any]:
    """
    Create mock document for testing.
    
    Args:
        doc_type: Type of document ('general', 'finance', 'legal', 'medical')
        include_annotations: Whether to include annotations
        include_metadata: Whether to include metadata
    
    Returns:
        Mock document dictionary
    """
    try:
        doc_id = str(uuid.uuid4())
        
        # Base document structure
        document = {
            'id': doc_id,
            'type': doc_type,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Generate content based on type
        if doc_type == 'finance':
            document.update({
                'title': f"Financial Report {fake.company()}",
                'content': _generate_financial_content(),
                'category': 'financial_document',
                'currency': fake.currency_code()
            })
        elif doc_type == 'legal':
            document.update({
                'title': f"Legal Document - {fake.catch_phrase()}",
                'content': _generate_legal_content(),
                'category': 'legal_document',
                'jurisdiction': fake.country()
            })
        elif doc_type == 'medical':
            document.update({
                'title': f"Medical Record - {fake.name()}",
                'content': _generate_medical_content(),
                'category': 'medical_document',
                'patient_id': fake.uuid4()
            })
        else:
            document.update({
                'title': fake.sentence(nb_words=6),
                'content': _generate_general_content(),
                'category': 'general_document'
            })
        
        # Add annotations if requested
        if include_annotations:
            document['annotations'] = _generate_annotations(doc_type)
        
        # Add metadata if requested
        if include_metadata:
            document['metadata'] = _generate_metadata(doc_type)
        
        return document
        
    except Exception as e:
        logger.error(f"Failed to create mock document: {e}")
        raise


def _generate_financial_content() -> str:
    """Generate mock financial document content"""
    return f"""
    QUARTERLY FINANCIAL REPORT
    Company: {fake.company()}
    Quarter: Q{random.randint(1, 4)} {fake.year()}
    
    REVENUE SUMMARY:
    Total Revenue: ${fake.random_int(min=100000, max=10000000):,}
    Cost of Goods Sold: ${fake.random_int(min=50000, max=5000000):,}
    Gross Profit: ${fake.random_int(min=50000, max=5000000):,}
    
    EXPENSES:
    Operating Expenses: ${fake.random_int(min=10000, max=1000000):,}
    Marketing: ${fake.random_int(min=5000, max=500000):,}
    R&D: ${fake.random_int(min=5000, max=500000):,}
    
    NET INCOME: ${fake.random_int(min=10000, max=2000000):,}
    
    Prepared by: {fake.name()}
    Date: {fake.date()}
    """


def _generate_legal_content() -> str:
    """Generate mock legal document content"""
    return f"""
    LEGAL AGREEMENT
    
    This agreement is entered into on {fake.date()} between:
    
    Party A: {fake.company()}
    Address: {fake.address()}
    
    Party B: {fake.company()}
    Address: {fake.address()}
    
    TERMS AND CONDITIONS:
    
    1. Scope of Work
    {fake.paragraph(nb_sentences=3)}
    
    2. Payment Terms
    Total Amount: ${fake.random_int(min=1000, max=100000):,}
    Payment Schedule: {fake.sentence()}
    
    3. Duration
    Start Date: {fake.date()}
    End Date: {fake.future_date()}
    
    4. Termination
    {fake.paragraph(nb_sentences=2)}
    
    Signatures:
    Party A: _________________ Date: _______
    Party B: _________________ Date: _______
    """


def _generate_medical_content() -> str:
    """Generate mock medical document content"""
    return f"""
    MEDICAL RECORD
    
    Patient: {fake.name()}
    DOB: {fake.date_of_birth()}
    MRN: {fake.random_int(min=100000, max=999999)}
    
    VISIT INFORMATION:
    Date: {fake.date()}
    Provider: Dr. {fake.last_name()}
    Department: {fake.random_element(['Cardiology', 'Orthopedics', 'Neurology', 'Internal Medicine'])}
    
    CHIEF COMPLAINT:
    {fake.sentence()}
    
    HISTORY OF PRESENT ILLNESS:
    {fake.paragraph(nb_sentences=4)}
    
    PHYSICAL EXAMINATION:
    Vital Signs: BP {fake.random_int(100, 140)}/{fake.random_int(60, 90)}, 
                HR {fake.random_int(60, 100)}, 
                Temp {fake.random_int(98, 102)}°F
    
    ASSESSMENT AND PLAN:
    {fake.paragraph(nb_sentences=3)}
    
    Provider Signature: Dr. {fake.last_name()}
    Date: {fake.date()}
    """


def _generate_general_content() -> str:
    """Generate mock general document content"""
    paragraphs = [fake.paragraph(nb_sentences=random.randint(3, 6)) for _ in range(3)]
    return "\n\n".join(paragraphs)


def _generate_annotations(doc_type: str) -> List[Dict[str, Any]]:
    """Generate mock annotations for document"""
    annotations = []
    
    # Generate random bounding boxes and labels
    for i in range(random.randint(3, 8)):
        x = random.randint(10, 400)
        y = random.randint(10, 600)
        width = random.randint(50, 200)
        height = random.randint(20, 50)
        
        if doc_type == 'finance':
            labels = ['amount', 'date', 'company', 'account_number', 'total']
        elif doc_type == 'legal':
            labels = ['party', 'date', 'signature', 'amount', 'clause']
        elif doc_type == 'medical':
            labels = ['patient_name', 'date', 'diagnosis', 'medication', 'provider']
        else:
            labels = ['text', 'heading', 'date', 'name', 'number']
        
        annotation = {
            'id': str(uuid.uuid4()),
            'bbox': [x, y, width, height],
            'category_name': random.choice(labels),
            'category_id': random.randint(1, len(labels)),
            'area': width * height,
            'iscrowd': 0,
            'confidence': round(random.uniform(0.7, 1.0), 2)
        }
        annotations.append(annotation)
    
    return annotations


def _generate_metadata(doc_type: str) -> Dict[str, Any]:
    """Generate mock metadata for document"""
    metadata = {
        'language': fake.language_code(),
        'page_count': random.randint(1, 10),
        'word_count': random.randint(100, 2000),
        'created_by': fake.name(),
        'file_size': random.randint(10240, 1048576),  # 10KB to 1MB
        'format': random.choice(['pdf', 'docx', 'txt']),
        'version': f"{random.randint(1, 3)}.{random.randint(0, 9)}",
        'tags': [fake.word() for _ in range(random.randint(2, 5))]
    }
    
    if doc_type == 'finance':
        metadata.update({
            'fiscal_year': fake.year(),
            'quarter': f"Q{random.randint(1, 4)}",
            'currency': fake.currency_code(),
            'audited': fake.boolean()
        })
    elif doc_type == 'legal':
        metadata.update({
            'jurisdiction': fake.country(),
            'law_type': random.choice(['contract', 'agreement', 'policy', 'regulation']),
            'effective_date': fake.date().isoformat(),
            'expiry_date': fake.future_date().isoformat()
        })
    elif doc_type == 'medical':
        metadata.update({
            'specialty': random.choice(['cardiology', 'neurology', 'orthopedics']),
            'confidentiality': 'high',
            'retention_period': f"{random.randint(5, 30)} years"
        })
    
    return metadata


def create_test_data(count: int = 10, doc_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Create multiple test documents.
    
    Args:
        count: Number of documents to create
        doc_types: List of document types to use
    
    Returns:
        List of mock documents
    """
    if doc_types is None:
        doc_types = ['general', 'finance', 'legal', 'medical']
    
    documents = []
    for i in range(count):
        doc_type = random.choice(doc_types)
        doc = create_mock_document(
            doc_type=doc_type,
            include_annotations=True,
            include_metadata=True
        )
        documents.append(doc)
    
    return documents


def generate_test_files(output_dir: Union[str, Path], 
                       file_types: Optional[List[str]] = None,
                       count: int = 5) -> List[Path]:
    """
    Generate test files of various formats.
    
    Args:
        output_dir: Directory to create files in
        file_types: List of file types to create
        count: Number of files per type
    
    Returns:
        List of created file paths
    """
    if file_types is None:
        file_types = ['json', 'txt', 'csv', 'xml', 'image']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    for file_type in file_types:
        for i in range(count):
            filename = f"test_{file_type}_{i+1}"
            
            if file_type == 'json':
                file_path = output_path / f"{filename}.json"
                data = create_mock_document()
                write_file(file_path, json.dumps(data, indent=2))
            
            elif file_type == 'txt':
                file_path = output_path / f"{filename}.txt"
                content = _generate_general_content()
                write_file(file_path, content)
            
            elif file_type == 'csv':
                file_path = output_path / f"{filename}.csv"
                content = _generate_csv_content()
                write_file(file_path, content)
            
            elif file_type == 'xml':
                file_path = output_path / f"{filename}.xml"
                content = _generate_xml_content()
                write_file(file_path, content)
            
            elif file_type == 'image':
                file_path = output_path / f"{filename}.png"
                _create_test_image(file_path)
            
            created_files.append(file_path)
    
    return created_files


def _generate_csv_content() -> str:
    """Generate mock CSV content"""
    headers = ['id', 'name', 'email', 'department', 'salary', 'hire_date']
    rows = [headers]
    
    for i in range(random.randint(5, 15)):
        row = [
            str(i + 1),
            fake.name(),
            fake.email(),
            fake.random_element(['Engineering', 'Sales', 'Marketing', 'HR']),
            str(fake.random_int(min=40000, max=120000)),
            fake.date().isoformat()
        ]
        rows.append(row)
    
    return '\n'.join(','.join(row) for row in rows)


def _generate_xml_content() -> str:
    """Generate mock XML content"""
    return f"""
<?xml version="1.0" encoding="UTF-8"?>
<document>
    <header>
        <id>{uuid.uuid4()}</id>
        <title>{fake.sentence()}</title>
        <created>{fake.date().isoformat()}</created>
        <author>{fake.name()}</author>
    </header>
    <content>
        <section id="1">
            <title>{fake.sentence(nb_words=4)}</title>
            <text>{fake.paragraph()}</text>
        </section>
        <section id="2">
            <title>{fake.sentence(nb_words=4)}</title>
            <text>{fake.paragraph()}</text>
        </section>
    </content>
    <metadata>
        <language>{fake.language_code()}</language>
        <category>{fake.word()}</category>
        <tags>
            <tag>{fake.word()}</tag>
            <tag>{fake.word()}</tag>
        </tags>
    </metadata>
</document>
"""


def _create_test_image(file_path: Path, width: int = 400, height: int = 300):
    """Create a test image file"""
    # Create image with random background color
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    image = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(image)
    
    # Add some random shapes and text
    for _ in range(random.randint(3, 8)):
        x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
        x2, y2 = random.randint(width//2, width), random.randint(height//2, height)
        shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        shape_type = random.choice(['rectangle', 'ellipse', 'line'])
        if shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=shape_color)
        elif shape_type == 'ellipse':
            draw.ellipse([x1, y1, x2, y2], fill=shape_color)
        else:
            draw.line([x1, y1, x2, y2], fill=shape_color, width=3)
    
    # Add text
    text = fake.sentence(nb_words=3)
    text_color = (0, 0, 0)
    draw.text((20, 20), text, fill=text_color)
    
    image.save(file_path)


def setup_test_environment(temp_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Set up test environment with temporary directories and files.
    
    Args:
        temp_dir: Custom temporary directory path
    
    Returns:
        Dictionary with test environment paths and info
    """
    if temp_dir:
        test_dir = Path(temp_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
    else:
        test_dir = Path(tempfile.mkdtemp(prefix='synthdata_test_'))
    
    # Create subdirectories
    dirs = {
        'input': test_dir / 'input',
        'output': test_dir / 'output',
        'temp': test_dir / 'temp',
        'config': test_dir / 'config',
        'logs': test_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        create_directory(dir_path)
    
    # Create test configuration file
    config = {
        'test_mode': True,
        'log_level': 'DEBUG',
        'output_dir': str(dirs['output']),
        'temp_dir': str(dirs['temp'])
    }
    
    config_file = dirs['config'] / 'test_config.json'
    write_file(config_file, json.dumps(config, indent=2))
    
    # Generate some test files
    test_files = generate_test_files(dirs['input'], count=3)
    
    return {
        'base_dir': test_dir,
        'directories': dirs,
        'config_file': config_file,
        'test_files': test_files,
        'config': config
    }


def cleanup_test_environment(test_env: Dict[str, Any]):
    """
    Clean up test environment.
    
    Args:
        test_env: Test environment dictionary from setup_test_environment
    """
    import shutil
    
    try:
        base_dir = test_env['base_dir']
        if base_dir.exists():
            shutil.rmtree(base_dir)
            logger.debug(f"Cleaned up test environment: {base_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup test environment: {e}")


def assert_file_exists(file_path: Union[str, Path], message: str = None):
    """
    Assert that file exists.
    
    Args:
        file_path: Path to file
        message: Custom error message
    """
    path = Path(file_path)
    if not path.exists():
        msg = message or f"File does not exist: {file_path}"
        raise AssertionError(msg)


def assert_valid_json(json_string: str, message: str = None):
    """
    Assert that string is valid JSON.
    
    Args:
        json_string: JSON string to validate
        message: Custom error message
    """
    try:
        json.loads(json_string)
    except json.JSONDecodeError as e:
        msg = message or f"Invalid JSON: {e}"
        raise AssertionError(msg)


def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any], 
                        message: str = None):
    """
    Assert that actual dictionary contains all expected keys and values.
    
    Args:
        actual: Actual dictionary
        expected: Expected keys and values
        message: Custom error message
    """
    for key, value in expected.items():
        if key not in actual:
            msg = message or f"Missing key '{key}' in dictionary"
            raise AssertionError(msg)
        if actual[key] != value:
            msg = message or f"Value mismatch for key '{key}': expected {value}, got {actual[key]}"
            raise AssertionError(msg)


def generate_random_string(length: int = 10, 
                          chars: str = string.ascii_letters + string.digits) -> str:
    """
    Generate random string for testing.
    
    Args:
        length: String length
        chars: Characters to choose from
    
    Returns:
        Random string
    """
    return ''.join(random.choice(chars) for _ in range(length))


def create_performance_test_data(size: str = 'small') -> Dict[str, Any]:
    """
    Create test data for performance testing.
    
    Args:
        size: Data size ('small', 'medium', 'large')
    
    Returns:
        Performance test data
    """
    sizes = {
        'small': {'docs': 10, 'content_size': 100},
        'medium': {'docs': 100, 'content_size': 1000},
        'large': {'docs': 1000, 'content_size': 10000}
    }
    
    config = sizes.get(size, sizes['small'])
    
    documents = []
    for i in range(config['docs']):
        content = ' '.join(fake.words(nb=config['content_size']))
        doc = {
            'id': str(uuid.uuid4()),
            'content': content,
            'size': len(content),
            'created_at': datetime.now().isoformat()
        }
        documents.append(doc)
    
    return {
        'documents': documents,
        'total_docs': len(documents),
        'total_size': sum(len(doc['content']) for doc in documents),
        'config': config
    }


__all__ = [
    'create_mock_document',
    'create_test_data',
    'generate_test_files',
    'setup_test_environment',
    'cleanup_test_environment',
    'assert_file_exists',
    'assert_valid_json',
    'assert_dict_contains',
    'generate_random_string',
    'create_performance_test_data'
]