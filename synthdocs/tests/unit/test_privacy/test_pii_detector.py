"""
Unit tests for PII detection functionality.

Tests the PII detector for identifying personally identifiable information
in various document types and formats.
"""

import pytest
from unittest.mock import Mock, patch
import re

from structured_docs_synth.privacy.pii_protection.pii_detector import (
    PIIDetector,
    PIIType,
    PIIMatch,
    PIIDetectorConfig,
    PIIPattern
)
from structured_docs_synth.core.exceptions import PrivacyError


class TestPIIDetectorConfig:
    """Test PII detector configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PIIDetectorConfig()
        
        assert config.enabled_types == set(PIIType)  # All types enabled by default
        assert config.confidence_threshold == 0.85
        assert config.context_window == 50
        assert config.custom_patterns == []
        assert config.language == 'en'
    
    def test_custom_config(self):
        """Test custom configuration."""
        enabled = {PIIType.SSN, PIIType.EMAIL, PIIType.PHONE}
        custom_patterns = [
            PIIPattern(
                name='employee_id',
                pattern=r'EMP\d{6}',
                pii_type=PIIType.CUSTOM
            )
        ]
        
        config = PIIDetectorConfig(
            enabled_types=enabled,
            confidence_threshold=0.90,
            custom_patterns=custom_patterns
        )
        
        assert config.enabled_types == enabled
        assert config.confidence_threshold == 0.90
        assert len(config.custom_patterns) == 1
        assert config.custom_patterns[0].name == 'employee_id'


class TestPIIMatch:
    """Test PII match results."""
    
    def test_pii_match_creation(self):
        """Test creating PII match."""
        match = PIIMatch(
            pii_type=PIIType.SSN,
            value='123-45-6789',
            start_pos=10,
            end_pos=21,
            confidence=0.95,
            context='SSN: 123-45-6789 for processing'
        )
        
        assert match.pii_type == PIIType.SSN
        assert match.value == '123-45-6789'
        assert match.start_pos == 10
        assert match.end_pos == 21
        assert match.confidence == 0.95
        assert match.context == 'SSN: 123-45-6789 for processing'
    
    def test_pii_match_redacted_value(self):
        """Test getting redacted value."""
        match = PIIMatch(
            pii_type=PIIType.SSN,
            value='123-45-6789',
            start_pos=0,
            end_pos=11,
            confidence=1.0
        )
        
        assert match.get_redacted_value() == 'XXX-XX-XXXX'
        
        # Test email redaction
        email_match = PIIMatch(
            pii_type=PIIType.EMAIL,
            value='john.doe@example.com',
            start_pos=0,
            end_pos=20,
            confidence=1.0
        )
        
        assert email_match.get_redacted_value() == 'j***.d**@e******.com'


class TestPIIDetector:
    """Test PII detector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Provide PII detector instance."""
        return PIIDetector()
    
    @pytest.fixture
    def custom_detector(self):
        """Provide PII detector with custom patterns."""
        custom_patterns = [
            PIIPattern(
                name='employee_id',
                pattern=r'EMP\d{6}',
                pii_type=PIIType.CUSTOM
            ),
            PIIPattern(
                name='case_number',
                pattern=r'CASE-\d{4}-\d{6}',
                pii_type=PIIType.CUSTOM
            )
        ]
        
        config = PIIDetectorConfig(custom_patterns=custom_patterns)
        return PIIDetector(config)
    
    def test_detect_ssn(self, detector):
        """Test SSN detection."""
        text = "The patient's SSN is 123-45-6789 for insurance."
        
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.SSN
        assert matches[0].value == '123-45-6789'
        assert matches[0].confidence >= 0.95
    
    def test_detect_multiple_ssn_formats(self, detector):
        """Test detecting SSNs in various formats."""
        text = """
        SSN formats:
        123-45-6789
        123 45 6789
        123456789
        """
        
        matches = detector.detect(text)
        
        assert len(matches) == 3
        assert all(m.pii_type == PIIType.SSN for m in matches)
    
    def test_detect_email(self, detector):
        """Test email detection."""
        text = "Contact me at john.doe@example.com or jane_smith+test@company.co.uk"
        
        matches = detector.detect(text)
        
        assert len(matches) == 2
        assert all(m.pii_type == PIIType.EMAIL for m in matches)
        assert matches[0].value == 'john.doe@example.com'
        assert matches[1].value == 'jane_smith+test@company.co.uk'
    
    def test_detect_phone_numbers(self, detector):
        """Test phone number detection."""
        text = """
        Call us at:
        (555) 123-4567
        555-123-4567
        +1-555-123-4567
        555.123.4567
        5551234567
        """
        
        matches = detector.detect(text)
        
        assert len(matches) >= 4  # Different formats detected
        assert all(m.pii_type == PIIType.PHONE for m in matches)
    
    def test_detect_credit_card(self, detector):
        """Test credit card detection."""
        text = """
        Payment cards:
        4111 1111 1111 1111 (Visa)
        5500-0000-0000-0004 (MasterCard)
        378282246310005 (Amex)
        """
        
        matches = detector.detect(text)
        
        assert len(matches) == 3
        assert all(m.pii_type == PIIType.CREDIT_CARD for m in matches)
    
    def test_detect_names(self, detector):
        """Test name detection."""
        text = "Meeting between John Smith and Dr. Sarah Johnson-Williams"
        
        matches = detector.detect(text)
        
        # Name detection is complex, should find at least some names
        name_matches = [m for m in matches if m.pii_type == PIIType.NAME]
        assert len(name_matches) >= 2
    
    def test_detect_addresses(self, detector):
        """Test address detection."""
        text = """
        Ship to:
        123 Main Street, Apt 4B
        New York, NY 10001
        """
        
        matches = detector.detect(text)
        
        address_matches = [m for m in matches if m.pii_type == PIIType.ADDRESS]
        assert len(address_matches) >= 1
    
    def test_detect_date_of_birth(self, detector):
        """Test date of birth detection."""
        text = """
        DOB: 01/15/1990
        Date of Birth: January 15, 1990
        Born: 15-01-1990
        """
        
        matches = detector.detect(text)
        
        dob_matches = [m for m in matches if m.pii_type == PIIType.DATE_OF_BIRTH]
        assert len(dob_matches) >= 2
    
    def test_detect_medical_record_number(self, detector):
        """Test medical record number detection."""
        text = "Patient MRN: 12345678 admitted on 01/01/2024"
        
        matches = detector.detect(text)
        
        mrn_matches = [m for m in matches if m.pii_type == PIIType.MEDICAL_RECORD]
        assert len(mrn_matches) >= 1
    
    def test_detect_custom_patterns(self, custom_detector):
        """Test custom pattern detection."""
        text = """
        Employee ID: EMP123456
        Case Reference: CASE-2024-000123
        """
        
        matches = custom_detector.detect(text)
        
        custom_matches = [m for m in matches if m.pii_type == PIIType.CUSTOM]
        assert len(custom_matches) == 2
        assert any('EMP123456' in m.value for m in custom_matches)
        assert any('CASE-2024-000123' in m.value for m in custom_matches)
    
    def test_confidence_threshold(self, detector):
        """Test confidence threshold filtering."""
        detector.config.confidence_threshold = 0.9
        
        # Ambiguous text that might be PII
        text = "Reference number: 123456789"  # Could be SSN without dashes
        
        matches = detector.detect(text)
        
        # Should filter out low confidence matches
        high_confidence = [m for m in matches if m.confidence >= 0.9]
        assert len(high_confidence) == len(matches)
    
    def test_disabled_pii_types(self, detector):
        """Test disabling specific PII types."""
        detector.config.enabled_types = {PIIType.SSN, PIIType.EMAIL}
        
        text = """
        SSN: 123-45-6789
        Email: test@example.com
        Phone: (555) 123-4567
        Credit Card: 4111-1111-1111-1111
        """
        
        matches = detector.detect(text)
        
        # Should only detect SSN and email
        detected_types = {m.pii_type for m in matches}
        assert detected_types == {PIIType.SSN, PIIType.EMAIL}
    
    def test_context_extraction(self, detector):
        """Test context extraction around PII."""
        detector.config.context_window = 20
        
        text = "This is a long text with SSN 123-45-6789 embedded in the middle of the content."
        
        matches = detector.detect(text)
        
        assert len(matches) == 1
        context = matches[0].context
        assert len(context) <= 60  # 20 chars before + PII + 20 chars after
        assert '123-45-6789' in context
    
    def test_overlapping_matches(self, detector):
        """Test handling of overlapping PII matches."""
        text = "Email and phone: john@example.com (555-123-4567)"
        
        matches = detector.detect(text)
        
        # Should detect both without overlap issues
        assert len(matches) == 2
        
        # Verify no overlapping positions
        for i, m1 in enumerate(matches):
            for m2 in matches[i+1:]:
                assert m1.end_pos <= m2.start_pos or m2.end_pos <= m1.start_pos
    
    def test_batch_detection(self, detector):
        """Test batch PII detection."""
        documents = [
            "SSN: 123-45-6789",
            "Email: test@example.com",
            "Phone: (555) 123-4567"
        ]
        
        all_matches = detector.detect_batch(documents)
        
        assert len(all_matches) == 3
        assert len(all_matches[0]) == 1  # SSN
        assert len(all_matches[1]) == 1  # Email
        assert len(all_matches[2]) == 1  # Phone
    
    def test_redact_pii(self, detector):
        """Test PII redaction."""
        text = "Contact John at john@example.com or 555-123-4567"
        
        redacted = detector.redact(text)
        
        assert 'john@example.com' not in redacted
        assert '555-123-4567' not in redacted
        assert 'j***@e******.com' in redacted or '[EMAIL]' in redacted
        assert '[PHONE]' in redacted or 'XXX-XXX-XXXX' in redacted
    
    def test_get_pii_summary(self, detector):
        """Test PII summary generation."""
        text = """
        Patient: John Smith
        SSN: 123-45-6789
        DOB: 01/15/1990
        Email: john.smith@email.com
        Phone: (555) 123-4567
        MRN: 12345678
        """
        
        summary = detector.get_pii_summary(text)
        
        assert summary['total_pii_found'] >= 5
        assert PIIType.SSN in summary['pii_types']
        assert PIIType.EMAIL in summary['pii_types']
        assert PIIType.PHONE in summary['pii_types']
        assert summary['risk_score'] > 0.5  # High risk due to multiple PII
    
    def test_international_formats(self, detector):
        """Test international PII format detection."""
        detector.config.language = 'multi'
        
        text = """
        UK Phone: +44 20 7123 4567
        German Phone: +49 30 12345678
        French SSN: 1 85 12 78 123 456 78
        UK Postcode: SW1A 1AA
        """
        
        matches = detector.detect(text)
        
        # Should detect international phone numbers
        phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) >= 2
    
    def test_false_positive_handling(self, detector):
        """Test false positive reduction."""
        text = """
        Product codes:
        123-45-6789 (not an SSN)
        ISBN: 978-3-16-148410-0
        Serial: 4111-1111-1111-1111-XYZ
        """
        
        # With context analysis, should reduce false positives
        matches = detector.detect(text)
        
        # Verify confidence scores reflect uncertainty
        for match in matches:
            if 'Product code' in match.context or 'ISBN' in match.context:
                assert match.confidence < 0.8