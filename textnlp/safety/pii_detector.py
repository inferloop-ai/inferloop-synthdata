"""
PII Detection Implementation for TextNLP
Advanced PII detection and masking for generated text content
"""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import json
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
import phonenumbers
from email_validator import validate_email, EmailNotValidError

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected"""
    EMAIL = "email"
    PHONE = "phone_number"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"
    PERSON_NAME = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    DATE_TIME = "date_time"
    MEDICAL_LICENSE = "medical_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    DRIVERS_LICENSE = "drivers_license"
    ADDRESS = "address"
    AGE = "age"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """Represents a detected PII match"""
    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float
    context: str = ""
    replacement: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIDetectionResult:
    """Result of PII detection on text"""
    original_text: str
    masked_text: str
    pii_matches: List[PIIMatch]
    detection_time: float
    total_pii_count: int
    pii_types_found: Set[PIIType]
    risk_level: str  # "low", "medium", "high"


class PIIDetector:
    """Advanced PII detection using multiple strategies"""
    
    def __init__(self, languages: List[str] = ["en"], 
                 confidence_threshold: float = 0.5,
                 mask_mode: str = "replace"):
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.mask_mode = mask_mode  # "replace", "hash", "remove"
        
        # Initialize Presidio engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Custom regex patterns
        self.custom_patterns = self._initialize_custom_patterns()
        
        # Replacement mappings
        self.replacements = {
            PIIType.EMAIL: "[EMAIL]",
            PIIType.PHONE: "[PHONE]",
            PIIType.SSN: "[SSN]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.IP_ADDRESS: "[IP_ADDRESS]",
            PIIType.URL: "[URL]",
            PIIType.PERSON_NAME: "[PERSON]",
            PIIType.LOCATION: "[LOCATION]",
            PIIType.ORGANIZATION: "[ORGANIZATION]",
            PIIType.DATE_TIME: "[DATE]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.AGE: "[AGE]"
        }
    
    def _initialize_custom_patterns(self) -> Dict[PIIType, List[re.Pattern]]:
        """Initialize custom regex patterns for PII detection"""
        patterns = {
            PIIType.SSN: [
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # 123-45-6789
                re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),  # 123 45 6789
                re.compile(r'\b\d{9}\b'),  # 123456789
            ],
            PIIType.CREDIT_CARD: [
                re.compile(r'\b4[0-9]{12}(?:[0-9]{3})?\b'),  # Visa
                re.compile(r'\b5[1-5][0-9]{14}\b'),  # MasterCard
                re.compile(r'\b3[47][0-9]{13}\b'),  # American Express
                re.compile(r'\b3[0-9]{4}\s[0-9]{6}\s[0-9]{5}\b'),  # Amex with spaces
            ],
            PIIType.IP_ADDRESS: [
                re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),  # IPv4
                re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),  # IPv6
            ],
            PIIType.URL: [
                re.compile(r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?'),
                re.compile(r'www\.(?:[-\w.])+\.(?:[a-zA-Z]{2,4})(?:/(?:[\w/_.])*)?')
            ],
            PIIType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            ],
            PIIType.PHONE: [
                re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),  # 123-456-7890
                re.compile(r'\(\d{3}\)\s\d{3}-\d{4}'),  # (123) 456-7890
                re.compile(r'\b\d{10}\b'),  # 1234567890
                re.compile(r'\+1\s\d{3}\s\d{3}\s\d{4}'),  # +1 123 456 7890
            ],
            PIIType.DRIVERS_LICENSE: [
                re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),  # State DL patterns
                re.compile(r'\bDL\s*[A-Z0-9]{8,12}\b'),
            ],
            PIIType.PASSPORT: [
                re.compile(r'\b[A-Z]{2}\d{7}\b'),  # US Passport
                re.compile(r'\bPassport\s*[A-Z0-9]{6,9}\b'),
            ],
            PIIType.MEDICAL_LICENSE: [
                re.compile(r'\bNPI\s*\d{10}\b'),  # National Provider Identifier
                re.compile(r'\bDEA\s*[A-Z]{2}\d{7}\b'),  # DEA Number
            ],
            PIIType.BANK_ACCOUNT: [
                re.compile(r'\bAccount\s*#?\s*\d{8,17}\b'),
                re.compile(r'\bRouting\s*#?\s*\d{9}\b'),
            ],
            PIIType.AGE: [
                re.compile(r'\b(?:age|aged?)\s+(\d{1,3})\b', re.IGNORECASE),
                re.compile(r'\b(\d{1,3})\s+years?\s+old\b', re.IGNORECASE),
                re.compile(r'\bI\s+am\s+(\d{1,3})\b', re.IGNORECASE),
            ]
        }
        return patterns
    
    async def detect_pii(self, text: str, context: str = "") -> PIIDetectionResult:
        """Detect PII in text using multiple detection strategies"""
        start_time = asyncio.get_event_loop().time()
        
        # Combine all detection methods
        pii_matches = []
        
        # 1. Use Presidio for comprehensive detection
        presidio_matches = await self._detect_with_presidio(text)
        pii_matches.extend(presidio_matches)
        
        # 2. Use custom regex patterns
        regex_matches = await self._detect_with_regex(text)
        pii_matches.extend(regex_matches)
        
        # 3. Use spaCy NER for entities
        ner_matches = await self._detect_with_ner(text)
        pii_matches.extend(ner_matches)
        
        # 4. Use custom validation for specific types
        validation_matches = await self._detect_with_validation(text)
        pii_matches.extend(validation_matches)
        
        # Remove duplicates and filter by confidence
        pii_matches = self._deduplicate_matches(pii_matches)
        pii_matches = [m for m in pii_matches if m.confidence >= self.confidence_threshold]
        
        # Sort by position
        pii_matches.sort(key=lambda x: x.start)
        
        # Generate masked text
        masked_text = self._mask_text(text, pii_matches)
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(pii_matches)
        
        detection_time = asyncio.get_event_loop().time() - start_time
        
        return PIIDetectionResult(
            original_text=text,
            masked_text=masked_text,
            pii_matches=pii_matches,
            detection_time=detection_time,
            total_pii_count=len(pii_matches),
            pii_types_found=set(match.pii_type for match in pii_matches),
            risk_level=risk_level
        )
    
    async def _detect_with_presidio(self, text: str) -> List[PIIMatch]:
        """Detect PII using Microsoft Presidio"""
        try:
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "CRYPTO", "DATE_TIME", "DOMAIN_NAME", "IBAN_CODE",
                    "IP_ADDRESS", "LOCATION", "MEDICAL_LICENSE", "NRP",
                    "ORGANIZATION", "PASSPORT", "SSN", "UK_NHS",
                    "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE",
                    "US_ITIN", "US_PASSPORT", "US_SSN"
                ]
            )
            
            matches = []
            for result in results:
                pii_type = self._map_presidio_entity(result.entity_type)
                if pii_type:
                    match = PIIMatch(
                        pii_type=pii_type,
                        text=text[result.start:result.end],
                        start=result.start,
                        end=result.end,
                        confidence=result.score,
                        metadata={"detector": "presidio", "entity_type": result.entity_type}
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Presidio detection failed: {e}")
            return []
    
    async def _detect_with_regex(self, text: str) -> List[PIIMatch]:
        """Detect PII using custom regex patterns"""
        matches = []
        
        for pii_type, patterns in self.custom_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Additional validation for certain types
                    if await self._validate_match(pii_type, match.group()):
                        pii_match = PIIMatch(
                            pii_type=pii_type,
                            text=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8,  # Default confidence for regex
                            metadata={"detector": "regex", "pattern": pattern.pattern}
                        )
                        matches.append(pii_match)
        
        return matches
    
    async def _detect_with_ner(self, text: str) -> List[PIIMatch]:
        """Detect PII using spaCy Named Entity Recognition"""
        try:
            doc = self.nlp(text)
            matches = []
            
            for ent in doc.ents:
                pii_type = self._map_spacy_entity(ent.label_)
                if pii_type:
                    confidence = 0.7  # Default confidence for NER
                    
                    # Adjust confidence based on entity type and context
                    if ent.label_ in ["PERSON", "ORG", "GPE"]:
                        confidence = min(0.9, confidence + 0.2)
                    
                    match = PIIMatch(
                        pii_type=pii_type,
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=confidence,
                        metadata={
                            "detector": "spacy",
                            "entity_label": ent.label_,
                            "entity_id": ent.ent_id_
                        }
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"spaCy NER detection failed: {e}")
            return []
    
    async def _detect_with_validation(self, text: str) -> List[PIIMatch]:
        """Detect PII using validation functions"""
        matches = []
        
        # Email validation
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        for match in email_pattern.finditer(text):
            email = match.group()
            try:
                validate_email(email)
                pii_match = PIIMatch(
                    pii_type=PIIType.EMAIL,
                    text=email,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    metadata={"detector": "validation", "validation_type": "email"}
                )
                matches.append(pii_match)
            except EmailNotValidError:
                pass
        
        # Phone number validation
        phone_patterns = [
            r'\b\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            r'\b(\d{3})[-.](\d{3})[-.](\d{4})\b'
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                phone = match.group()
                try:
                    # Normalize phone number
                    normalized = re.sub(r'[^\d+]', '', phone)
                    if normalized.startswith('1') and len(normalized) == 11:
                        normalized = '+' + normalized
                    elif len(normalized) == 10:
                        normalized = '+1' + normalized
                    
                    parsed = phonenumbers.parse(normalized, None)
                    if phonenumbers.is_valid_number(parsed):
                        pii_match = PIIMatch(
                            pii_type=PIIType.PHONE,
                            text=phone,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.95,
                            metadata={
                                "detector": "validation",
                                "validation_type": "phone",
                                "country": phonenumbers.region_code_for_number(parsed)
                            }
                        )
                        matches.append(pii_match)
                except phonenumbers.NumberParseException:
                    pass
        
        return matches
    
    async def _validate_match(self, pii_type: PIIType, text: str) -> bool:
        """Additional validation for specific PII types"""
        if pii_type == PIIType.SSN:
            # Basic SSN validation
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) == 9:
                # Check for invalid SSN patterns
                invalid_patterns = [
                    '000', '666', '900', '999',  # Invalid area numbers
                    '00',  # Invalid group number
                    '0000'  # Invalid serial number
                ]
                area = digits[:3]
                group = digits[3:5]
                serial = digits[5:]
                
                return (area not in invalid_patterns and 
                       group != '00' and 
                       serial != '0000')
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn algorithm validation
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) >= 13:
                return self._luhn_check(digits)
        
        elif pii_type == PIIType.IP_ADDRESS:
            # IPv4 validation
            parts = text.split('.')
            if len(parts) == 4:
                try:
                    return all(0 <= int(part) <= 255 for part in parts)
                except ValueError:
                    pass
        
        return True  # Default to valid for other types
    
    def _luhn_check(self, digits: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        digits = [int(d) for d in digits[::-1]]
        checksum = sum(digits[::2])
        
        for digit in digits[1::2]:
            doubled = digit * 2
            checksum += doubled if doubled < 10 else doubled - 9
        
        return checksum % 10 == 0
    
    def _map_presidio_entity(self, entity_type: str) -> Optional[PIIType]:
        """Map Presidio entity types to PIIType enum"""
        mapping = {
            "PERSON": PIIType.PERSON_NAME,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "DATE_TIME": PIIType.DATE_TIME,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "LOCATION": PIIType.LOCATION,
            "MEDICAL_LICENSE": PIIType.MEDICAL_LICENSE,
            "ORGANIZATION": PIIType.ORGANIZATION,
            "PASSPORT": PIIType.PASSPORT,
            "SSN": PIIType.SSN,
            "US_SSN": PIIType.SSN,
            "URL": PIIType.URL,
            "US_BANK_NUMBER": PIIType.BANK_ACCOUNT,
            "US_DRIVER_LICENSE": PIIType.DRIVERS_LICENSE,
            "US_PASSPORT": PIIType.PASSPORT
        }
        return mapping.get(entity_type)
    
    def _map_spacy_entity(self, entity_label: str) -> Optional[PIIType]:
        """Map spaCy entity labels to PIIType enum"""
        mapping = {
            "PERSON": PIIType.PERSON_NAME,
            "ORG": PIIType.ORGANIZATION,
            "GPE": PIIType.LOCATION,  # Geopolitical entity
            "LOC": PIIType.LOCATION,
            "DATE": PIIType.DATE_TIME,
            "TIME": PIIType.DATE_TIME
        }
        return mapping.get(entity_label)
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove duplicate and overlapping matches"""
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda x: x.start)
        
        deduplicated = []
        for match in matches:
            # Check for overlaps with existing matches
            overlapping = False
            for existing in deduplicated:
                # Check if ranges overlap
                if (match.start < existing.end and match.end > existing.start):
                    # Keep the match with higher confidence
                    if match.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    overlapping = True
                    break
            
            if not overlapping:
                deduplicated.append(match)
        
        return deduplicated
    
    def _mask_text(self, text: str, matches: List[PIIMatch]) -> str:
        """Mask detected PII in text"""
        if not matches:
            return text
        
        # Sort matches by position (reverse order for replacement)
        matches.sort(key=lambda x: x.start, reverse=True)
        
        masked_text = text
        for match in matches:
            if self.mask_mode == "replace":
                replacement = self.replacements.get(match.pii_type, "[PII]")
            elif self.mask_mode == "hash":
                replacement = f"[{hashlib.md5(match.text.encode()).hexdigest()[:8]}]"
            elif self.mask_mode == "remove":
                replacement = ""
            else:
                replacement = "[REDACTED]"
            
            masked_text = (
                masked_text[:match.start] + 
                replacement + 
                masked_text[match.end:]
            )
            match.replacement = replacement
        
        return masked_text
    
    def _calculate_risk_level(self, matches: List[PIIMatch]) -> str:
        """Calculate risk level based on detected PII"""
        if not matches:
            return "low"
        
        high_risk_types = {
            PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT,
            PIIType.DRIVERS_LICENSE, PIIType.BANK_ACCOUNT,
            PIIType.MEDICAL_LICENSE
        }
        
        medium_risk_types = {
            PIIType.EMAIL, PIIType.PHONE, PIIType.ADDRESS,
            PIIType.PERSON_NAME
        }
        
        pii_types = set(match.pii_type for match in matches)
        
        if pii_types & high_risk_types:
            return "high"
        elif len(pii_types & medium_risk_types) >= 2 or len(matches) > 5:
            return "medium"
        else:
            return "low"
    
    async def batch_detect(self, texts: List[str]) -> List[PIIDetectionResult]:
        """Detect PII in multiple texts concurrently"""
        tasks = [self.detect_pii(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def create_detection_report(self, results: List[PIIDetectionResult]) -> Dict[str, Any]:
        """Create a comprehensive detection report"""
        total_texts = len(results)
        texts_with_pii = sum(1 for r in results if r.total_pii_count > 0)
        total_pii_instances = sum(r.total_pii_count for r in results)
        
        # Count by type
        type_counts = {}
        for result in results:
            for pii_type in result.pii_types_found:
                type_counts[pii_type.value] = type_counts.get(pii_type.value, 0) + 1
        
        # Risk distribution
        risk_distribution = {"low": 0, "medium": 0, "high": 0}
        for result in results:
            risk_distribution[result.risk_level] += 1
        
        return {
            "summary": {
                "total_texts_analyzed": total_texts,
                "texts_with_pii": texts_with_pii,
                "pii_detection_rate": texts_with_pii / total_texts if total_texts > 0 else 0,
                "total_pii_instances": total_pii_instances,
                "average_pii_per_text": total_pii_instances / total_texts if total_texts > 0 else 0
            },
            "pii_type_distribution": type_counts,
            "risk_distribution": risk_distribution,
            "detection_performance": {
                "average_detection_time": sum(r.detection_time for r in results) / total_texts if total_texts > 0 else 0,
                "total_detection_time": sum(r.detection_time for r in results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class PIIDetectionService:
    """Service wrapper for PII detection with caching and configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = PIIDetector(
            languages=config.get("languages", ["en"]),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            mask_mode=config.get("mask_mode", "replace")
        )
        self.cache = {}  # Simple in-memory cache
        self.cache_enabled = config.get("enable_cache", True)
        self.max_cache_size = config.get("max_cache_size", 1000)
    
    async def detect_and_mask(self, text: str, cache_key: Optional[str] = None) -> PIIDetectionResult:
        """Detect and mask PII with optional caching"""
        if self.cache_enabled and cache_key:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        result = await self.detector.detect_pii(text)
        
        if self.cache_enabled and cache_key:
            # Simple LRU cache implementation
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_enabled": self.cache_enabled,
            "config": self.config
        }


# Example usage
if __name__ == "__main__":
    async def example():
        # Sample text with various PII
        sample_text = """
        Hi John Smith, your email john.smith@example.com has been verified.
        Please call us at (555) 123-4567 or visit our office at 123 Main St, New York, NY.
        Your SSN 123-45-6789 and credit card 4111-1111-1111-1111 are on file.
        IP address: 192.168.1.1
        """
        
        # Initialize detector
        detector = PIIDetector(confidence_threshold=0.5)
        
        # Detect PII
        result = await detector.detect_pii(sample_text)
        
        print(f"Original text: {result.original_text}")
        print(f"Masked text: {result.masked_text}")
        print(f"Risk level: {result.risk_level}")
        print(f"PII found: {len(result.pii_matches)} instances")
        
        for match in result.pii_matches:
            print(f"- {match.pii_type.value}: {match.text} (confidence: {match.confidence:.2f})")
    
    # Run example
    # asyncio.run(example())