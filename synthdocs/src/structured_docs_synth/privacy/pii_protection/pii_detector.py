"""
PII (Personally Identifiable Information) detection module
"""

import re
import json
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ...core import get_logger, PrivacyError


class PIIType(Enum):
    """Types of PII that can be detected"""
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    MEDICAL_RECORD = "medical_record"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """Represents a detected PII match"""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    field_name: Optional[str] = None
    context: Optional[str] = None


@dataclass
class PIIDetectionResult:
    """Result of PII detection on a document or text"""
    text: str
    matches: List[PIIMatch]
    total_matches: int
    pii_types_found: Set[PIIType]
    has_pii: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL


class PIIDetector:
    """Main PII detection engine"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Regex patterns for different PII types
        self.patterns = {
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                r'\b\d{3}\s\d{2}\s\d{4}\b',  # XXX XX XXXX
                r'\b\d{9}\b'  # XXXXXXXXX
            ],
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIIType.PHONE: [
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (XXX) XXX-XXXX
                r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
                r'\b\d{3}\.\d{3}\.\d{4}\b',  # XXX.XXX.XXXX
                r'\b\d{10}\b'  # XXXXXXXXXX
            ],
            PIIType.CREDIT_CARD: [
                r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Visa
                r'\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # MasterCard
                r'\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b',  # American Express
                r'\b6011[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'  # Discover
            ],
            PIIType.DATE_OF_BIRTH: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # MM/DD/YYYY or M/D/YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2}\b'  # MM/DD/YY
            ],
            PIIType.DRIVER_LICENSE: [
                r'\b[A-Z]{1,2}\d{6,8}\b',  # State format (varies)
                r'\bDL\s*\d{8,12}\b'
            ],
            PIIType.MEDICAL_RECORD: [
                r'\bMRN\s*:?\s*\d{6,12}\b',
                r'\bMedical\s+Record\s+Number\s*:?\s*\d{6,12}\b'
            ],
            PIIType.BANK_ACCOUNT: [
                r'\b\d{8,17}\b'  # Basic bank account pattern
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # IPv4
            ]
        }
        
        # Name patterns (more complex due to variation)
        self.name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # Last, First
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # First M. Last
        ]
        
        # Address patterns
        self.address_patterns = [
            r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
            r'\bP\.?O\.?\s+Box\s+\d+\b'
        ]
        
        # Common first and last names for better name detection
        self.common_first_names = {
            'james', 'mary', 'john', 'patricia', 'robert', 'jennifer', 'michael', 'linda',
            'william', 'elizabeth', 'david', 'barbara', 'richard', 'susan', 'joseph', 'jessica',
            'thomas', 'sarah', 'christopher', 'karen', 'charles', 'nancy', 'daniel', 'lisa'
        }
        
        self.common_last_names = {
            'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis',
            'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson', 'thomas'
        }
        
        self.logger.info("PII Detector initialized")
    
    def detect_pii(self, text: str, field_name: Optional[str] = None) -> PIIDetectionResult:
        """Detect PII in the given text"""
        
        matches = []
        
        # Detect each PII type
        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    confidence = self._calculate_confidence(pii_type, match.group(), text, match.start())
                    
                    if confidence > 0.3:  # Threshold for inclusion
                        pii_match = PIIMatch(
                            pii_type=pii_type,
                            value=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                            field_name=field_name,
                            context=self._get_context(text, match.start(), match.end())
                        )
                        matches.append(pii_match)
        
        # Detect names with special handling
        name_matches = self._detect_names(text, field_name)
        matches.extend(name_matches)
        
        # Detect addresses with special handling
        address_matches = self._detect_addresses(text, field_name)
        matches.extend(address_matches)
        
        # Remove duplicates and overlaps
        matches = self._deduplicate_matches(matches)
        
        # Create result
        pii_types_found = {match.pii_type for match in matches}
        risk_level = self._assess_risk_level(matches)
        
        result = PIIDetectionResult(
            text=text,
            matches=matches,
            total_matches=len(matches),
            pii_types_found=pii_types_found,
            has_pii=len(matches) > 0,
            risk_level=risk_level
        )
        
        if result.has_pii:
            self.logger.warning(f"PII detected: {len(matches)} matches found ({risk_level} risk)")
        
        return result
    
    def detect_pii_in_document(self, document_data: Dict[str, Any]) -> Dict[str, PIIDetectionResult]:
        """Detect PII in a complete document data structure"""
        
        results = {}
        
        def process_value(key: str, value: Any, path: str = ""):
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, str):
                result = self.detect_pii(value, field_name=current_path)
                if result.has_pii:
                    results[current_path] = result
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    process_value(sub_key, sub_value, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    process_value(f"[{i}]", item, current_path)
        
        # Process all fields in the document
        if isinstance(document_data, dict):
            for key, value in document_data.items():
                process_value(key, value)
        
        return results
    
    def _detect_names(self, text: str, field_name: Optional[str] = None) -> List[PIIMatch]:
        """Detect potential names with higher accuracy"""
        
        matches = []
        
        # Higher confidence if field name suggests it's a name field
        field_confidence_boost = 0.4 if field_name and any(
            name_indicator in field_name.lower() 
            for name_indicator in ['name', 'patient', 'applicant', 'taxpayer', 'person']
        ) else 0.0
        
        for pattern in self.name_patterns:
            for match in re.finditer(pattern, text):
                potential_name = match.group().strip()
                confidence = self._calculate_name_confidence(potential_name) + field_confidence_boost
                
                if confidence > 0.5:
                    pii_match = PIIMatch(
                        pii_type=PIIType.NAME,
                        value=potential_name,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=min(confidence, 1.0),
                        field_name=field_name,
                        context=self._get_context(text, match.start(), match.end())
                    )
                    matches.append(pii_match)
        
        return matches
    
    def _detect_addresses(self, text: str, field_name: Optional[str] = None) -> List[PIIMatch]:
        """Detect potential addresses"""
        
        matches = []
        
        # Higher confidence if field name suggests it's an address field
        field_confidence_boost = 0.3 if field_name and 'address' in field_name.lower() else 0.0
        
        for pattern in self.address_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                confidence = 0.7 + field_confidence_boost
                
                pii_match = PIIMatch(
                    pii_type=PIIType.ADDRESS,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=min(confidence, 1.0),
                    field_name=field_name,
                    context=self._get_context(text, match.start(), match.end())
                )
                matches.append(pii_match)
        
        return matches
    
    def _calculate_confidence(self, pii_type: PIIType, value: str, text: str, position: int) -> float:
        """Calculate confidence score for a PII match"""
        
        base_confidence = 0.6
        
        # Adjust confidence based on PII type
        type_confidence = {
            PIIType.SSN: 0.9,
            PIIType.EMAIL: 0.95,
            PIIType.CREDIT_CARD: 0.85,
            PIIType.PHONE: 0.8,
            PIIType.DATE_OF_BIRTH: 0.6,
            PIIType.IP_ADDRESS: 0.7
        }.get(pii_type, base_confidence)
        
        # Additional validation for specific types
        if pii_type == PIIType.SSN:
            # Basic SSN validation (not 000-00-0000, etc.)
            clean_ssn = re.sub(r'[^0-9]', '', value)
            if clean_ssn in ['000000000', '123456789'] or clean_ssn.startswith('000'):
                type_confidence *= 0.3
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Basic Luhn algorithm check could be added here
            pass
        
        return min(type_confidence, 1.0)
    
    def _calculate_name_confidence(self, name: str) -> float:
        """Calculate confidence that a string is actually a name"""
        
        parts = name.split()
        if len(parts) < 2:
            return 0.2
        
        confidence = 0.4
        
        # Check if parts are in common names lists
        for part in parts:
            part_lower = part.lower().replace(',', '')
            if part_lower in self.common_first_names or part_lower in self.common_last_names:
                confidence += 0.2
        
        # Bonus for proper capitalization
        if all(part[0].isupper() and part[1:].islower() for part in parts if part.isalpha()):
            confidence += 0.2
        
        # Penalty for numbers or special characters
        if any(char.isdigit() for char in name):
            confidence -= 0.3
        
        return min(confidence, 1.0)
    
    def _get_context(self, text: str, start: int, end: int, context_size: int = 20) -> str:
        """Get surrounding context for a PII match"""
        
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        context = text[context_start:context_end]
        
        # Mark the actual match
        match_start = start - context_start
        match_end = end - context_start
        
        return (
            context[:match_start] + 
            "[" + context[match_start:match_end] + "]" + 
            context[match_end:]
        ).strip()
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove duplicate and overlapping matches, keeping the highest confidence ones"""
        
        if not matches:
            return matches
        
        # Sort by position
        matches.sort(key=lambda x: (x.start_pos, x.end_pos))
        
        deduplicated = []
        last_end = -1
        
        for match in matches:
            # Skip if this match overlaps with the previous one
            if match.start_pos < last_end:
                # Keep the one with higher confidence
                if deduplicated and match.confidence > deduplicated[-1].confidence:
                    deduplicated[-1] = match
                continue
            
            deduplicated.append(match)
            last_end = match.end_pos
        
        return deduplicated
    
    def _assess_risk_level(self, matches: List[PIIMatch]) -> str:
        """Assess the overall risk level based on PII matches found"""
        
        if not matches:
            return "LOW"
        
        high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_RECORD, PIIType.BANK_ACCOUNT}
        medium_risk_types = {PIIType.EMAIL, PIIType.PHONE, PIIType.DATE_OF_BIRTH, PIIType.DRIVER_LICENSE}
        
        has_high_risk = any(match.pii_type in high_risk_types for match in matches)
        has_medium_risk = any(match.pii_type in medium_risk_types for match in matches)
        
        total_matches = len(matches)
        unique_types = len(set(match.pii_type for match in matches))
        
        if has_high_risk or (unique_types >= 3 and total_matches >= 5):
            return "CRITICAL"
        elif has_medium_risk or (unique_types >= 2 and total_matches >= 3):
            return "HIGH"
        elif total_matches >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def mask_pii(self, text: str, detection_result: PIIDetectionResult, mask_char: str = "*") -> str:
        """Mask detected PII in text"""
        
        if not detection_result.has_pii:
            return text
        
        # Sort matches by position (reverse order to maintain positions)
        matches = sorted(detection_result.matches, key=lambda x: x.start_pos, reverse=True)
        
        masked_text = text
        for match in matches:
            # Different masking strategies based on PII type
            if match.pii_type in [PIIType.SSN, PIIType.CREDIT_CARD]:
                # Show only last 4 digits
                masked_value = mask_char * (len(match.value) - 4) + match.value[-4:]
            elif match.pii_type == PIIType.EMAIL:
                # Mask username but keep domain
                parts = match.value.split('@')
                if len(parts) == 2:
                    masked_value = mask_char * len(parts[0]) + "@" + parts[1]
                else:
                    masked_value = mask_char * len(match.value)
            else:
                # Full masking for other types
                masked_value = mask_char * len(match.value)
            
            masked_text = (
                masked_text[:match.start_pos] + 
                masked_value + 
                masked_text[match.end_pos:]
            )
        
        return masked_text
    
    def generate_pii_report(self, detection_results: Dict[str, PIIDetectionResult]) -> Dict[str, Any]:
        """Generate a comprehensive PII detection report"""
        
        total_matches = sum(result.total_matches for result in detection_results.values())
        all_pii_types = set()
        risk_levels = []
        
        for result in detection_results.values():
            all_pii_types.update(result.pii_types_found)
            risk_levels.append(result.risk_level)
        
        # Overall risk assessment
        overall_risk = "LOW"
        if "CRITICAL" in risk_levels:
            overall_risk = "CRITICAL"
        elif "HIGH" in risk_levels:
            overall_risk = "HIGH"
        elif "MEDIUM" in risk_levels:
            overall_risk = "MEDIUM"
        
        report = {
            "summary": {
                "total_fields_with_pii": len(detection_results),
                "total_pii_matches": total_matches,
                "unique_pii_types": len(all_pii_types),
                "pii_types_found": [pii_type.value for pii_type in all_pii_types],
                "overall_risk_level": overall_risk
            },
            "field_details": {},
            "recommendations": self._generate_recommendations(all_pii_types, overall_risk)
        }
        
        for field_name, result in detection_results.items():
            report["field_details"][field_name] = {
                "matches_count": result.total_matches,
                "pii_types": [pii_type.value for pii_type in result.pii_types_found],
                "risk_level": result.risk_level,
                "matches": [
                    {
                        "type": match.pii_type.value,
                        "value": match.value,
                        "confidence": match.confidence,
                        "context": match.context
                    }
                    for match in result.matches
                ]
            }
        
        return report
    
    def _generate_recommendations(self, pii_types: Set[PIIType], risk_level: str) -> List[str]:
        """Generate recommendations based on detected PII"""
        
        recommendations = []
        
        if PIIType.SSN in pii_types:
            recommendations.append("Consider masking or removing Social Security Numbers")
        
        if PIIType.CREDIT_CARD in pii_types:
            recommendations.append("Credit card numbers detected - ensure PCI DSS compliance")
        
        if PIIType.MEDICAL_RECORD in pii_types:
            recommendations.append("Medical record numbers found - verify HIPAA compliance")
        
        if PIIType.EMAIL in pii_types:
            recommendations.append("Email addresses detected - consider anonymization")
        
        if risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append("High risk PII detected - immediate review and remediation recommended")
            recommendations.append("Consider implementing additional data masking or anonymization")
        
        if len(pii_types) > 3:
            recommendations.append("Multiple PII types detected - comprehensive privacy review needed")
        
        return recommendations