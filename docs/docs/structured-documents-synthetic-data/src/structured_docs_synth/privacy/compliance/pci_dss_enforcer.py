"""
PCI DSS (Payment Card Industry Data Security Standard) compliance enforcer
Implements PCI DSS requirements for cardholder data protection
"""

import re
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ...core import get_logger, ComplianceError, PrivacyError
from ..pii_protection.pii_detector import PIIDetector, PIIType, PIIDetectionResult


class CardholderDataElement(Enum):
    """PCI DSS-defined cardholder data elements"""
    PRIMARY_ACCOUNT_NUMBER = "primary_account_number"  # PAN
    CARDHOLDER_NAME = "cardholder_name"
    EXPIRATION_DATE = "expiration_date"
    SERVICE_CODE = "service_code"


class SensitiveAuthenticationData(Enum):
    """PCI DSS-defined sensitive authentication data (must never be stored)"""
    FULL_TRACK_DATA = "full_track_data"
    CAV2_CVC2_CVV2_CID = "cav2_cvc2_cvv2_cid"  # Card verification codes
    PIN_BLOCK = "pin_block"


class PCIDSSRequirement(Enum):
    """PCI DSS Requirements"""
    FIREWALL_CONFIGURATION = "1_firewall_configuration"
    DEFAULT_PASSWORDS = "2_default_passwords"
    STORED_CARDHOLDER_DATA = "3_stored_cardholder_data"
    ENCRYPTED_TRANSMISSION = "4_encrypted_transmission"
    ANTIVIRUS_SOFTWARE = "5_antivirus_software"
    SECURE_SYSTEMS = "6_secure_systems"
    ACCESS_CONTROL = "7_access_control"
    UNIQUE_IDS = "8_unique_ids"
    PHYSICAL_ACCESS = "9_physical_access"
    NETWORK_MONITORING = "10_network_monitoring"
    SECURITY_TESTING = "11_security_testing"
    INFORMATION_SECURITY = "12_information_security"


class DataClassification(Enum):
    """PCI DSS data classification levels"""
    CARDHOLDER_DATA = "cardholder_data"
    SENSITIVE_AUTHENTICATION_DATA = "sensitive_authentication_data"
    PUBLIC_DATA = "public_data"
    INTERNAL_DATA = "internal_data"


class ComplianceLevel(Enum):
    """PCI DSS compliance levels based on transaction volume"""
    LEVEL_1 = "level_1"  # 6M+ transactions/year
    LEVEL_2 = "level_2"  # 1M-6M transactions/year
    LEVEL_3 = "level_3"  # 20K-1M transactions/year
    LEVEL_4 = "level_4"  # <20K transactions/year


@dataclass
class CardholderDataAssessment:
    """Assessment of cardholder data in the dataset"""
    contains_cardholder_data: bool
    data_elements_found: List[CardholderDataElement]
    sensitive_auth_data_found: List[SensitiveAuthenticationData]
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    storage_permitted: bool
    requires_encryption: bool
    recommendations: List[str]


@dataclass
class PCIDSSAssessment:
    """Comprehensive PCI DSS compliance assessment"""
    is_compliant: bool
    violations: List[str]
    warnings: List[str]
    requirements_affected: List[PCIDSSRequirement]
    cardholder_data_assessment: CardholderDataAssessment
    data_classification: DataClassification
    compliance_level_required: ComplianceLevel
    risk_level: str


@dataclass
class TokenizationResult:
    """Result of PCI DSS tokenization process"""
    original_pan_count: int
    tokens_generated: int
    tokenization_method: str
    token_format: str
    is_reversible: bool
    security_level: str


class PCIDSSEnforcer:
    """PCI DSS compliance enforcement and validation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pii_detector = PIIDetector()
        
        # PCI DSS cardholder data patterns
        self.cardholder_data_patterns = {
            CardholderDataElement.PRIMARY_ACCOUNT_NUMBER: [
                r'\b4[0-9]{12}(?:[0-9]{3})?\b',  # Visa
                r'\b5[1-5][0-9]{14}\b',  # MasterCard
                r'\b3[47][0-9]{13}\b',  # American Express
                r'\b3[0-9]{13}\b',  # Diners Club
                r'\b6(?:011|5[0-9]{2})[0-9]{12}\b',  # Discover
                r'\b(?:2131|1800|35\d{3})\d{11}\b'  # JCB
            ],
            CardholderDataElement.CARDHOLDER_NAME: [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?=.*(?:card|visa|master|amex))',
                r'(?:card\s*holder|cardholder):\s*[A-Z][a-z]+\s+[A-Z][a-z]+',
                r'name\s*on\s*card:\s*[A-Z][a-z]+\s+[A-Z][a-z]+'
            ],
            CardholderDataElement.EXPIRATION_DATE: [
                r'\b(0[1-9]|1[0-2])\/([0-9]{2})\b',  # MM/YY
                r'\b(0[1-9]|1[0-2])\/([0-9]{4})\b',  # MM/YYYY
                r'\bexp(?:iry)?:?\s*(0[1-9]|1[0-2])\/([0-9]{2,4})\b'
            ],
            CardholderDataElement.SERVICE_CODE: [
                r'\bservice\s*code:?\s*[0-9]{3}\b',
                r'\b[0-9]{3}(?=.*service)'
            ]
        }
        
        # Sensitive authentication data patterns (should NEVER be stored)
        self.sensitive_auth_patterns = {
            SensitiveAuthenticationData.CAV2_CVC2_CVV2_CID: [
                r'\bcvv2?:?\s*[0-9]{3,4}\b',
                r'\bcvc2?:?\s*[0-9]{3,4}\b',
                r'\bcid:?\s*[0-9]{4}\b',
                r'\bsecurity\s*code:?\s*[0-9]{3,4}\b'
            ],
            SensitiveAuthenticationData.FULL_TRACK_DATA: [
                r'%[A-Z]?[0-9]{13,19}\^[A-Z\s\/]+\^[0-9]{4}[0-9]*\?',  # Track 1
                r';[0-9]{13,19}=[0-9]{4}[0-9]*\?'  # Track 2
            ],
            SensitiveAuthenticationData.PIN_BLOCK: [
                r'\bpin\s*block:?\s*[0-9A-F]{16}\b',
                r'\bpin:?\s*[0-9]{4,6}\b'
            ]
        }
        
        # Payment-related field indicators
        self.payment_indicators = [
            'card', 'payment', 'transaction', 'purchase', 'billing',
            'credit', 'debit', 'visa', 'mastercard', 'amex', 'discover',
            'pan', 'cardholder', 'merchant', 'acquirer', 'processor'
        ]
        
        self.logger.info("PCI DSS Enforcer initialized")
    
    def assess_pci_dss_compliance(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        transaction_volume_annual: int = 0,
        stores_cardholder_data: bool = True,
        processes_cards: bool = True,
        environment_type: str = "production"
    ) -> PCIDSSAssessment:
        """Comprehensive PCI DSS compliance assessment"""
        
        try:
            violations = []
            warnings = []
            requirements_affected = []
            
            # Convert to list for uniform processing
            if isinstance(data, dict):
                data = [data]
            
            # 1. Assess cardholder data content
            cardholder_assessment = self._assess_cardholder_data_content(data)
            
            # 2. Determine compliance level based on transaction volume
            compliance_level = self._determine_compliance_level(transaction_volume_annual)
            
            # 3. Check data storage requirements (Requirement 3)
            if cardholder_assessment.contains_cardholder_data:
                storage_violations = self._check_data_storage_requirements(
                    data, stores_cardholder_data, cardholder_assessment
                )
                if storage_violations:
                    violations.extend(storage_violations)
                    requirements_affected.append(PCIDSSRequirement.STORED_CARDHOLDER_DATA)
            
            # 4. Check for prohibited sensitive authentication data
            auth_data_violations = self._check_sensitive_auth_data(cardholder_assessment)
            if auth_data_violations:
                violations.extend(auth_data_violations)
                requirements_affected.append(PCIDSSRequirement.STORED_CARDHOLDER_DATA)
            
            # 5. Check encryption requirements (Requirement 4)
            if cardholder_assessment.contains_cardholder_data:
                encryption_warnings = self._check_encryption_requirements(environment_type)
                if encryption_warnings:
                    warnings.extend(encryption_warnings)
                    requirements_affected.append(PCIDSSRequirement.ENCRYPTED_TRANSMISSION)
            
            # 6. Check access control requirements (Requirement 7)
            if processes_cards:
                access_warnings = self._check_access_control_requirements(cardholder_assessment)
                if access_warnings:
                    warnings.extend(access_warnings)
                    requirements_affected.append(PCIDSSRequirement.ACCESS_CONTROL)
            
            # 7. Determine data classification
            data_classification = self._classify_data(cardholder_assessment)
            
            # 8. Overall risk assessment
            risk_level = self._assess_overall_pci_risk(
                cardholder_assessment, violations, warnings, compliance_level
            )
            
            assessment = PCIDSSAssessment(
                is_compliant=len(violations) == 0,
                violations=violations,
                warnings=warnings,
                requirements_affected=requirements_affected,
                cardholder_data_assessment=cardholder_assessment,
                data_classification=data_classification,
                compliance_level_required=compliance_level,
                risk_level=risk_level
            )
            
            self.logger.info(
                f"PCI DSS assessment complete: Compliant={assessment.is_compliant}, "
                f"CHD={cardholder_assessment.contains_cardholder_data}, "
                f"Level={compliance_level.value}, Risk={risk_level}"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error during PCI DSS assessment: {str(e)}")
            raise ComplianceError(
                f"PCI DSS compliance assessment failed: {str(e)}",
                details={"transaction_volume": transaction_volume_annual}
            )
    
    def _assess_cardholder_data_content(
        self, 
        data: List[Dict[str, Any]]
    ) -> CardholderDataAssessment:
        """Assess if data contains cardholder data"""
        
        data_elements_found = []
        sensitive_auth_found = []
        contains_payment_context = False
        
        # Check for payment context
        contains_payment_context = self._detect_payment_context(data)
        
        # Check for cardholder data elements
        for record in data:
            for field_name, value in record.items():
                if isinstance(value, str):
                    # Check cardholder data
                    chd_elements = self._detect_cardholder_data_elements(field_name, value)
                    data_elements_found.extend(chd_elements)
                    
                    # Check sensitive authentication data
                    sad_elements = self._detect_sensitive_auth_data(field_name, value)
                    sensitive_auth_found.extend(sad_elements)
        
        # Remove duplicates
        data_elements_found = list(set(data_elements_found))
        sensitive_auth_found = list(set(sensitive_auth_found))
        
        # Determine if this constitutes cardholder data
        contains_cardholder_data = (
            contains_payment_context and 
            len(data_elements_found) > 0
        )
        
        # Assess storage permissions
        storage_permitted = self._assess_storage_permissions(
            data_elements_found, sensitive_auth_found
        )
        
        # Check encryption requirements
        requires_encryption = contains_cardholder_data and storage_permitted
        
        # Generate recommendations
        recommendations = self._generate_cardholder_data_recommendations(
            contains_cardholder_data, data_elements_found, sensitive_auth_found
        )
        
        # Assess risk level
        risk_level = self._assess_cardholder_data_risk(
            contains_cardholder_data, data_elements_found, sensitive_auth_found
        )
        
        return CardholderDataAssessment(
            contains_cardholder_data=contains_cardholder_data,
            data_elements_found=data_elements_found,
            sensitive_auth_data_found=sensitive_auth_found,
            risk_level=risk_level,
            storage_permitted=storage_permitted,
            requires_encryption=requires_encryption,
            recommendations=recommendations
        )
    
    def _detect_payment_context(self, data: List[Dict[str, Any]]) -> bool:
        """Detect if data is in a payment/card processing context"""
        
        for record in data:
            for field_name, value in record.items():
                # Check field names
                field_text = field_name.lower()
                if any(indicator in field_text for indicator in self.payment_indicators):
                    return True
                
                # Check field values
                if isinstance(value, str):
                    value_text = value.lower()
                    if any(indicator in value_text for indicator in self.payment_indicators):
                        return True
        
        return False
    
    def _detect_cardholder_data_elements(
        self, 
        field_name: str, 
        value: str
    ) -> List[CardholderDataElement]:
        """Detect PCI DSS cardholder data elements"""
        
        elements_found = []
        
        for element, patterns in self.cardholder_data_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    elements_found.append(element)
                    break
        
        # Special cases based on field names
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ['pan', 'card_number', 'account_number']):
            # Check if it looks like a PAN
            if re.search(r'\b[0-9]{13,19}\b', value):
                elements_found.append(CardholderDataElement.PRIMARY_ACCOUNT_NUMBER)
        
        if any(term in field_lower for term in ['cardholder', 'card_holder', 'name_on_card']):
            elements_found.append(CardholderDataElement.CARDHOLDER_NAME)
        
        if any(term in field_lower for term in ['expiry', 'expiration', 'exp_date']):
            elements_found.append(CardholderDataElement.EXPIRATION_DATE)
        
        return list(set(elements_found))
    
    def _detect_sensitive_auth_data(
        self, 
        field_name: str, 
        value: str
    ) -> List[SensitiveAuthenticationData]:
        """Detect sensitive authentication data (should never be stored)"""
        
        auth_data_found = []
        
        for auth_data, patterns in self.sensitive_auth_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    auth_data_found.append(auth_data)
                    break
        
        # Field name checks
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ['cvv', 'cvc', 'security_code', 'cid']):
            auth_data_found.append(SensitiveAuthenticationData.CAV2_CVC2_CVV2_CID)
        
        if any(term in field_lower for term in ['track', 'magnetic']):
            auth_data_found.append(SensitiveAuthenticationData.FULL_TRACK_DATA)
        
        if 'pin' in field_lower:
            auth_data_found.append(SensitiveAuthenticationData.PIN_BLOCK)
        
        return list(set(auth_data_found))
    
    def _determine_compliance_level(self, transaction_volume: int) -> ComplianceLevel:
        """Determine PCI DSS compliance level based on transaction volume"""
        
        if transaction_volume >= 6_000_000:
            return ComplianceLevel.LEVEL_1
        elif transaction_volume >= 1_000_000:
            return ComplianceLevel.LEVEL_2
        elif transaction_volume >= 20_000:
            return ComplianceLevel.LEVEL_3
        else:
            return ComplianceLevel.LEVEL_4
    
    def _check_data_storage_requirements(
        self,
        data: List[Dict[str, Any]],
        stores_cardholder_data: bool,
        assessment: CardholderDataAssessment
    ) -> List[str]:
        """Check PCI DSS data storage requirements"""
        
        violations = []
        
        if assessment.contains_cardholder_data and stores_cardholder_data:
            # Check if business justification exists
            if CardholderDataElement.PRIMARY_ACCOUNT_NUMBER in assessment.data_elements_found:
                violations.append(
                    "PAN storage requires business justification and annual review"
                )
            
            # Check retention period
            violations.append(
                "Stored cardholder data must have defined retention period"
            )
            
            # Check if PAN is masked when displayed
            violations.append(
                "PAN must be masked when displayed (show only first 6 and last 4 digits)"
            )
        
        return violations
    
    def _check_sensitive_auth_data(
        self, 
        assessment: CardholderDataAssessment
    ) -> List[str]:
        """Check for prohibited sensitive authentication data"""
        
        violations = []
        
        if assessment.sensitive_auth_data_found:
            for sad_element in assessment.sensitive_auth_data_found:
                if sad_element == SensitiveAuthenticationData.CAV2_CVC2_CVV2_CID:
                    violations.append(
                        "CVV2/CVC2/CID must NEVER be stored after authorization"
                    )
                elif sad_element == SensitiveAuthenticationData.FULL_TRACK_DATA:
                    violations.append(
                        "Full track data must NEVER be stored after authorization"
                    )
                elif sad_element == SensitiveAuthenticationData.PIN_BLOCK:
                    violations.append(
                        "PIN/PIN blocks must NEVER be stored"
                    )
        
        return violations
    
    def _check_encryption_requirements(self, environment_type: str) -> List[str]:
        """Check encryption requirements for cardholder data"""
        
        warnings = []
        
        if environment_type == "production":
            warnings.extend([
                "Cardholder data must be encrypted during transmission over open networks",
                "Stored cardholder data must be encrypted using strong cryptography",
                "Encryption keys must be managed securely"
            ])
        
        return warnings
    
    def _check_access_control_requirements(
        self, 
        assessment: CardholderDataAssessment
    ) -> List[str]:
        """Check access control requirements"""
        
        warnings = []
        
        if assessment.contains_cardholder_data:
            warnings.extend([
                "Implement role-based access control for cardholder data",
                "Restrict access to cardholder data on business need-to-know",
                "Assign unique IDs to each person with computer access",
                "Implement two-factor authentication for remote access"
            ])
        
        return warnings
    
    def _assess_storage_permissions(
        self,
        data_elements: List[CardholderDataElement],
        sensitive_auth: List[SensitiveAuthenticationData]
    ) -> bool:
        """Assess if data storage is permitted under PCI DSS"""
        
        # Sensitive authentication data must NEVER be stored
        if sensitive_auth:
            return False
        
        # Cardholder data may be stored with proper protections
        return True
    
    def _classify_data(self, assessment: CardholderDataAssessment) -> DataClassification:
        """Classify data according to PCI DSS standards"""
        
        if assessment.sensitive_auth_data_found:
            return DataClassification.SENSITIVE_AUTHENTICATION_DATA
        elif assessment.contains_cardholder_data:
            return DataClassification.CARDHOLDER_DATA
        else:
            return DataClassification.PUBLIC_DATA
    
    def _generate_cardholder_data_recommendations(
        self,
        contains_cardholder_data: bool,
        data_elements: List[CardholderDataElement],
        sensitive_auth: List[SensitiveAuthenticationData]
    ) -> List[str]:
        """Generate recommendations for cardholder data handling"""
        
        recommendations = []
        
        if sensitive_auth:
            recommendations.append("CRITICAL: Remove all sensitive authentication data immediately")
        
        if contains_cardholder_data:
            recommendations.append("Implement PCI DSS controls for cardholder data protection")
            
            if CardholderDataElement.PRIMARY_ACCOUNT_NUMBER in data_elements:
                recommendations.extend([
                    "Mask PAN when displayed (show only first 6 and last 4 digits)",
                    "Encrypt stored PAN using strong cryptography"
                ])
            
            if CardholderDataElement.CARDHOLDER_NAME in data_elements:
                recommendations.append("Protect cardholder name with appropriate access controls")
        else:
            recommendations.append("No cardholder data detected - standard security practices apply")
        
        return recommendations
    
    def _assess_cardholder_data_risk(
        self,
        contains_cardholder_data: bool,
        data_elements: List[CardholderDataElement],
        sensitive_auth: List[SensitiveAuthenticationData]
    ) -> str:
        """Assess risk level for cardholder data"""
        
        if sensitive_auth:
            return "CRITICAL"
        
        if not contains_cardholder_data:
            return "LOW"
        
        if CardholderDataElement.PRIMARY_ACCOUNT_NUMBER in data_elements:
            return "HIGH"
        elif len(data_elements) >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_overall_pci_risk(
        self,
        cardholder_assessment: CardholderDataAssessment,
        violations: List[str],
        warnings: List[str],
        compliance_level: ComplianceLevel
    ) -> str:
        """Assess overall PCI DSS compliance risk"""
        
        if cardholder_assessment.sensitive_auth_data_found:
            return "CRITICAL"
        
        if len(violations) >= 3:
            return "CRITICAL"
        elif len(violations) >= 1 and cardholder_assessment.risk_level == "HIGH":
            return "HIGH"
        elif len(violations) >= 1:
            return "MEDIUM"
        elif cardholder_assessment.contains_cardholder_data:
            return "MEDIUM"
        else:
            return "LOW"
    
    def tokenize_cardholder_data(
        self,
        data: Dict[str, Any],
        tokenization_method: str = "format_preserving"
    ) -> TokenizationResult:
        """Tokenize cardholder data for PCI DSS compliance"""
        
        # This would implement actual tokenization
        # For now, return a mock result
        return TokenizationResult(
            original_pan_count=1,
            tokens_generated=1,
            tokenization_method=tokenization_method,
            token_format="4111-1111-1111-****",
            is_reversible=True,
            security_level="HIGH"
        )
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive PCI DSS compliance report"""
        
        return {
            "cardholder_data_elements_monitored": [element.value for element in CardholderDataElement],
            "sensitive_auth_data_prohibited": [sad.value for sad in SensitiveAuthenticationData],
            "pci_dss_requirements": [req.value for req in PCIDSSRequirement],
            "compliance_levels": [level.value for level in ComplianceLevel],
            "payment_indicators": self.payment_indicators
        }