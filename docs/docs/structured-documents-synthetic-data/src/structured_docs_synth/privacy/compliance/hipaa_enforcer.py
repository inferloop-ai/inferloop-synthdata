"""
HIPAA (Health Insurance Portability and Accountability Act) compliance enforcer
Implements US HIPAA requirements for protected health information (PHI)
"""

import re
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ...core import get_logger, ComplianceError, PrivacyError
from ..pii_protection.pii_detector import PIIDetector, PIIType, PIIDetectionResult


class PHIIdentifier(Enum):
    """HIPAA-defined identifiers that make health information PHI"""
    NAMES = "names"
    GEOGRAPHIC_SUBDIVISIONS = "geographic_subdivisions"  # Smaller than state
    DATES = "dates"  # Except year
    TELEPHONE_NUMBERS = "telephone_numbers"
    FAX_NUMBERS = "fax_numbers"
    EMAIL_ADDRESSES = "email_addresses"
    SSN = "social_security_numbers"
    MEDICAL_RECORD_NUMBERS = "medical_record_numbers"
    HEALTH_PLAN_BENEFICIARY_NUMBERS = "health_plan_beneficiary_numbers"
    ACCOUNT_NUMBERS = "account_numbers"
    CERTIFICATE_LICENSE_NUMBERS = "certificate_license_numbers"
    VEHICLE_IDENTIFIERS = "vehicle_identifiers"
    DEVICE_IDENTIFIERS = "device_identifiers"
    WEB_URLS = "web_urls"
    IP_ADDRESSES = "ip_addresses"
    BIOMETRIC_IDENTIFIERS = "biometric_identifiers"
    FULL_FACE_PHOTOS = "full_face_photos"
    OTHER_UNIQUE_IDENTIFYING_NUMBERS = "other_unique_identifying_numbers"


class HIPAARule(Enum):
    """HIPAA Rules"""
    PRIVACY_RULE = "privacy_rule"
    SECURITY_RULE = "security_rule"
    BREACH_NOTIFICATION_RULE = "breach_notification_rule"
    ENFORCEMENT_RULE = "enforcement_rule"


class SafeHarborMethod(Enum):
    """HIPAA Safe Harbor de-identification methods"""
    REMOVE_IDENTIFIERS = "remove_identifiers"
    NO_ACTUAL_KNOWLEDGE = "no_actual_knowledge"


class LimitedDataSet(Enum):
    """HIPAA Limited Data Set permitted identifiers"""
    DATES = "dates"
    CITY = "city"
    STATE = "state"
    ZIP_CODE = "zip_code"  # First 3 digits only
    AGE = "age"  # Over 89 must be aggregated


@dataclass
class PHIAssessment:
    """Assessment of PHI content in data"""
    contains_phi: bool
    phi_identifiers_found: List[PHIIdentifier]
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    safe_harbor_compliant: bool
    limited_data_set_eligible: bool
    recommendations: List[str]


@dataclass
class DeidentificationResult:
    """Result of HIPAA de-identification process"""
    method_used: SafeHarborMethod
    identifiers_removed: List[PHIIdentifier]
    is_safe_harbor_compliant: bool
    is_limited_data_set: bool
    residual_risk_assessment: str
    compliance_certification: bool


@dataclass
class HIPAAAssessment:
    """Comprehensive HIPAA compliance assessment"""
    is_compliant: bool
    violations: List[str]
    warnings: List[str]
    rules_affected: List[HIPAARule]
    phi_assessment: PHIAssessment
    deidentification_required: bool
    business_associate_agreement_required: bool
    risk_level: str


@dataclass
class BusinessAssociateAgreement:
    """BAA tracking for HIPAA compliance"""
    agreement_id: str
    associate_name: str
    services_provided: List[str]
    phi_access_permitted: bool
    agreement_date: datetime
    expiration_date: Optional[datetime]
    is_active: bool


class HIPAAEnforcer:
    """HIPAA compliance enforcement and validation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pii_detector = PIIDetector()
        
        # HIPAA PHI identifier patterns
        self.phi_patterns = {
            PHIIdentifier.NAMES: [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b'  # Last, First
            ],
            PHIIdentifier.GEOGRAPHIC_SUBDIVISIONS: [
                r'\b\d{5}(-\d{4})?\b',  # ZIP codes
                r'\b[A-Z][a-z]+\s+(County|Parish|Borough)\b',
                r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd)\b'
            ],
            PHIIdentifier.DATES: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ],
            PHIIdentifier.TELEPHONE_NUMBERS: [
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\b\d{10}\b'
            ],
            PHIIdentifier.EMAIL_ADDRESSES: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PHIIdentifier.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b'
            ],
            PHIIdentifier.MEDICAL_RECORD_NUMBERS: [
                r'\bMRN\s*:?\s*\d{6,12}\b',
                r'\bMedical\s+Record\s+(Number|#)\s*:?\s*\d{6,12}\b',
                r'\bPatient\s+ID\s*:?\s*\d{6,12}\b'
            ],
            PHIIdentifier.ACCOUNT_NUMBERS: [
                r'\bAccount\s*(Number|#)\s*:?\s*\d{6,20}\b',
                r'\bPolicy\s*(Number|#)\s*:?\s*[A-Z0-9]{6,20}\b'
            ],
            PHIIdentifier.IP_ADDRESSES: [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ],
            PHIIdentifier.WEB_URLS: [
                r'https?://[^\s]+',
                r'www\.[^\s]+\.[a-z]{2,}'
            ]
        }
        
        # Medical/health-related field indicators
        self.health_indicators = [
            'diagnosis', 'treatment', 'medication', 'prescription', 'therapy',
            'surgery', 'procedure', 'symptom', 'condition', 'disease',
            'medical', 'health', 'patient', 'doctor', 'physician', 'nurse',
            'hospital', 'clinic', 'healthcare', 'provider', 'insurance'
        ]
        
        # Business Associate Agreements
        self.baa_records: Dict[str, BusinessAssociateAgreement] = {}
        
        self.logger.info("HIPAA Enforcer initialized")
    
    def assess_hipaa_compliance(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        is_covered_entity: bool = True,
        has_baa: bool = False,
        intended_use: str = "healthcare_operations"
    ) -> HIPAAAssessment:
        """Comprehensive HIPAA compliance assessment"""
        
        try:
            violations = []
            warnings = []
            rules_affected = []
            
            # Convert to list for uniform processing
            if isinstance(data, dict):
                data = [data]
            
            # 1. Assess if data contains PHI
            phi_assessment = self._assess_phi_content(data)
            
            # 2. Check Privacy Rule compliance
            if phi_assessment.contains_phi:
                privacy_violations = self._check_privacy_rule_compliance(
                    data, is_covered_entity, has_baa, intended_use
                )
                if privacy_violations:
                    violations.extend(privacy_violations)
                    rules_affected.append(HIPAARule.PRIVACY_RULE)
            
            # 3. Check Security Rule compliance
            security_warnings = self._check_security_rule_compliance(phi_assessment.contains_phi)
            if security_warnings:
                warnings.extend(security_warnings)
                rules_affected.append(HIPAARule.SECURITY_RULE)
            
            # 4. Check if de-identification is required
            deidentification_required = (
                phi_assessment.contains_phi and 
                not phi_assessment.safe_harbor_compliant and
                intended_use in ["research", "analytics", "testing"]
            )
            
            # 5. Check BAA requirements
            baa_required = (
                phi_assessment.contains_phi and 
                not is_covered_entity and 
                not has_baa
            )
            
            if baa_required:
                violations.append("Business Associate Agreement required for PHI access")
            
            # 6. Overall risk assessment
            risk_level = self._assess_overall_risk(
                phi_assessment, violations, warnings, intended_use
            )
            
            assessment = HIPAAAssessment(
                is_compliant=len(violations) == 0,
                violations=violations,
                warnings=warnings,
                rules_affected=rules_affected,
                phi_assessment=phi_assessment,
                deidentification_required=deidentification_required,
                business_associate_agreement_required=baa_required,
                risk_level=risk_level
            )
            
            self.logger.info(
                f"HIPAA assessment complete: Compliant={assessment.is_compliant}, "
                f"PHI={phi_assessment.contains_phi}, Risk={risk_level}"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error during HIPAA assessment: {str(e)}")
            raise ComplianceError(
                f"HIPAA compliance assessment failed: {str(e)}",
                details={"intended_use": intended_use}
            )
    
    def _assess_phi_content(self, data: List[Dict[str, Any]]) -> PHIAssessment:
        """Assess if data contains Protected Health Information"""
        
        phi_identifiers_found = []
        contains_health_info = False
        
        # Check for health information context
        contains_health_info = self._detect_health_context(data)
        
        # Check for HIPAA identifiers
        for record in data:
            for field_name, value in record.items():
                if isinstance(value, str):
                    field_identifiers = self._detect_phi_identifiers(field_name, value)
                    phi_identifiers_found.extend(field_identifiers)
        
        # Remove duplicates
        phi_identifiers_found = list(set(phi_identifiers_found))
        
        # Determine if this constitutes PHI
        contains_phi = contains_health_info and len(phi_identifiers_found) > 0
        
        # Check Safe Harbor compliance
        safe_harbor_compliant = self._check_safe_harbor_compliance(phi_identifiers_found)
        
        # Check Limited Data Set eligibility
        limited_data_set_eligible = self._check_limited_data_set_eligibility(phi_identifiers_found)
        
        # Generate recommendations
        recommendations = self._generate_phi_recommendations(
            contains_phi, phi_identifiers_found, safe_harbor_compliant
        )
        
        # Assess risk level
        risk_level = self._assess_phi_risk(contains_phi, phi_identifiers_found)
        
        return PHIAssessment(
            contains_phi=contains_phi,
            phi_identifiers_found=phi_identifiers_found,
            risk_level=risk_level,
            safe_harbor_compliant=safe_harbor_compliant,
            limited_data_set_eligible=limited_data_set_eligible,
            recommendations=recommendations
        )
    
    def _detect_health_context(self, data: List[Dict[str, Any]]) -> bool:
        """Detect if data is in a health/medical context"""
        
        for record in data:
            for field_name, value in record.items():
                # Check field names
                field_text = field_name.lower()
                if any(indicator in field_text for indicator in self.health_indicators):
                    return True
                
                # Check field values
                if isinstance(value, str):
                    value_text = value.lower()
                    if any(indicator in value_text for indicator in self.health_indicators):
                        return True
        
        return False
    
    def _detect_phi_identifiers(self, field_name: str, value: str) -> List[PHIIdentifier]:
        """Detect HIPAA PHI identifiers in a field"""
        
        identifiers_found = []
        
        for identifier, patterns in self.phi_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    identifiers_found.append(identifier)
                    break
        
        # Special cases based on field names
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ['name', 'patient', 'doctor', 'physician']):
            identifiers_found.append(PHIIdentifier.NAMES)
        
        if any(term in field_lower for term in ['zip', 'postal', 'address']):
            identifiers_found.append(PHIIdentifier.GEOGRAPHIC_SUBDIVISIONS)
        
        if any(term in field_lower for term in ['date', 'birth', 'admission', 'discharge']):
            # Check if it's more specific than just year
            if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', value):
                identifiers_found.append(PHIIdentifier.DATES)
        
        return list(set(identifiers_found))
    
    def _check_safe_harbor_compliance(self, identifiers_found: List[PHIIdentifier]) -> bool:
        """Check if data meets HIPAA Safe Harbor requirements"""
        
        # Safe Harbor requires removal of all 18 identifier types
        prohibited_identifiers = [
            PHIIdentifier.NAMES,
            PHIIdentifier.GEOGRAPHIC_SUBDIVISIONS,
            PHIIdentifier.DATES,  # Except year
            PHIIdentifier.TELEPHONE_NUMBERS,
            PHIIdentifier.FAX_NUMBERS,
            PHIIdentifier.EMAIL_ADDRESSES,
            PHIIdentifier.SSN,
            PHIIdentifier.MEDICAL_RECORD_NUMBERS,
            PHIIdentifier.HEALTH_PLAN_BENEFICIARY_NUMBERS,
            PHIIdentifier.ACCOUNT_NUMBERS,
            PHIIdentifier.CERTIFICATE_LICENSE_NUMBERS,
            PHIIdentifier.VEHICLE_IDENTIFIERS,
            PHIIdentifier.DEVICE_IDENTIFIERS,
            PHIIdentifier.WEB_URLS,
            PHIIdentifier.IP_ADDRESSES,
            PHIIdentifier.BIOMETRIC_IDENTIFIERS,
            PHIIdentifier.FULL_FACE_PHOTOS,
            PHIIdentifier.OTHER_UNIQUE_IDENTIFYING_NUMBERS
        ]
        
        # Check if any prohibited identifiers are present
        for identifier in identifiers_found:
            if identifier in prohibited_identifiers:
                return False
        
        return True
    
    def _check_limited_data_set_eligibility(self, identifiers_found: List[PHIIdentifier]) -> bool:
        """Check if data can be treated as a Limited Data Set"""
        
        # Limited Data Set prohibits certain identifiers
        prohibited_for_lds = [
            PHIIdentifier.NAMES,
            PHIIdentifier.TELEPHONE_NUMBERS,
            PHIIdentifier.FAX_NUMBERS,
            PHIIdentifier.EMAIL_ADDRESSES,
            PHIIdentifier.SSN,
            PHIIdentifier.MEDICAL_RECORD_NUMBERS,
            PHIIdentifier.ACCOUNT_NUMBERS,
            PHIIdentifier.IP_ADDRESSES,
            PHIIdentifier.WEB_URLS,
            PHIIdentifier.BIOMETRIC_IDENTIFIERS,
            PHIIdentifier.FULL_FACE_PHOTOS
        ]
        
        for identifier in identifiers_found:
            if identifier in prohibited_for_lds:
                return False
        
        return True
    
    def _check_privacy_rule_compliance(
        self,
        data: List[Dict[str, Any]],
        is_covered_entity: bool,
        has_baa: bool,
        intended_use: str
    ) -> List[str]:
        """Check HIPAA Privacy Rule compliance"""
        
        violations = []
        
        # Covered entities must follow Privacy Rule
        if is_covered_entity:
            # Check minimum necessary standard
            if intended_use not in ["treatment", "payment", "healthcare_operations"]:
                violations.append(
                    "Non-TPO uses require minimum necessary determination"
                )
        
        # Business associates need BAA
        if not is_covered_entity and not has_baa:
            violations.append(
                "Business Associate Agreement required for PHI access"
            )
        
        # Check for patient authorization if required
        if intended_use in ["research", "marketing", "psychotherapy_notes"]:
            violations.append(
                "Patient authorization may be required for this use"
            )
        
        return violations
    
    def _check_security_rule_compliance(self, contains_phi: bool) -> List[str]:
        """Check HIPAA Security Rule compliance"""
        
        warnings = []
        
        if contains_phi:
            warnings.extend([
                "Implement access controls for PHI systems",
                "Ensure audit controls are in place",
                "Verify integrity controls for PHI",
                "Implement person or entity authentication",
                "Ensure transmission security for PHI"
            ])
        
        return warnings
    
    def _generate_phi_recommendations(
        self,
        contains_phi: bool,
        identifiers_found: List[PHIIdentifier],
        safe_harbor_compliant: bool
    ) -> List[str]:
        """Generate recommendations for PHI handling"""
        
        recommendations = []
        
        if contains_phi:
            recommendations.append("Data contains PHI - HIPAA compliance required")
            
            if not safe_harbor_compliant:
                recommendations.append("Remove or mask PHI identifiers for Safe Harbor compliance")
                
                # Specific recommendations based on identifiers found
                if PHIIdentifier.NAMES in identifiers_found:
                    recommendations.append("Remove or pseudonymize all names")
                
                if PHIIdentifier.DATES in identifiers_found:
                    recommendations.append("Remove dates or retain year only")
                
                if PHIIdentifier.SSN in identifiers_found:
                    recommendations.append("Remove Social Security Numbers")
        else:
            recommendations.append("No PHI detected - standard data protection practices apply")
        
        return recommendations
    
    def _assess_phi_risk(self, contains_phi: bool, identifiers_found: List[PHIIdentifier]) -> str:
        """Assess risk level for PHI data"""
        
        if not contains_phi:
            return "LOW"
        
        high_risk_identifiers = [
            PHIIdentifier.SSN,
            PHIIdentifier.MEDICAL_RECORD_NUMBERS,
            PHIIdentifier.BIOMETRIC_IDENTIFIERS
        ]
        
        # Check for high-risk identifiers
        high_risk_found = any(identifier in high_risk_identifiers for identifier in identifiers_found)
        
        if high_risk_found or len(identifiers_found) >= 3:
            return "CRITICAL"
        elif len(identifiers_found) >= 2:
            return "HIGH"
        elif len(identifiers_found) > 0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_overall_risk(
        self,
        phi_assessment: PHIAssessment,
        violations: List[str],
        warnings: List[str],
        intended_use: str
    ) -> str:
        """Assess overall HIPAA compliance risk"""
        
        if len(violations) >= 2:
            return "CRITICAL"
        elif phi_assessment.risk_level == "CRITICAL" and violations:
            return "CRITICAL"
        elif len(violations) >= 1:
            return "HIGH"
        elif phi_assessment.risk_level in ["HIGH", "CRITICAL"]:
            return "HIGH"
        else:
            return "MEDIUM"
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive HIPAA compliance report"""
        
        return {
            "total_baa_agreements": len(self.baa_records),
            "phi_identifiers_monitored": [identifier.value for identifier in PHIIdentifier],
            "compliance_rules": [rule.value for rule in HIPAARule],
            "health_indicators": self.health_indicators
        }