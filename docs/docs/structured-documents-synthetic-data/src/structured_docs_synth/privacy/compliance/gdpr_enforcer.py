"""
GDPR (General Data Protection Regulation) compliance enforcer
Implements EU GDPR requirements for data protection and privacy
"""

import re
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ...core import get_logger, ComplianceError, PrivacyError
from ..pii_protection.pii_detector import PIIDetector, PIIType, PIIDetectionResult


class GDPRArticle(Enum):
    """GDPR Articles relevant to data processing"""
    ARTICLE_5 = "article_5"  # Principles relating to processing of personal data
    ARTICLE_6 = "article_6"  # Lawfulness of processing
    ARTICLE_7 = "article_7"  # Conditions for consent
    ARTICLE_9 = "article_9"  # Processing of special categories of personal data
    ARTICLE_12 = "article_12"  # Transparent information, communication
    ARTICLE_13 = "article_13"  # Information to be provided where personal data are collected
    ARTICLE_15 = "article_15"  # Right of access by the data subject
    ARTICLE_16 = "article_16"  # Right to rectification
    ARTICLE_17 = "article_17"  # Right to erasure ('right to be forgotten')
    ARTICLE_18 = "article_18"  # Right to restriction of processing
    ARTICLE_20 = "article_20"  # Right to data portability
    ARTICLE_25 = "article_25"  # Data protection by design and by default
    ARTICLE_32 = "article_32"  # Security of processing
    ARTICLE_44 = "article_44"  # General principle for transfers


class LawfulBasis(Enum):
    """GDPR Article 6 lawful bases for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataCategory(Enum):
    """Categories of personal data under GDPR"""
    PERSONAL_DATA = "personal_data"
    SPECIAL_CATEGORY = "special_category"  # Article 9 data
    CRIMINAL_DATA = "criminal_data"
    PSEUDONYMIZED = "pseudonymized"
    ANONYMOUS = "anonymous"


class ProcessingPurpose(Enum):
    """Common processing purposes under GDPR"""
    RESEARCH = "research"
    STATISTICS = "statistics"
    TESTING = "testing"
    TRAINING = "training"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    COMPLIANCE = "compliance"
    SECURITY = "security"


@dataclass
class DataSubjectRights:
    """GDPR rights of data subjects"""
    access: bool = True  # Article 15
    rectification: bool = True  # Article 16
    erasure: bool = True  # Article 17
    restriction: bool = True  # Article 18
    portability: bool = True  # Article 20
    objection: bool = True  # Article 21
    automated_decision_making: bool = True  # Article 22


@dataclass
class ConsentRecord:
    """Record of consent given by data subject"""
    subject_id: str
    consent_given: bool
    consent_date: datetime
    consent_purpose: List[ProcessingPurpose]
    consent_method: str  # How consent was obtained
    withdrawable: bool = True
    withdrawn_date: Optional[datetime] = None


@dataclass
class ProcessingRecord:
    """Record of data processing activity (Article 30)"""
    processing_id: str
    controller_name: str
    purpose: List[ProcessingPurpose]
    lawful_basis: LawfulBasis
    data_categories: List[DataCategory]
    recipients: List[str]
    third_country_transfers: List[str]
    retention_period: Optional[int]  # Days
    security_measures: List[str]
    timestamp: datetime


@dataclass
class GDPRAssessment:
    """Assessment of GDPR compliance"""
    is_compliant: bool
    violations: List[str]
    warnings: List[str]
    required_actions: List[str]
    articles_affected: List[GDPRArticle]
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    compliance_score: float  # 0-1


@dataclass
class DataProtectionImpactAssessment:
    """DPIA as required by Article 35"""
    assessment_id: str
    processing_description: str
    necessity_assessment: str
    proportionality_assessment: str
    risks_identified: List[str]
    mitigation_measures: List[str]
    residual_risks: List[str]
    decision: str  # PROCEED, MODIFY, STOP
    consultation_required: bool
    timestamp: datetime


class GDPREnforcer:
    """GDPR compliance enforcement and validation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pii_detector = PIIDetector()
        
        # GDPR-specific PII categories
        self.special_category_patterns = {
            "racial_ethnic": [r"\b(race|ethnic|racial|ethnicity)\b"],
            "political": [r"\b(political|party|vote|election)\b"],
            "religious": [r"\b(religious|religion|faith|belief)\b"],
            "philosophical": [r"\b(philosophical|philosophy|belief)\b"],
            "trade_union": [r"\b(union|trade union|membership)\b"],
            "genetic": [r"\b(genetic|dna|gene|genome)\b"],
            "biometric": [r"\b(biometric|fingerprint|facial|iris)\b"],
            "health": [r"\b(health|medical|diagnosis|treatment|medication)\b"],
            "sex_life": [r"\b(sexual|sexuality|orientation|intimate)\b"]
        }
        
        # Data retention limits by category
        self.retention_limits = {
            DataCategory.PERSONAL_DATA: 365 * 7,  # 7 years default
            DataCategory.SPECIAL_CATEGORY: 365 * 3,  # 3 years for sensitive data
            DataCategory.CRIMINAL_DATA: 365 * 10,  # 10 years for criminal data
            DataCategory.PSEUDONYMIZED: 365 * 10,  # Longer for pseudonymized
            DataCategory.ANONYMOUS: None  # No limit for truly anonymous data
        }
        
        # Consent records
        self.consent_records: Dict[str, ConsentRecord] = {}
        
        # Processing records (Article 30)
        self.processing_records: List[ProcessingRecord] = []
        
        self.logger.info("GDPR Enforcer initialized")
    
    def assess_gdpr_compliance(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        processing_purpose: ProcessingPurpose,
        lawful_basis: LawfulBasis,
        data_subject_id: Optional[str] = None,
        cross_border_transfer: bool = False
    ) -> GDPRAssessment:
        """Comprehensive GDPR compliance assessment"""
        
        try:
            violations = []
            warnings = []
            required_actions = []
            articles_affected = []
            
            # Convert to list for uniform processing
            if isinstance(data, dict):
                data = [data]
            
            # 1. Detect personal data (Article 4)
            personal_data_detected = self._detect_personal_data(data)
            
            # 2. Classify data categories
            data_categories = self._classify_data_categories(data, personal_data_detected)
            
            # 3. Check lawful basis (Article 6)
            lawful_basis_valid, basis_violations = self._validate_lawful_basis(
                lawful_basis, processing_purpose, data_categories, data_subject_id
            )
            if not lawful_basis_valid:
                violations.extend(basis_violations)
                articles_affected.append(GDPRArticle.ARTICLE_6)
            
            # 4. Check special category processing (Article 9)
            if DataCategory.SPECIAL_CATEGORY in data_categories:
                special_valid, special_violations = self._validate_special_category_processing(
                    lawful_basis, processing_purpose
                )
                if not special_valid:
                    violations.extend(special_violations)
                    articles_affected.append(GDPRArticle.ARTICLE_9)
            
            # 5. Check consent requirements (Article 7)
            if lawful_basis == LawfulBasis.CONSENT:
                consent_valid, consent_violations = self._validate_consent(data_subject_id)
                if not consent_valid:
                    violations.extend(consent_violations)
                    articles_affected.append(GDPRArticle.ARTICLE_7)
            
            # 6. Check data minimization (Article 5)
            minimization_valid, min_violations = self._validate_data_minimization(
                data, processing_purpose
            )
            if not minimization_valid:
                violations.extend(min_violations)
                articles_affected.append(GDPRArticle.ARTICLE_5)
            
            # 7. Check retention limits (Article 5)
            retention_valid, retention_violations = self._validate_retention(data_categories)
            if not retention_valid:
                violations.extend(retention_violations)
                articles_affected.append(GDPRArticle.ARTICLE_5)
            
            # 8. Check cross-border transfer requirements (Chapter V)
            if cross_border_transfer:
                transfer_valid, transfer_violations = self._validate_cross_border_transfer()
                if not transfer_valid:
                    violations.extend(transfer_violations)
                    articles_affected.append(GDPRArticle.ARTICLE_44)
            
            # 9. Check security measures (Article 32)
            security_warnings = self._check_security_measures(data_categories)
            warnings.extend(security_warnings)
            if security_warnings:
                articles_affected.append(GDPRArticle.ARTICLE_32)
            
            # 10. Generate required actions
            required_actions = self._generate_required_actions(
                violations, warnings, data_categories, processing_purpose
            )
            
            # Calculate compliance score and risk level
            compliance_score = self._calculate_compliance_score(violations, warnings)
            risk_level = self._assess_risk_level(violations, data_categories)
            
            # Record processing activity (Article 30)
            self._record_processing_activity(
                processing_purpose, lawful_basis, data_categories, cross_border_transfer
            )
            
            assessment = GDPRAssessment(
                is_compliant=len(violations) == 0,
                violations=violations,
                warnings=warnings,
                required_actions=required_actions,
                articles_affected=articles_affected,
                risk_level=risk_level,
                compliance_score=compliance_score
            )
            
            self.logger.info(
                f"GDPR assessment complete: Compliant={assessment.is_compliant}, "
                f"Score={compliance_score:.2f}, Risk={risk_level}"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error during GDPR assessment: {str(e)}")
            raise ComplianceError(
                f"GDPR compliance assessment failed: {str(e)}",
                details={
                    "purpose": processing_purpose.value,
                    "lawful_basis": lawful_basis.value
                }
            )
    
    def _detect_personal_data(self, data: List[Dict[str, Any]]) -> PIIDetectionResult:
        """Detect personal data according to GDPR definition"""
        
        all_matches = []
        all_pii_types = set()
        
        for record in data:
            detection_results = self.pii_detector.detect_pii_in_document(record)
            
            for field_name, result in detection_results.items():
                all_matches.extend(result.matches)
                all_pii_types.update(result.pii_types_found)
        
        # Check for special category data
        special_category_found = self._detect_special_category_data(data)
        
        combined_result = PIIDetectionResult(
            text="Combined data assessment",
            matches=all_matches,
            total_matches=len(all_matches),
            pii_types_found=all_pii_types,
            has_pii=len(all_matches) > 0 or special_category_found,
            risk_level="HIGH" if special_category_found else "MEDIUM" if all_matches else "LOW"
        )
        
        return combined_result
    
    def _detect_special_category_data(self, data: List[Dict[str, Any]]) -> bool:
        """Detect Article 9 special category personal data"""
        
        for record in data:
            for field_name, value in record.items():
                if isinstance(value, str):
                    field_text = f"{field_name} {value}".lower()
                    
                    # Check against special category patterns
                    for category, patterns in self.special_category_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, field_text, re.IGNORECASE):
                                self.logger.warning(f"Special category data detected: {category}")
                                return True
        
        return False
    
    def _classify_data_categories(
        self,
        data: List[Dict[str, Any]],
        personal_data_result: PIIDetectionResult
    ) -> List[DataCategory]:
        """Classify data into GDPR categories"""
        
        categories = []
        
        if personal_data_result.has_pii:
            categories.append(DataCategory.PERSONAL_DATA)
        
        if self._detect_special_category_data(data):
            categories.append(DataCategory.SPECIAL_CATEGORY)
        
        # Check for criminal data
        if self._detect_criminal_data(data):
            categories.append(DataCategory.CRIMINAL_DATA)
        
        # If no personal data detected, consider it anonymous
        if not categories:
            categories.append(DataCategory.ANONYMOUS)
        
        return categories
    
    def _detect_criminal_data(self, data: List[Dict[str, Any]]) -> bool:
        """Detect criminal conviction and offences data"""
        
        criminal_patterns = [
            r"\b(criminal|conviction|offense|offence|crime|arrest|court|sentence)\b",
            r"\b(felony|misdemeanor|violation|penalty|fine|prison|jail)\b"
        ]
        
        for record in data:
            for field_name, value in record.items():
                if isinstance(value, str):
                    text = f"{field_name} {value}".lower()
                    
                    for pattern in criminal_patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            return True
        
        return False
    
    def _validate_lawful_basis(
        self,
        lawful_basis: LawfulBasis,
        purpose: ProcessingPurpose,
        data_categories: List[DataCategory],
        data_subject_id: Optional[str]
    ) -> tuple[bool, List[str]]:
        """Validate lawful basis for processing under Article 6"""
        
        violations = []
        
        # Check if lawful basis is appropriate for purpose
        if purpose == ProcessingPurpose.MARKETING and lawful_basis not in [
            LawfulBasis.CONSENT, LawfulBasis.LEGITIMATE_INTERESTS
        ]:
            violations.append("Marketing requires consent or legitimate interests basis")
        
        if purpose == ProcessingPurpose.RESEARCH and lawful_basis == LawfulBasis.CONTRACT:
            violations.append("Contract basis inappropriate for research purposes")
        
        # Special category data requires additional conditions
        if DataCategory.SPECIAL_CATEGORY in data_categories:
            if lawful_basis not in [LawfulBasis.CONSENT, LawfulBasis.VITAL_INTERESTS]:
                violations.append("Special category data requires explicit consent or vital interests")
        
        # Consent-specific validations
        if lawful_basis == LawfulBasis.CONSENT:
            if not data_subject_id:
                violations.append("Consent basis requires identifiable data subject")
            elif data_subject_id not in self.consent_records:
                violations.append("No consent record found for data subject")
        
        return len(violations) == 0, violations
    
    def _validate_special_category_processing(
        self,
        lawful_basis: LawfulBasis,
        purpose: ProcessingPurpose
    ) -> tuple[bool, List[str]]:
        """Validate Article 9 special category data processing"""
        
        violations = []
        
        # Article 9 requires both lawful basis AND additional condition
        valid_conditions = [
            "explicit_consent",
            "employment_law",
            "vital_interests",
            "nonprofit_activities",
            "public_data",
            "legal_claims",
            "substantial_public_interest",
            "healthcare",
            "public_health",
            "research_statistics"
        ]
        
        # Simplified validation - in practice, would need more detailed checks
        if purpose not in [ProcessingPurpose.RESEARCH, ProcessingPurpose.STATISTICS]:
            if lawful_basis != LawfulBasis.CONSENT:
                violations.append(
                    "Special category data processing requires explicit consent "
                    "or specific Article 9 condition"
                )
        
        return len(violations) == 0, violations
    
    def _validate_consent(self, data_subject_id: Optional[str]) -> tuple[bool, List[str]]:
        """Validate consent requirements under Article 7"""
        
        violations = []
        
        if not data_subject_id:
            violations.append("Data subject identification required for consent validation")
            return False, violations
        
        if data_subject_id not in self.consent_records:
            violations.append("No consent record found for data subject")
            return False, violations
        
        consent = self.consent_records[data_subject_id]
        
        # Check if consent is still valid
        if not consent.consent_given:
            violations.append("Consent not given by data subject")
        
        if consent.withdrawn_date:
            violations.append("Consent has been withdrawn")
        
        # Check consent age (not older than reasonable period)
        consent_age = (datetime.now() - consent.consent_date).days
        if consent_age > 365 * 2:  # 2 years
            violations.append("Consent is too old and may need refreshing")
        
        return len(violations) == 0, violations
    
    def _validate_data_minimization(
        self,
        data: List[Dict[str, Any]],
        purpose: ProcessingPurpose
    ) -> tuple[bool, List[str]]:
        """Validate data minimization principle (Article 5)"""
        
        violations = []
        
        # Define necessary fields for different purposes
        necessary_fields = {
            ProcessingPurpose.RESEARCH: ["age_range", "category", "status"],
            ProcessingPurpose.STATISTICS: ["aggregated_data", "category"],
            ProcessingPurpose.TESTING: ["test_data", "validation_fields"],
            ProcessingPurpose.MARKETING: ["contact_info", "preferences"],
            ProcessingPurpose.ANALYTICS: ["usage_data", "metrics"]
        }
        
        if purpose in necessary_fields:
            required_fields = necessary_fields[purpose]
            
            # Check if data contains more than necessary
            for record in data:
                extra_fields = set(record.keys()) - set(required_fields)
                sensitive_extra = [f for f in extra_fields if self._is_sensitive_field(f)]
                
                if sensitive_extra:
                    violations.append(
                        f"Unnecessary sensitive data detected for {purpose.value}: "
                        f"{', '.join(sensitive_extra)}"
                    )
        
        return len(violations) == 0, violations
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field contains sensitive data"""
        
        sensitive_indicators = [
            "ssn", "social", "credit", "card", "bank", "account",
            "medical", "health", "diagnosis", "political", "religious",
            "biometric", "genetic", "sexual", "criminal"
        ]
        
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in sensitive_indicators)
    
    def _validate_retention(self, data_categories: List[DataCategory]) -> tuple[bool, List[str]]:
        """Validate data retention limits (Article 5)"""
        
        violations = []
        
        for category in data_categories:
            if category in self.retention_limits:
                limit = self.retention_limits[category]
                if limit is not None:
                    # In practice, would check actual retention against limit
                    # For now, just flag if special category data with short limit
                    if category == DataCategory.SPECIAL_CATEGORY:
                        violations.append(
                            f"Special category data requires justified retention period "
                            f"(recommended max: {limit} days)"
                        )
        
        return len(violations) == 0, violations
    
    def _validate_cross_border_transfer(self) -> tuple[bool, List[str]]:
        """Validate international data transfer requirements (Chapter V)"""
        
        violations = []
        
        # Simplified validation - would need detailed adequacy decision checks
        violations.append(
            "Cross-border transfer requires adequacy decision, "
            "appropriate safeguards, or derogation"
        )
        
        return False, violations
    
    def _check_security_measures(self, data_categories: List[DataCategory]) -> List[str]:
        """Check security measures (Article 32)"""
        
        warnings = []
        
        if DataCategory.SPECIAL_CATEGORY in data_categories:
            warnings.append("Special category data requires enhanced security measures")
        
        if DataCategory.PERSONAL_DATA in data_categories:
            warnings.append("Ensure encryption, access controls, and audit logs are in place")
        
        warnings.append("Verify pseudonymization and anonymization techniques are adequate")
        warnings.append("Implement data breach detection and notification procedures")
        
        return warnings
    
    def _generate_required_actions(
        self,
        violations: List[str],
        warnings: List[str],
        data_categories: List[DataCategory],
        purpose: ProcessingPurpose
    ) -> List[str]:
        """Generate required actions for compliance"""
        
        actions = []
        
        if violations:
            actions.append("IMMEDIATE: Resolve all compliance violations before processing")
        
        if DataCategory.SPECIAL_CATEGORY in data_categories:
            actions.append("Conduct Data Protection Impact Assessment (DPIA)")
            actions.append("Implement enhanced security measures for special category data")
        
        if purpose in [ProcessingPurpose.RESEARCH, ProcessingPurpose.STATISTICS]:
            actions.append("Ensure data minimization and anonymization where possible")
        
        actions.append("Update privacy notice to reflect processing activities")
        actions.append("Maintain records of processing activities (Article 30)")
        
        if warnings:
            actions.append("Address security and privacy warnings")
        
        return actions
    
    def _calculate_compliance_score(self, violations: List[str], warnings: List[str]) -> float:
        """Calculate overall compliance score (0-1)"""
        
        base_score = 1.0
        
        # Deduct for violations
        violation_penalty = len(violations) * 0.2
        
        # Deduct for warnings
        warning_penalty = len(warnings) * 0.05
        
        total_penalty = violation_penalty + warning_penalty
        
        return max(0.0, base_score - total_penalty)
    
    def _assess_risk_level(self, violations: List[str], data_categories: List[DataCategory]) -> str:
        """Assess overall risk level"""
        
        if len(violations) >= 3:
            return "CRITICAL"
        elif DataCategory.SPECIAL_CATEGORY in data_categories and violations:
            return "CRITICAL"
        elif len(violations) >= 2:
            return "HIGH"
        elif len(violations) >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _record_processing_activity(
        self,
        purpose: ProcessingPurpose,
        lawful_basis: LawfulBasis,
        data_categories: List[DataCategory],
        cross_border: bool
    ):
        """Record processing activity as required by Article 30"""
        
        record = ProcessingRecord(
            processing_id=str(uuid.uuid4()),
            controller_name="Structured Documents Synthetic Data Generator",
            purpose=[purpose],
            lawful_basis=lawful_basis,
            data_categories=data_categories,
            recipients=["Internal Processing"],
            third_country_transfers=["Various"] if cross_border else [],
            retention_period=self.retention_limits.get(data_categories[0]) if data_categories else None,
            security_measures=["Encryption", "Access Control", "Audit Logging"],
            timestamp=datetime.now()
        )
        
        self.processing_records.append(record)
    
    def register_consent(
        self,
        data_subject_id: str,
        consent_given: bool,
        purposes: List[ProcessingPurpose],
        consent_method: str = "electronic_form"
    ) -> ConsentRecord:
        """Register data subject consent (Article 7)"""
        
        consent_record = ConsentRecord(
            subject_id=data_subject_id,
            consent_given=consent_given,
            consent_date=datetime.now(),
            consent_purpose=purposes,
            consent_method=consent_method
        )
        
        self.consent_records[data_subject_id] = consent_record
        
        self.logger.info(f"Consent registered for subject {data_subject_id}: {consent_given}")
        return consent_record
    
    def withdraw_consent(self, data_subject_id: str) -> bool:
        """Withdraw consent for data subject (Article 7)"""
        
        if data_subject_id in self.consent_records:
            self.consent_records[data_subject_id].consent_given = False
            self.consent_records[data_subject_id].withdrawn_date = datetime.now()
            
            self.logger.info(f"Consent withdrawn for subject {data_subject_id}")
            return True
        
        return False
    
    def conduct_dpia(
        self,
        processing_description: str,
        data_categories: List[DataCategory],
        processing_purposes: List[ProcessingPurpose]
    ) -> DataProtectionImpactAssessment:
        """Conduct Data Protection Impact Assessment (Article 35)"""
        
        # Simplified DPIA - real implementation would be more comprehensive
        risks = []
        mitigation_measures = []
        
        if DataCategory.SPECIAL_CATEGORY in data_categories:
            risks.append("High risk of discrimination if special category data is re-identified")
            mitigation_measures.append("Implement strong anonymization techniques")
        
        if ProcessingPurpose.RESEARCH in processing_purposes:
            risks.append("Risk of function creep - data used beyond research purposes")
            mitigation_measures.append("Implement purpose limitation controls")
        
        assessment = DataProtectionImpactAssessment(
            assessment_id=str(uuid.uuid4()),
            processing_description=processing_description,
            necessity_assessment="Processing necessary for specified purposes",
            proportionality_assessment="Data minimization principles applied",
            risks_identified=risks,
            mitigation_measures=mitigation_measures,
            residual_risks=["Minimal residual risks after mitigation"],
            decision="PROCEED" if len(risks) <= 2 else "MODIFY",
            consultation_required=len(risks) > 3,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"DPIA conducted: {assessment.decision}")
        return assessment
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive GDPR compliance report"""
        
        return {
            "processing_records": len(self.processing_records),
            "consent_records": len(self.consent_records),
            "active_consents": sum(1 for c in self.consent_records.values() if c.consent_given),
            "withdrawn_consents": sum(1 for c in self.consent_records.values() if c.withdrawn_date),
            "recent_processing": [
                {
                    "id": record.processing_id,
                    "purpose": [p.value for p in record.purpose],
                    "lawful_basis": record.lawful_basis.value,
                    "timestamp": record.timestamp.isoformat()
                }
                for record in self.processing_records[-10:]  # Last 10 records
            ],
            "compliance_summary": {
                "total_assessments": len(self.processing_records),
                "articles_monitored": [article.value for article in GDPRArticle],
                "data_categories_processed": list(set(
                    cat.value for record in self.processing_records 
                    for cat in record.data_categories
                ))
            }
        }