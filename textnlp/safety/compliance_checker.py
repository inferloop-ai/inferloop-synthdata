"""
Compliance Checking Implementation for TextNLP
Advanced compliance validation for regulatory and policy requirements
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import yaml
from collections import defaultdict

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Supported compliance standards"""
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    FERPA = "ferpa"  # Family Educational Rights and Privacy Act
    COPPA = "coppa"  # Children's Online Privacy Protection Act
    FTC_FAIR_CREDIT = "ftc_fair_credit"  # Fair Credit Reporting Act
    EU_AI_ACT = "eu_ai_act"  # EU AI Act
    NIST_AI_RMF = "nist_ai_rmf"  # NIST AI Risk Management Framework
    ISO_27001 = "iso_27001"  # Information Security Management
    CUSTOM = "custom"


class ComplianceViolationType(Enum):
    """Types of compliance violations"""
    DATA_PROTECTION = "data_protection"
    PRIVACY = "privacy"
    CONSENT = "consent"
    DISCLOSURE = "disclosure"
    RETENTION = "retention"
    SECURITY = "security"
    TRANSPARENCY = "transparency"
    BIAS_FAIRNESS = "bias_fairness"
    ACCURACY = "accuracy"
    ACCOUNTABILITY = "accountability"
    PROHIBITED_CONTENT = "prohibited_content"
    LEGAL_REQUIREMENT = "legal_requirement"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Individual compliance violation"""
    standard: ComplianceStandard
    violation_type: ComplianceViolationType
    severity: ViolationSeverity
    title: str
    description: str
    text_span: str
    start: int
    end: int
    confidence: float
    regulatory_reference: str
    remediation_steps: List[str]
    legal_risk: str
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCheckResult:
    """Result of compliance checking"""
    original_text: str
    violations: List[ComplianceViolation]
    standards_checked: List[ComplianceStandard]
    overall_compliance_score: float
    is_compliant: bool
    violation_summary: Dict[ComplianceStandard, int]
    severity_distribution: Dict[ViolationSeverity, int]
    processing_time: float
    remediation_required: bool
    compliance_report: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceChecker:
    """Advanced compliance checking for multiple regulatory standards"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled_standards = self.config.get("enabled_standards", [
            ComplianceStandard.GDPR, ComplianceStandard.CCPA, ComplianceStandard.HIPAA
        ])
        self.violation_threshold = self.config.get("violation_threshold", 0.5)
        
        # Load compliance rules and patterns
        self._load_compliance_rules()
        self._load_prohibited_content_patterns()
        self._load_data_classification_rules()
        
        # Initialize external compliance checkers if configured
        self._initialize_external_checkers()
    
    def _load_compliance_rules(self):
        """Load compliance rules for different standards"""
        self.compliance_rules = {
            ComplianceStandard.GDPR: {
                "data_categories": [
                    "personal_data", "special_category_data", "pseudonymized_data",
                    "anonymous_data", "genetic_data", "biometric_data"
                ],
                "prohibited_processing": [
                    "processing without consent", "automated decision making",
                    "profiling without explicit consent", "cross-border transfer"
                ],
                "required_disclosures": [
                    "purpose of processing", "legal basis", "retention period",
                    "data subject rights", "controller identity"
                ],
                "patterns": [
                    re.compile(r'\b(process|collect|store|analyze)\s+personal\s+data\b', re.IGNORECASE),
                    re.compile(r'\bautomated\s+(decision|profiling)\b', re.IGNORECASE),
                    re.compile(r'\btransfer.*outside\s+(eu|eea)\b', re.IGNORECASE),
                    re.compile(r'\bspecial\s+category\s+data\b', re.IGNORECASE)
                ]
            },
            ComplianceStandard.CCPA: {
                "data_categories": [
                    "personal_information", "sensitive_personal_information",
                    "biometric_information", "commercial_information"
                ],
                "consumer_rights": [
                    "right to know", "right to delete", "right to opt-out",
                    "right to non-discrimination"
                ],
                "patterns": [
                    re.compile(r'\bsell\s+personal\s+information\b', re.IGNORECASE),
                    re.compile(r'\bshare.*third\s+party\b', re.IGNORECASE),
                    re.compile(r'\bbiometric\s+(data|information)\b', re.IGNORECASE),
                    re.compile(r'\bcommercial\s+purposes\b', re.IGNORECASE)
                ]
            },
            ComplianceStandard.HIPAA: {
                "data_categories": [
                    "protected_health_information", "phi", "medical_records",
                    "health_data", "treatment_information"
                ],
                "security_requirements": [
                    "encryption", "access_controls", "audit_logs",
                    "breach_notification", "business_associate_agreement"
                ],
                "patterns": [
                    re.compile(r'\b(phi|protected\s+health\s+information)\b', re.IGNORECASE),
                    re.compile(r'\bmedical\s+(record|information|data)\b', re.IGNORECASE),
                    re.compile(r'\bhealth\s+(data|information)\b', re.IGNORECASE),
                    re.compile(r'\b(diagnosis|treatment|medication)\b', re.IGNORECASE),
                    re.compile(r'\bdoctor.*patient\b', re.IGNORECASE)
                ]
            },
            ComplianceStandard.EU_AI_ACT: {
                "risk_categories": [
                    "unacceptable_risk", "high_risk", "limited_risk", "minimal_risk"
                ],
                "prohibited_practices": [
                    "subliminal techniques", "exploitation of vulnerabilities",
                    "social scoring", "real-time remote biometric identification"
                ],
                "transparency_requirements": [
                    "ai_system_disclosure", "human_oversight", "risk_assessment",
                    "conformity_assessment", "post_market_monitoring"
                ],
                "patterns": [
                    re.compile(r'\b(ai|artificial\s+intelligence)\s+system\b', re.IGNORECASE),
                    re.compile(r'\bautomated\s+(decision|processing)\b', re.IGNORECASE),
                    re.compile(r'\bmachine\s+learning\s+model\b', re.IGNORECASE),
                    re.compile(r'\bbiometric\s+(identification|verification)\b', re.IGNORECASE),
                    re.compile(r'\bhigh.risk\s+ai\b', re.IGNORECASE)
                ]
            },
            ComplianceStandard.COPPA: {
                "age_restrictions": ["under_13", "children", "minors"],
                "prohibited_data": [
                    "personal_information_from_children", "online_contact_information",
                    "location_information", "photos_videos_audio"
                ],
                "patterns": [
                    re.compile(r'\b(children|kids|minors)\s+under\s+13\b', re.IGNORECASE),
                    re.compile(r'\bcollect.*from\s+(children|kids|minors)\b', re.IGNORECASE),
                    re.compile(r'\bparental\s+consent\b', re.IGNORECASE),
                    re.compile(r'\bchild.*personal\s+information\b', re.IGNORECASE)
                ]
            }
        }
    
    def _load_prohibited_content_patterns(self):
        """Load patterns for prohibited content across standards"""
        self.prohibited_patterns = {
            "financial_fraud": [
                re.compile(r'\bmoney\s+laundering\b', re.IGNORECASE),
                re.compile(r'\bterrorist\s+financing\b', re.IGNORECASE),
                re.compile(r'\bfraudulent\s+(transaction|activity)\b', re.IGNORECASE),
                re.compile(r'\bponzi\s+scheme\b', re.IGNORECASE)
            ],
            "healthcare_fraud": [
                re.compile(r'\bbilling\s+fraud\b', re.IGNORECASE),
                re.compile(r'\bfalse\s+claims\b', re.IGNORECASE),
                re.compile(r'\bunauthorized\s+medical\s+practice\b', re.IGNORECASE)
            ],
            "discrimination": [
                re.compile(r'\bdiscriminate\s+based\s+on\b', re.IGNORECASE),
                re.compile(r'\bredlining\b', re.IGNORECASE),
                re.compile(r'\bunfair\s+lending\s+practices\b', re.IGNORECASE)
            ],
            "privacy_violations": [
                re.compile(r'\bunauthorized\s+(access|disclosure)\b', re.IGNORECASE),
                re.compile(r'\bdata\s+breach\b', re.IGNORECASE),
                re.compile(r'\bprivacy\s+violation\b', re.IGNORECASE)
            ]
        }
    
    def _load_data_classification_rules(self):
        """Load data classification rules for sensitivity assessment"""
        self.data_classification = {
            "public": {
                "keywords": ["public", "open", "general", "marketing"],
                "risk_level": "low"
            },
            "internal": {
                "keywords": ["internal", "company", "employee", "staff"],
                "risk_level": "medium"
            },
            "confidential": {
                "keywords": ["confidential", "proprietary", "trade secret"],
                "risk_level": "high"
            },
            "restricted": {
                "keywords": ["personal", "sensitive", "private", "classified"],
                "risk_level": "critical"
            }
        }
    
    def _initialize_external_checkers(self):
        """Initialize external compliance checking services"""
        self.external_checkers = {}
        
        # Placeholder for external API integrations
        # In practice, these would integrate with compliance services like:
        # - Microsoft Compliance Manager
        # - AWS Config Rules
        # - Google Cloud Security Command Center
        # - Specialized compliance platforms
        
        if self.config.get("use_external_gdpr_checker"):
            # Initialize GDPR compliance checker
            pass
        
        if self.config.get("use_external_hipaa_checker"):
            # Initialize HIPAA compliance checker
            pass
    
    async def check_compliance(self, text: str, 
                             standards: Optional[List[ComplianceStandard]] = None,
                             context: Optional[Dict[str, Any]] = None) -> ComplianceCheckResult:
        """Check text compliance against specified standards"""
        start_time = asyncio.get_event_loop().time()
        
        if standards is None:
            standards = self.enabled_standards
        
        violations = []
        
        # Check each enabled standard
        for standard in standards:
            try:
                standard_violations = await self._check_standard_compliance(text, standard, context)
                violations.extend(standard_violations)
            except Exception as e:
                logger.error(f"Error checking {standard.value} compliance: {e}")
        
        # Remove duplicates and sort by severity
        violations = self._deduplicate_violations(violations)
        violations.sort(key=lambda v: (v.severity.value, -v.confidence))
        
        # Calculate compliance metrics
        overall_score = self._calculate_compliance_score(violations, text)
        is_compliant = overall_score >= 0.7 and not any(v.severity == ViolationSeverity.CRITICAL for v in violations)
        
        # Create summary statistics
        violation_summary = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for violation in violations:
            violation_summary[violation.standard] += 1
            severity_distribution[violation.severity] += 1
        
        # Determine if remediation is required
        remediation_required = any(v.severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL] for v in violations)
        
        # Generate compliance report
        compliance_report = self._generate_compliance_report(violations, standards, context)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ComplianceCheckResult(
            original_text=text,
            violations=violations,
            standards_checked=standards,
            overall_compliance_score=overall_score,
            is_compliant=is_compliant,
            violation_summary=dict(violation_summary),
            severity_distribution=dict(severity_distribution),
            processing_time=processing_time,
            remediation_required=remediation_required,
            compliance_report=compliance_report,
            metadata={
                "total_violations": len(violations),
                "standards_count": len(standards),
                "context": context
            }
        )
    
    async def _check_standard_compliance(self, text: str, 
                                        standard: ComplianceStandard,
                                        context: Optional[Dict[str, Any]] = None) -> List[ComplianceViolation]:
        """Check compliance for a specific standard"""
        violations = []
        
        if standard == ComplianceStandard.GDPR:
            violations.extend(await self._check_gdpr_compliance(text, context))
        elif standard == ComplianceStandard.CCPA:
            violations.extend(await self._check_ccpa_compliance(text, context))
        elif standard == ComplianceStandard.HIPAA:
            violations.extend(await self._check_hipaa_compliance(text, context))
        elif standard == ComplianceStandard.EU_AI_ACT:
            violations.extend(await self._check_eu_ai_act_compliance(text, context))
        elif standard == ComplianceStandard.COPPA:
            violations.extend(await self._check_coppa_compliance(text, context))
        elif standard == ComplianceStandard.SOX:
            violations.extend(await self._check_sox_compliance(text, context))
        elif standard == ComplianceStandard.PCI_DSS:
            violations.extend(await self._check_pci_dss_compliance(text, context))
        
        return violations
    
    async def _check_gdpr_compliance(self, text: str, context: Optional[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check GDPR compliance"""
        violations = []
        rules = self.compliance_rules[ComplianceStandard.GDPR]
        
        # Check for personal data processing without proper disclosure
        for pattern in rules["patterns"]:
            for match in pattern.finditer(text):
                if "personal data" in match.group().lower():
                    # Check if proper GDPR disclosures are present
                    if not self._has_gdpr_disclosures(text):
                        violation = ComplianceViolation(
                            standard=ComplianceStandard.GDPR,
                            violation_type=ComplianceViolationType.DATA_PROTECTION,
                            severity=ViolationSeverity.HIGH,
                            title="Missing GDPR Data Processing Disclosure",
                            description="Personal data processing mentioned without required GDPR disclosures",
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8,
                            regulatory_reference="GDPR Article 13-14",
                            remediation_steps=[
                                "Add purpose of processing disclosure",
                                "Specify legal basis for processing",
                                "Include data retention period",
                                "Provide data subject rights information",
                                "Identify data controller"
                            ],
                            legal_risk="High - Potential fines up to 4% of annual turnover",
                            context=text[max(0, match.start()-100):match.end()+100]
                        )
                        violations.append(violation)
        
        # Check for automated decision making
        automated_patterns = [
            re.compile(r'\bautomated\s+(decision|profiling)\b', re.IGNORECASE),
            re.compile(r'\balgorithmic\s+(decision|scoring)\b', re.IGNORECASE)
        ]
        
        for pattern in automated_patterns:
            for match in pattern.finditer(text):
                if not self._has_automated_decision_safeguards(text):
                    violation = ComplianceViolation(
                        standard=ComplianceStandard.GDPR,
                        violation_type=ComplianceViolationType.TRANSPARENCY,
                        severity=ViolationSeverity.MEDIUM,
                        title="Automated Decision Making Without Safeguards",
                        description="Automated decision making mentioned without GDPR Article 22 safeguards",
                        text_span=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7,
                        regulatory_reference="GDPR Article 22",
                        remediation_steps=[
                            "Provide meaningful information about the logic",
                            "Explain significance and consequences",
                            "Ensure human intervention rights",
                            "Allow challenging the decision"
                        ],
                        legal_risk="Medium - Requires explicit consent or legal basis",
                        context=text[max(0, match.start()-50):match.end()+50]
                    )
                    violations.append(violation)
        
        # Check for international data transfers
        transfer_patterns = [
            re.compile(r'\btransfer.*outside\s+(eu|eea|europe)\b', re.IGNORECASE),
            re.compile(r'\bcross.border\s+transfer\b', re.IGNORECASE),
            re.compile(r'\bthird\s+country\s+transfer\b', re.IGNORECASE)
        ]
        
        for pattern in transfer_patterns:
            for match in pattern.finditer(text):
                if not self._has_transfer_safeguards(text):
                    violation = ComplianceViolation(
                        standard=ComplianceStandard.GDPR,
                        violation_type=ComplianceViolationType.DATA_PROTECTION,
                        severity=ViolationSeverity.HIGH,
                        title="International Data Transfer Without Safeguards",
                        description="International data transfer mentioned without adequate safeguards",
                        text_span=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        regulatory_reference="GDPR Articles 44-49",
                        remediation_steps=[
                            "Implement adequacy decision",
                            "Use appropriate safeguards (SCCs, BCRs)",
                            "Obtain explicit consent if needed",
                            "Document transfer mechanism"
                        ],
                        legal_risk="High - Potential suspension of transfers",
                        context=text[max(0, match.start()-50):match.end()+50]
                    )
                    violations.append(violation)
        
        return violations
    
    async def _check_ccpa_compliance(self, text: str, context: Optional[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check CCPA compliance"""
        violations = []
        rules = self.compliance_rules[ComplianceStandard.CCPA]
        
        # Check for sale of personal information
        for pattern in rules["patterns"]:
            for match in pattern.finditer(text):
                if "sell" in match.group().lower() and "personal" in match.group().lower():
                    if not self._has_ccpa_opt_out_notice(text):
                        violation = ComplianceViolation(
                            standard=ComplianceStandard.CCPA,
                            violation_type=ComplianceViolationType.PRIVACY,
                            severity=ViolationSeverity.HIGH,
                            title="Sale of Personal Information Without Opt-Out Notice",
                            description="Sale of personal information mentioned without required opt-out notice",
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8,
                            regulatory_reference="CCPA Section 1798.135",
                            remediation_steps=[
                                "Provide clear opt-out notice",
                                "Include 'Do Not Sell My Personal Information' link",
                                "Implement opt-out mechanism",
                                "Honor opt-out requests within 15 days"
                            ],
                            legal_risk="High - Potential fines up to $7,500 per violation",
                            context=text[max(0, match.start()-50):match.end()+50]
                        )
                        violations.append(violation)
        
        return violations
    
    async def _check_hipaa_compliance(self, text: str, context: Optional[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check HIPAA compliance"""
        violations = []
        rules = self.compliance_rules[ComplianceStandard.HIPAA]
        
        # Check for PHI exposure
        for pattern in rules["patterns"]:
            for match in pattern.finditer(text):
                if any(term in match.group().lower() for term in ["phi", "medical", "health"]):
                    if not self._has_hipaa_safeguards(text, context):
                        violation = ComplianceViolation(
                            standard=ComplianceStandard.HIPAA,
                            violation_type=ComplianceViolationType.DATA_PROTECTION,
                            severity=ViolationSeverity.CRITICAL,
                            title="Potential PHI Exposure",
                            description="Protected Health Information referenced without adequate safeguards",
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.9,
                            regulatory_reference="HIPAA Privacy Rule 45 CFR 164.502",
                            remediation_steps=[
                                "Ensure minimum necessary standard",
                                "Implement access controls",
                                "Add audit logging",
                                "Verify business associate agreements",
                                "Apply encryption in transit and at rest"
                            ],
                            legal_risk="Critical - Potential fines up to $1.5M per incident",
                            context=text[max(0, match.start()-50):match.end()+50]
                        )
                        violations.append(violation)
        
        return violations
    
    async def _check_eu_ai_act_compliance(self, text: str, context: Optional[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check EU AI Act compliance"""
        violations = []
        rules = self.compliance_rules[ComplianceStandard.EU_AI_ACT]
        
        # Check for AI system disclosure requirements
        for pattern in rules["patterns"]:
            for match in pattern.finditer(text):
                if "ai" in match.group().lower() or "artificial intelligence" in match.group().lower():
                    if not self._has_ai_transparency_notice(text):
                        violation = ComplianceViolation(
                            standard=ComplianceStandard.EU_AI_ACT,
                            violation_type=ComplianceViolationType.TRANSPARENCY,
                            severity=ViolationSeverity.MEDIUM,
                            title="AI System Without Transparency Notice",
                            description="AI system referenced without required transparency disclosures",
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.7,
                            regulatory_reference="EU AI Act Article 52",
                            remediation_steps=[
                                "Disclose AI system usage",
                                "Provide human oversight information",
                                "Include accuracy limitations",
                                "Specify decision-making boundaries"
                            ],
                            legal_risk="Medium - Administrative fines up to €15M or 3% of turnover",
                            context=text[max(0, match.start()-50):match.end()+50]
                        )
                        violations.append(violation)
        
        return violations
    
    async def _check_coppa_compliance(self, text: str, context: Optional[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check COPPA compliance"""
        violations = []
        rules = self.compliance_rules[ComplianceStandard.COPPA]
        
        # Check for children's data collection
        for pattern in rules["patterns"]:
            for match in pattern.finditer(text):
                if "children" in match.group().lower() or "under 13" in match.group().lower():
                    if not self._has_parental_consent_mechanism(text):
                        violation = ComplianceViolation(
                            standard=ComplianceStandard.COPPA,
                            violation_type=ComplianceViolationType.CONSENT,
                            severity=ViolationSeverity.CRITICAL,
                            title="Children's Data Collection Without Parental Consent",
                            description="Collection from children under 13 without verifiable parental consent",
                            text_span=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.9,
                            regulatory_reference="COPPA Rule 16 CFR 312.5",
                            remediation_steps=[
                                "Implement verifiable parental consent",
                                "Provide clear privacy notice to parents",
                                "Allow parental review and deletion",
                                "Limit data collection to what's necessary"
                            ],
                            legal_risk="Critical - Fines up to $43,280 per violation",
                            context=text[max(0, match.start()-50):match.end()+50]
                        )
                        violations.append(violation)
        
        return violations
    
    async def _check_sox_compliance(self, text: str, context: Optional[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check Sarbanes-Oxley compliance"""
        violations = []
        
        # Check for financial reporting issues
        sox_patterns = [
            re.compile(r'\bfinancial\s+(statement|reporting|disclosure)\b', re.IGNORECASE),
            re.compile(r'\binternal\s+control\b', re.IGNORECASE),
            re.compile(r'\baudit\s+(trail|log|record)\b', re.IGNORECASE)
        ]
        
        for pattern in sox_patterns:
            for match in pattern.finditer(text):
                if not self._has_sox_controls(text, context):
                    violation = ComplianceViolation(
                        standard=ComplianceStandard.SOX,
                        violation_type=ComplianceViolationType.ACCOUNTABILITY,
                        severity=ViolationSeverity.HIGH,
                        title="Inadequate Financial Controls",
                        description="Financial operations mentioned without adequate SOX controls",
                        text_span=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.6,
                        regulatory_reference="SOX Section 404",
                        remediation_steps=[
                            "Implement internal control framework",
                            "Ensure audit trail completeness",
                            "Document control procedures",
                            "Regular control testing"
                        ],
                        legal_risk="High - Criminal and civil penalties",
                        context=text[max(0, match.start()-50):match.end()+50]
                    )
                    violations.append(violation)
        
        return violations
    
    async def _check_pci_dss_compliance(self, text: str, context: Optional[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check PCI DSS compliance"""
        violations = []
        
        # Check for payment card data handling
        pci_patterns = [
            re.compile(r'\b(credit\s+card|payment\s+card|cardholder)\s+data\b', re.IGNORECASE),
            re.compile(r'\bpan\s+(primary\s+account\s+number)?\b', re.IGNORECASE),
            re.compile(r'\bcard\s+verification\s+(value|code)\b', re.IGNORECASE)
        ]
        
        for pattern in pci_patterns:
            for match in pattern.finditer(text):
                if not self._has_pci_security_measures(text, context):
                    violation = ComplianceViolation(
                        standard=ComplianceStandard.PCI_DSS,
                        violation_type=ComplianceViolationType.SECURITY,
                        severity=ViolationSeverity.CRITICAL,
                        title="Payment Card Data Without PCI Controls",
                        description="Payment card data referenced without adequate PCI DSS security measures",
                        text_span=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        regulatory_reference="PCI DSS Requirement 3",
                        remediation_steps=[
                            "Encrypt stored cardholder data",
                            "Implement strong access controls",
                            "Use secure transmission protocols",
                            "Regular security testing",
                            "Maintain vulnerability management"
                        ],
                        legal_risk="Critical - Card brand fines and liability",
                        context=text[max(0, match.start()-50):match.end()+50]
                    )
                    violations.append(violation)
        
        return violations
    
    def _has_gdpr_disclosures(self, text: str) -> bool:
        """Check if text contains required GDPR disclosures"""
        disclosure_indicators = [
            "purpose of processing", "legal basis", "data retention",
            "data subject rights", "controller", "processor",
            "legitimate interest", "consent", "withdrawal"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in disclosure_indicators)
    
    def _has_automated_decision_safeguards(self, text: str) -> bool:
        """Check if automated decision making has proper safeguards"""
        safeguard_indicators = [
            "human intervention", "contest the decision", "meaningful information",
            "logic involved", "significance", "consequences"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in safeguard_indicators)
    
    def _has_transfer_safeguards(self, text: str) -> bool:
        """Check if international transfers have adequate safeguards"""
        transfer_safeguards = [
            "adequacy decision", "standard contractual clauses", "binding corporate rules",
            "sccs", "bcrs", "privacy shield", "explicit consent"
        ]
        text_lower = text.lower()
        return any(safeguard in text_lower for safeguard in transfer_safeguards)
    
    def _has_ccpa_opt_out_notice(self, text: str) -> bool:
        """Check if CCPA opt-out notice is present"""
        opt_out_indicators = [
            "do not sell", "opt out", "opt-out", "right to opt out",
            "ccpa", "california consumer privacy"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in opt_out_indicators)
    
    def _has_hipaa_safeguards(self, text: str, context: Optional[Dict[str, Any]]) -> bool:
        """Check if HIPAA safeguards are present"""
        safeguard_indicators = [
            "minimum necessary", "authorized", "business associate",
            "encryption", "access control", "audit log"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in safeguard_indicators)
    
    def _has_ai_transparency_notice(self, text: str) -> bool:
        """Check if AI transparency notice is present"""
        transparency_indicators = [
            "ai system", "automated", "machine learning", "algorithm",
            "human oversight", "artificial intelligence", "decision support"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in transparency_indicators)
    
    def _has_parental_consent_mechanism(self, text: str) -> bool:
        """Check if parental consent mechanism is described"""
        consent_indicators = [
            "parental consent", "verifiable consent", "parent approval",
            "guardian permission", "age verification"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in consent_indicators)
    
    def _has_sox_controls(self, text: str, context: Optional[Dict[str, Any]]) -> bool:
        """Check if SOX controls are mentioned"""
        control_indicators = [
            "internal control", "audit trail", "segregation of duties",
            "authorization", "documentation", "testing"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in control_indicators)
    
    def _has_pci_security_measures(self, text: str, context: Optional[Dict[str, Any]]) -> bool:
        """Check if PCI security measures are described"""
        security_indicators = [
            "encryption", "tokenization", "secure transmission",
            "access control", "pci compliant", "security testing"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in security_indicators)
    
    def _deduplicate_violations(self, violations: List[ComplianceViolation]) -> List[ComplianceViolation]:
        """Remove duplicate violations"""
        seen = set()
        deduplicated = []
        
        for violation in violations:
            # Create a unique key based on violation characteristics
            key = (
                violation.standard.value,
                violation.violation_type.value,
                violation.start,
                violation.end,
                violation.title
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(violation)
        
        return deduplicated
    
    def _calculate_compliance_score(self, violations: List[ComplianceViolation], text: str) -> float:
        """Calculate overall compliance score"""
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.LOW: 0.1,
            ViolationSeverity.MEDIUM: 0.3,
            ViolationSeverity.HIGH: 0.6,
            ViolationSeverity.CRITICAL: 1.0
        }
        
        total_penalty = sum(severity_weights[v.severity] * v.confidence for v in violations)
        
        # Normalize by text length and number of standards checked
        normalized_penalty = total_penalty / max(1, len(text) / 1000)
        
        # Calculate score (0-1, where 1 is fully compliant)
        compliance_score = max(0.0, 1.0 - normalized_penalty)
        
        return compliance_score
    
    def _generate_compliance_report(self, violations: List[ComplianceViolation], 
                                   standards: List[ComplianceStandard],
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed compliance report"""
        report = {
            "executive_summary": {
                "total_violations": len(violations),
                "critical_violations": sum(1 for v in violations if v.severity == ViolationSeverity.CRITICAL),
                "standards_evaluated": [s.value for s in standards],
                "overall_risk_level": self._determine_overall_risk(violations)
            },
            "violations_by_standard": {},
            "violations_by_type": {},
            "remediation_priority": [],
            "recommended_actions": [],
            "legal_risks": []
        }
        
        # Group violations by standard
        for standard in standards:
            standard_violations = [v for v in violations if v.standard == standard]
            report["violations_by_standard"][standard.value] = {
                "count": len(standard_violations),
                "severity_breakdown": {
                    severity.value: sum(1 for v in standard_violations if v.severity == severity)
                    for severity in ViolationSeverity
                }
            }
        
        # Group violations by type
        for violation_type in ComplianceViolationType:
            type_violations = [v for v in violations if v.violation_type == violation_type]
            if type_violations:
                report["violations_by_type"][violation_type.value] = len(type_violations)
        
        # Prioritize remediation
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        high_violations = [v for v in violations if v.severity == ViolationSeverity.HIGH]
        
        report["remediation_priority"] = (
            [v.title for v in critical_violations] +
            [v.title for v in high_violations]
        )
        
        # Collect recommended actions
        all_remediation_steps = []
        for violation in violations:
            all_remediation_steps.extend(violation.remediation_steps)
        
        # Remove duplicates while preserving order
        seen = set()
        report["recommended_actions"] = [
            step for step in all_remediation_steps
            if not (step in seen or seen.add(step))
        ]
        
        # Collect legal risks
        report["legal_risks"] = list(set(v.legal_risk for v in violations))
        
        return report
    
    def _determine_overall_risk(self, violations: List[ComplianceViolation]) -> str:
        """Determine overall risk level"""
        if any(v.severity == ViolationSeverity.CRITICAL for v in violations):
            return "Critical"
        elif any(v.severity == ViolationSeverity.HIGH for v in violations):
            return "High"
        elif any(v.severity == ViolationSeverity.MEDIUM for v in violations):
            return "Medium"
        elif violations:
            return "Low"
        else:
            return "Minimal"
    
    async def batch_check_compliance(self, texts: List[str], 
                                   standards: Optional[List[ComplianceStandard]] = None) -> List[ComplianceCheckResult]:
        """Check compliance for multiple texts"""
        tasks = [self.check_compliance(text, standards) for text in texts]
        return await asyncio.gather(*tasks)
    
    def create_compliance_summary(self, results: List[ComplianceCheckResult]) -> Dict[str, Any]:
        """Create summary report for multiple compliance checks"""
        total_texts = len(results)
        compliant_texts = sum(1 for r in results if r.is_compliant)
        
        # Aggregate violations by standard
        standard_violations = defaultdict(int)
        for result in results:
            for standard, count in result.violation_summary.items():
                standard_violations[standard.value] += count
        
        # Aggregate by severity
        severity_counts = defaultdict(int)
        for result in results:
            for severity, count in result.severity_distribution.items():
                severity_counts[severity.value] += count
        
        return {
            "summary": {
                "total_texts_analyzed": total_texts,
                "compliant_texts": compliant_texts,
                "compliance_rate": compliant_texts / total_texts if total_texts > 0 else 0,
                "average_compliance_score": sum(r.overall_compliance_score for r in results) / total_texts if total_texts > 0 else 0,
                "texts_requiring_remediation": sum(1 for r in results if r.remediation_required)
            },
            "violations_by_standard": dict(standard_violations),
            "severity_distribution": dict(severity_counts),
            "performance": {
                "average_processing_time": sum(r.processing_time for r in results) / total_texts if total_texts > 0 else 0,
                "total_processing_time": sum(r.processing_time for r in results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class ComplianceService:
    """Service wrapper for compliance checking with configuration management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checker = ComplianceChecker(config)
        self.cache = {}
        self.cache_enabled = config.get("enable_cache", True)
        self.max_cache_size = config.get("max_cache_size", 1000)
    
    async def check_and_report(self, text: str, 
                             standards: Optional[List[ComplianceStandard]] = None,
                             context: Optional[Dict[str, Any]] = None) -> ComplianceCheckResult:
        """Check compliance and generate comprehensive report"""
        cache_key = None
        if self.cache_enabled:
            cache_key = hashlib.md5(f"{text}{standards}{context}".encode()).hexdigest()
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        result = await self.checker.check_compliance(text, standards, context)
        
        if self.cache_enabled and cache_key:
            # Simple LRU cache implementation
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
        
        return result
    
    def get_supported_standards(self) -> List[str]:
        """Get list of supported compliance standards"""
        return [standard.value for standard in ComplianceStandard]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_enabled": self.cache_enabled,
            "supported_standards": self.get_supported_standards(),
            "config": self.config
        }


# Example usage
if __name__ == "__main__":
    async def example():
        # Sample texts with compliance issues
        test_texts = [
            "We process personal data for marketing purposes and share it with third parties.",  # GDPR violation
            "Our AI system makes automated decisions about loan approvals.",  # EU AI Act issue
            "We collect information from children under 13 for our service.",  # COPPA violation
            "This is compliant text that follows all regulations.",  # Clean
        ]
        
        # Initialize compliance checker
        config = {
            "enabled_standards": [
                ComplianceStandard.GDPR,
                ComplianceStandard.CCPA,
                ComplianceStandard.EU_AI_ACT,
                ComplianceStandard.COPPA
            ],
            "violation_threshold": 0.5
        }
        checker = ComplianceChecker(config)
        
        # Check compliance for each text
        for text in test_texts:
            result = await checker.check_compliance(text)
            print(f"\nText: {text}")
            print(f"Compliant: {result.is_compliant}")
            print(f"Compliance Score: {result.overall_compliance_score:.3f}")
            print(f"Violations: {len(result.violations)}")
            
            for violation in result.violations:
                print(f"- {violation.standard.value}: {violation.title} ({violation.severity.value})")
    
    # Run example
    # asyncio.run(example())