"""
SOX (Sarbanes-Oxley Act) compliance enforcer
Implements SOX requirements for financial reporting and data protection
"""

import re
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ...core import get_logger, ComplianceError, PrivacyError
from ..pii_protection.pii_detector import PIIDetector, PIIType, PIIDetectionResult


class SOXSection(Enum):
    """SOX Act sections and requirements"""
    SECTION_302 = "section_302"  # Corporate responsibility for financial reports
    SECTION_404 = "section_404"  # Management assessment of internal controls
    SECTION_409 = "section_409"  # Real-time disclosure
    SECTION_802 = "section_802"  # Criminal penalties for document destruction
    SECTION_906 = "section_906"  # Criminal penalties for CEO/CFO certification


class FinancialDataType(Enum):
    """Types of financial data subject to SOX requirements"""
    REVENUE_DATA = "revenue_data"
    EXPENSE_DATA = "expense_data"
    ASSET_DATA = "asset_data"
    LIABILITY_DATA = "liability_data"
    EQUITY_DATA = "equity_data"
    CASH_FLOW_DATA = "cash_flow_data"
    INVESTMENT_DATA = "investment_data"
    DERIVATIVE_DATA = "derivative_data"
    FOREIGN_EXCHANGE_DATA = "foreign_exchange_data"
    INTERNAL_CONTROLS_DATA = "internal_controls_data"


class DataSensitivityLevel(Enum):
    """SOX data sensitivity classification"""
    PUBLIC = "public"  # Already disclosed financial information
    INTERNAL = "internal"  # Internal financial metrics
    CONFIDENTIAL = "confidential"  # Pre-disclosure financial data
    MATERIAL = "material"  # Material information affecting stock price


class ControlObjective(Enum):
    """SOX internal control objectives"""
    EXISTENCE_OCCURRENCE = "existence_occurrence"
    COMPLETENESS = "completeness"
    ACCURACY_VALUATION = "accuracy_valuation"
    CUTOFF = "cutoff"
    CLASSIFICATION = "classification"
    PRESENTATION_DISCLOSURE = "presentation_disclosure"


class RetentionRequirement(Enum):
    """SOX document retention requirements"""
    SEVEN_YEARS = "seven_years"  # Audit workpapers
    FIVE_YEARS = "five_years"  # Financial records
    THREE_YEARS = "three_years"  # Supporting documentation
    PERMANENT = "permanent"  # Articles of incorporation, board minutes


@dataclass
class FinancialDataAssessment:
    """Assessment of financial data subject to SOX"""
    contains_financial_data: bool
    data_types_found: List[FinancialDataType]
    sensitivity_level: DataSensitivityLevel
    is_material_information: bool
    requires_disclosure: bool
    retention_period: RetentionRequirement
    control_objectives_affected: List[ControlObjective]
    recommendations: List[str]


@dataclass
class InternalControlAssessment:
    """Assessment of internal controls effectiveness"""
    controls_documented: bool
    control_deficiencies: List[str]
    material_weaknesses: List[str]
    significant_deficiencies: List[str]
    remediation_required: bool
    management_certification_required: bool


@dataclass
class SOXAssessment:
    """Comprehensive SOX compliance assessment"""
    is_compliant: bool
    violations: List[str]
    warnings: List[str]
    sections_affected: List[SOXSection]
    financial_data_assessment: FinancialDataAssessment
    internal_control_assessment: InternalControlAssessment
    disclosure_timeline: Optional[datetime]
    risk_level: str


@dataclass
class AuditTrailEntry:
    """SOX audit trail entry"""
    entry_id: str
    timestamp: datetime
    user_id: str
    action: str
    data_affected: str
    before_value: Optional[str]
    after_value: Optional[str]
    business_justification: str
    approval_required: bool
    approved_by: Optional[str]


class SOXEnforcer:
    """SOX compliance enforcement and validation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pii_detector = PIIDetector()
        
        # Financial data patterns
        self.financial_patterns = {
            FinancialDataType.REVENUE_DATA: [
                r'\brevenue\b', r'\bsales\b', r'\bincome\b', r'\bearnings\b',
                r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?',
                r'\bnet\s+income\b', r'\bgross\s+profit\b'
            ],
            FinancialDataType.EXPENSE_DATA: [
                r'\bexpenses?\b', r'\bcosts?\b', r'\boperating\s+expense\b',
                r'\bCOGS\b', r'\bcost\s+of\s+goods\s+sold\b',
                r'\bR&D\b', r'\bresearch\s+and\s+development\b'
            ],
            FinancialDataType.ASSET_DATA: [
                r'\bassets?\b', r'\bcash\b', r'\binventory\b',
                r'\baccounts\s+receivable\b', r'\bfixed\s+assets?\b',
                r'\bintangible\s+assets?\b', r'\bgoodwill\b'
            ],
            FinancialDataType.LIABILITY_DATA: [
                r'\bliabilit(?:y|ies)\b', r'\bdebt\b', r'\baccounts\s+payable\b',
                r'\baccrued\s+expenses?\b', r'\blong.term\s+debt\b'
            ],
            FinancialDataType.CASH_FLOW_DATA: [
                r'\bcash\s+flow\b', r'\boperating\s+cash\s+flow\b',
                r'\bfree\s+cash\s+flow\b', r'\bcapital\s+expenditures?\b'
            ]
        }
        
        # Financial field indicators
        self.financial_indicators = [
            'revenue', 'income', 'profit', 'loss', 'earnings', 'ebitda',
            'assets', 'liabilities', 'equity', 'cash', 'debt', 'investment',
            'financial', 'accounting', 'audit', 'balance', 'statement',
            'quarterly', 'annual', 'fiscal', 'sec', 'gaap', 'ifrs'
        ]
        
        # Material information indicators
        self.materiality_indicators = [
            'material', 'significant', 'substantial', 'major', 'critical',
            'merger', 'acquisition', 'divestiture', 'bankruptcy', 'litigation',
            'regulatory', 'investigation', 'restatement', 'going concern'
        ]
        
        # Audit trail storage
        self.audit_trail: List[AuditTrailEntry] = []
        
        self.logger.info("SOX Enforcer initialized")
    
    def assess_sox_compliance(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        is_public_company: bool = True,
        fiscal_period: str = "quarterly",
        disclosure_deadline: Optional[datetime] = None,
        data_context: str = "financial_reporting"
    ) -> SOXAssessment:
        """Comprehensive SOX compliance assessment"""
        
        try:
            violations = []
            warnings = []
            sections_affected = []
            
            # Convert to list for uniform processing
            if isinstance(data, dict):
                data = [data]
            
            # 1. Assess financial data content
            financial_assessment = self._assess_financial_data_content(data)
            
            # 2. Assess internal controls
            control_assessment = self._assess_internal_controls(
                data, financial_assessment, data_context
            )
            
            # 3. Check Section 302 compliance (Corporate responsibility)
            if is_public_company and financial_assessment.contains_financial_data:
                section_302_violations = self._check_section_302_compliance(
                    financial_assessment, fiscal_period
                )
                if section_302_violations:
                    violations.extend(section_302_violations)
                    sections_affected.append(SOXSection.SECTION_302)
            
            # 4. Check Section 404 compliance (Internal controls)
            if is_public_company:
                section_404_violations = self._check_section_404_compliance(
                    control_assessment, financial_assessment
                )
                if section_404_violations:
                    violations.extend(section_404_violations)
                    sections_affected.append(SOXSection.SECTION_404)
            
            # 5. Check Section 409 compliance (Real-time disclosure)
            if financial_assessment.is_material_information:
                section_409_violations = self._check_section_409_compliance(
                    financial_assessment, disclosure_deadline
                )
                if section_409_violations:
                    violations.extend(section_409_violations)
                    sections_affected.append(SOXSection.SECTION_409)
            
            # 6. Check Section 802 compliance (Document retention)
            retention_warnings = self._check_document_retention_requirements(
                financial_assessment
            )
            if retention_warnings:
                warnings.extend(retention_warnings)
                sections_affected.append(SOXSection.SECTION_802)
            
            # 7. Determine disclosure timeline
            disclosure_timeline = self._determine_disclosure_timeline(
                financial_assessment, fiscal_period, disclosure_deadline
            )
            
            # 8. Overall risk assessment
            risk_level = self._assess_overall_sox_risk(
                financial_assessment, control_assessment, violations, warnings
            )
            
            assessment = SOXAssessment(
                is_compliant=len(violations) == 0,
                violations=violations,
                warnings=warnings,
                sections_affected=sections_affected,
                financial_data_assessment=financial_assessment,
                internal_control_assessment=control_assessment,
                disclosure_timeline=disclosure_timeline,
                risk_level=risk_level
            )
            
            self.logger.info(
                f"SOX assessment complete: Compliant={assessment.is_compliant}, "
                f"Financial={financial_assessment.contains_financial_data}, "
                f"Material={financial_assessment.is_material_information}, "
                f"Risk={risk_level}"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error during SOX assessment: {str(e)}")
            raise ComplianceError(
                f"SOX compliance assessment failed: {str(e)}",
                details={"fiscal_period": fiscal_period, "data_context": data_context}
            )
    
    def _assess_financial_data_content(
        self, 
        data: List[Dict[str, Any]]
    ) -> FinancialDataAssessment:
        """Assess if data contains financial information subject to SOX"""
        
        data_types_found = []
        contains_financial_context = False
        is_material = False
        
        # Check for financial context
        contains_financial_context = self._detect_financial_context(data)
        
        # Check for financial data types
        for record in data:
            for field_name, value in record.items():
                if isinstance(value, str):
                    financial_types = self._detect_financial_data_types(field_name, value)
                    data_types_found.extend(financial_types)
                    
                    # Check for material information indicators
                    if self._is_material_information(field_name, value):
                        is_material = True
        
        # Remove duplicates
        data_types_found = list(set(data_types_found))
        
        # Determine if this constitutes financial data
        contains_financial_data = (
            contains_financial_context and 
            len(data_types_found) > 0
        )
        
        # Determine sensitivity level
        sensitivity_level = self._determine_sensitivity_level(
            data_types_found, is_material
        )
        
        # Check disclosure requirements
        requires_disclosure = self._requires_disclosure(
            contains_financial_data, is_material, sensitivity_level
        )
        
        # Determine retention period
        retention_period = self._determine_retention_period(data_types_found)
        
        # Identify affected control objectives
        control_objectives = self._identify_control_objectives(data_types_found)
        
        # Generate recommendations
        recommendations = self._generate_financial_data_recommendations(
            contains_financial_data, data_types_found, is_material, sensitivity_level
        )
        
        return FinancialDataAssessment(
            contains_financial_data=contains_financial_data,
            data_types_found=data_types_found,
            sensitivity_level=sensitivity_level,
            is_material_information=is_material,
            requires_disclosure=requires_disclosure,
            retention_period=retention_period,
            control_objectives_affected=control_objectives,
            recommendations=recommendations
        )
    
    def _detect_financial_context(self, data: List[Dict[str, Any]]) -> bool:
        """Detect if data is in a financial/accounting context"""
        
        for record in data:
            for field_name, value in record.items():
                # Check field names
                field_text = field_name.lower()
                if any(indicator in field_text for indicator in self.financial_indicators):
                    return True
                
                # Check field values
                if isinstance(value, str):
                    value_text = value.lower()
                    if any(indicator in value_text for indicator in self.financial_indicators):
                        return True
        
        return False
    
    def _detect_financial_data_types(
        self, 
        field_name: str, 
        value: str
    ) -> List[FinancialDataType]:
        """Detect SOX financial data types"""
        
        types_found = []
        
        for data_type, patterns in self.financial_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    types_found.append(data_type)
                    break
        
        # Additional checks based on field names
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ['revenue', 'sales', 'income']):
            types_found.append(FinancialDataType.REVENUE_DATA)
        
        if any(term in field_lower for term in ['expense', 'cost', 'expenditure']):
            types_found.append(FinancialDataType.EXPENSE_DATA)
        
        if any(term in field_lower for term in ['asset', 'cash', 'inventory']):
            types_found.append(FinancialDataType.ASSET_DATA)
        
        if any(term in field_lower for term in ['liability', 'debt', 'payable']):
            types_found.append(FinancialDataType.LIABILITY_DATA)
        
        return list(set(types_found))
    
    def _is_material_information(self, field_name: str, value: str) -> bool:
        """Check if information is material under SOX standards"""
        
        combined_text = f"{field_name} {value}".lower()
        
        return any(
            indicator in combined_text 
            for indicator in self.materiality_indicators
        )
    
    def _assess_internal_controls(
        self,
        data: List[Dict[str, Any]],
        financial_assessment: FinancialDataAssessment,
        data_context: str
    ) -> InternalControlAssessment:
        """Assess internal controls effectiveness"""
        
        control_deficiencies = []
        material_weaknesses = []
        significant_deficiencies = []
        
        if financial_assessment.contains_financial_data:
            # Check for common control deficiencies
            if data_context == "manual_entry":
                control_deficiencies.append(
                    "Manual data entry without automated controls"
                )
            
            # Check for segregation of duties
            control_deficiencies.append(
                "Segregation of duties verification required"
            )
            
            # Check for authorization controls
            control_deficiencies.append(
                "Authorization controls must be documented and tested"
            )
            
            # Classify deficiencies
            if len(control_deficiencies) >= 3:
                material_weaknesses.extend(control_deficiencies[:2])
                significant_deficiencies.extend(control_deficiencies[2:])
            elif len(control_deficiencies) >= 1:
                significant_deficiencies.extend(control_deficiencies)
        
        return InternalControlAssessment(
            controls_documented=len(control_deficiencies) == 0,
            control_deficiencies=control_deficiencies,
            material_weaknesses=material_weaknesses,
            significant_deficiencies=significant_deficiencies,
            remediation_required=len(material_weaknesses) > 0,
            management_certification_required=len(material_weaknesses) > 0
        )
    
    def _check_section_302_compliance(
        self,
        financial_assessment: FinancialDataAssessment,
        fiscal_period: str
    ) -> List[str]:
        """Check SOX Section 302 compliance (Corporate responsibility)"""
        
        violations = []
        
        if financial_assessment.contains_financial_data:
            violations.extend([
                "CEO/CFO certification required for financial reports",
                "Management must establish and maintain disclosure controls",
                "Quarterly evaluation of disclosure controls effectiveness required"
            ])
            
            if financial_assessment.is_material_information:
                violations.append(
                    "Material changes in internal controls must be disclosed"
                )
        
        return violations
    
    def _check_section_404_compliance(
        self,
        control_assessment: InternalControlAssessment,
        financial_assessment: FinancialDataAssessment
    ) -> List[str]:
        """Check SOX Section 404 compliance (Internal controls)"""
        
        violations = []
        
        if financial_assessment.contains_financial_data:
            if not control_assessment.controls_documented:
                violations.append(
                    "Management must document internal controls over financial reporting"
                )
            
            if control_assessment.material_weaknesses:
                violations.append(
                    "Material weaknesses in internal controls must be remediated"
                )
            
            violations.append(
                "Annual assessment of internal controls effectiveness required"
            )
        
        return violations
    
    def _check_section_409_compliance(
        self,
        financial_assessment: FinancialDataAssessment,
        disclosure_deadline: Optional[datetime]
    ) -> List[str]:
        """Check SOX Section 409 compliance (Real-time disclosure)"""
        
        violations = []
        
        if financial_assessment.is_material_information:
            if not disclosure_deadline:
                violations.append(
                    "Material information requires timely disclosure (within 4 business days)"
                )
            elif disclosure_deadline < datetime.now():
                violations.append(
                    "Material information disclosure deadline has passed"
                )
        
        return violations
    
    def _check_document_retention_requirements(
        self,
        financial_assessment: FinancialDataAssessment
    ) -> List[str]:
        """Check SOX document retention requirements"""
        
        warnings = []
        
        if financial_assessment.contains_financial_data:
            warnings.extend([
                f"Financial records must be retained for {financial_assessment.retention_period.value}",
                "Audit workpapers must be retained for 7 years",
                "Document retention policy must be established and followed"
            ])
        
        return warnings
    
    def _determine_sensitivity_level(
        self,
        data_types: List[FinancialDataType],
        is_material: bool
    ) -> DataSensitivityLevel:
        """Determine data sensitivity level under SOX"""
        
        if is_material:
            return DataSensitivityLevel.MATERIAL
        elif any(dt in [FinancialDataType.REVENUE_DATA, FinancialDataType.CASH_FLOW_DATA] 
                for dt in data_types):
            return DataSensitivityLevel.CONFIDENTIAL
        elif data_types:
            return DataSensitivityLevel.INTERNAL
        else:
            return DataSensitivityLevel.PUBLIC
    
    def _requires_disclosure(
        self,
        contains_financial_data: bool,
        is_material: bool,
        sensitivity_level: DataSensitivityLevel
    ) -> bool:
        """Determine if data requires public disclosure"""
        
        return (
            contains_financial_data and 
            (is_material or sensitivity_level == DataSensitivityLevel.MATERIAL)
        )
    
    def _determine_retention_period(
        self,
        data_types: List[FinancialDataType]
    ) -> RetentionRequirement:
        """Determine required retention period"""
        
        if any(dt in [FinancialDataType.REVENUE_DATA, FinancialDataType.ASSET_DATA] 
               for dt in data_types):
            return RetentionRequirement.SEVEN_YEARS
        elif data_types:
            return RetentionRequirement.FIVE_YEARS
        else:
            return RetentionRequirement.THREE_YEARS
    
    def _identify_control_objectives(
        self,
        data_types: List[FinancialDataType]
    ) -> List[ControlObjective]:
        """Identify affected control objectives"""
        
        objectives = [ControlObjective.EXISTENCE_OCCURRENCE, ControlObjective.COMPLETENESS]
        
        if any(dt in [FinancialDataType.REVENUE_DATA, FinancialDataType.ASSET_DATA] 
               for dt in data_types):
            objectives.append(ControlObjective.ACCURACY_VALUATION)
        
        if data_types:
            objectives.extend([
                ControlObjective.CLASSIFICATION,
                ControlObjective.PRESENTATION_DISCLOSURE
            ])
        
        return objectives
    
    def _determine_disclosure_timeline(
        self,
        financial_assessment: FinancialDataAssessment,
        fiscal_period: str,
        disclosure_deadline: Optional[datetime]
    ) -> Optional[datetime]:
        """Determine disclosure timeline requirements"""
        
        if financial_assessment.is_material_information:
            if disclosure_deadline:
                return disclosure_deadline
            else:
                # Material information must be disclosed within 4 business days
                return datetime.now() + timedelta(days=4)
        
        return None
    
    def _generate_financial_data_recommendations(
        self,
        contains_financial_data: bool,
        data_types: List[FinancialDataType],
        is_material: bool,
        sensitivity_level: DataSensitivityLevel
    ) -> List[str]:
        """Generate recommendations for financial data handling"""
        
        recommendations = []
        
        if contains_financial_data:
            recommendations.append("Implement SOX controls for financial data protection")
            
            if is_material:
                recommendations.extend([
                    "Material information requires timely disclosure",
                    "Implement additional controls for material information"
                ])
            
            if sensitivity_level in [DataSensitivityLevel.CONFIDENTIAL, DataSensitivityLevel.MATERIAL]:
                recommendations.extend([
                    "Restrict access to confidential financial data",
                    "Implement audit trails for all data access"
                ])
        else:
            recommendations.append("No financial data detected - standard SOX practices apply")
        
        return recommendations
    
    def _assess_overall_sox_risk(
        self,
        financial_assessment: FinancialDataAssessment,
        control_assessment: InternalControlAssessment,
        violations: List[str],
        warnings: List[str]
    ) -> str:
        """Assess overall SOX compliance risk"""
        
        if control_assessment.material_weaknesses:
            return "CRITICAL"
        
        if len(violations) >= 3:
            return "CRITICAL"
        elif financial_assessment.is_material_information and violations:
            return "HIGH"
        elif len(violations) >= 1:
            return "MEDIUM"
        elif financial_assessment.contains_financial_data:
            return "MEDIUM"
        else:
            return "LOW"
    
    def log_audit_trail(
        self,
        user_id: str,
        action: str,
        data_affected: str,
        before_value: Optional[str] = None,
        after_value: Optional[str] = None,
        business_justification: str = "",
        approval_required: bool = False,
        approved_by: Optional[str] = None
    ) -> str:
        """Log audit trail entry for SOX compliance"""
        
        entry = AuditTrailEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            data_affected=data_affected,
            before_value=before_value,
            after_value=after_value,
            business_justification=business_justification,
            approval_required=approval_required,
            approved_by=approved_by
        )
        
        self.audit_trail.append(entry)
        
        self.logger.info(
            f"Audit trail logged: {entry.entry_id} - {action} by {user_id}"
        )
        
        return entry.entry_id
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive SOX compliance report"""
        
        return {
            "sox_sections_monitored": [section.value for section in SOXSection],
            "financial_data_types": [dtype.value for dtype in FinancialDataType],
            "control_objectives": [obj.value for obj in ControlObjective],
            "retention_requirements": [req.value for req in RetentionRequirement],
            "audit_trail_entries": len(self.audit_trail),
            "financial_indicators": self.financial_indicators
        }