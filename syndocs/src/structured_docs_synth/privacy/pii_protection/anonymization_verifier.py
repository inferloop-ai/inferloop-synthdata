"""
Anonymization verification module for validating privacy preservation effectiveness
"""

import re
import hashlib
import statistics
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import difflib
from collections import Counter

from ...core import get_logger, PrivacyError
from .pii_detector import PIIDetector, PIIDetectionResult, PIIType


class AnonymizationMethod(Enum):
    """Types of anonymization methods"""
    MASKING = "masking"
    GENERALIZATION = "generalization"
    SUPPRESSION = "suppression"
    SUBSTITUTION = "substitution"
    PERTURBATION = "perturbation"
    TOKENIZATION = "tokenization"


class RiskLevel(Enum):
    """Re-identification risk levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AnonymizationResult:
    """Result of anonymization process"""
    original_data: Dict[str, Any]
    anonymized_data: Dict[str, Any]
    method_used: AnonymizationMethod
    fields_processed: List[str]
    timestamp: str


@dataclass
class ReidentificationRisk:
    """Assessment of re-identification risk"""
    risk_level: RiskLevel
    confidence: float
    risk_factors: List[str]
    unique_combinations: int
    k_anonymity: int
    l_diversity: float
    t_closeness: float


@dataclass
class AnonymizationQuality:
    """Quality metrics for anonymization"""
    data_utility: float  # 0-1 scale
    information_loss: float  # 0-1 scale
    semantic_similarity: float  # 0-1 scale
    structural_preservation: float  # 0-1 scale
    overall_quality: float  # 0-1 scale


@dataclass
class VerificationResult:
    """Complete verification result"""
    is_adequately_anonymized: bool
    reidentification_risk: ReidentificationRisk
    quality_metrics: AnonymizationQuality
    residual_pii: PIIDetectionResult
    recommendations: List[str]
    compliance_status: Dict[str, bool]


class AnonymizationVerifier:
    """Verifier for anonymization effectiveness and privacy preservation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pii_detector = PIIDetector()
        
        # Risk thresholds for different privacy standards
        self.risk_thresholds = {
            "gdpr": {"max_reident_risk": 0.05, "min_k_anonymity": 3},
            "hipaa": {"max_reident_risk": 0.02, "min_k_anonymity": 5},
            "pci_dss": {"max_reident_risk": 0.01, "min_k_anonymity": 10},
            "sox": {"max_reident_risk": 0.03, "min_k_anonymity": 3}
        }
        
        self.logger.info("Anonymization Verifier initialized")
    
    def verify_anonymization(
        self,
        original_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        anonymized_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        method_used: AnonymizationMethod,
        compliance_standards: List[str] = None
    ) -> VerificationResult:
        """
        Comprehensive verification of anonymization effectiveness
        """
        
        try:
            if compliance_standards is None:
                compliance_standards = ["gdpr"]
            
            # Convert single records to lists for uniform processing
            if isinstance(original_data, dict):
                original_data = [original_data]
                anonymized_data = [anonymized_data]
            
            # 1. Check for residual PII
            residual_pii = self._detect_residual_pii(anonymized_data)
            
            # 2. Assess re-identification risk
            reident_risk = self._assess_reidentification_risk(
                original_data, anonymized_data, method_used
            )
            
            # 3. Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                original_data, anonymized_data
            )
            
            # 4. Check compliance with standards
            compliance_status = self._check_compliance(
                reident_risk, residual_pii, compliance_standards
            )
            
            # 5. Generate recommendations
            recommendations = self._generate_recommendations(
                reident_risk, quality_metrics, residual_pii, compliance_status
            )
            
            # 6. Overall assessment
            is_adequate = self._assess_overall_adequacy(
                reident_risk, residual_pii, compliance_status
            )
            
            result = VerificationResult(
                is_adequately_anonymized=is_adequate,
                reidentification_risk=reident_risk,
                quality_metrics=quality_metrics,
                residual_pii=residual_pii,
                recommendations=recommendations,
                compliance_status=compliance_status
            )
            
            self.logger.info(
                f"Anonymization verification complete: "
                f"Adequate={is_adequate}, Risk={reident_risk.risk_level.value}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during anonymization verification: {str(e)}")
            raise PrivacyError(
                f"Anonymization verification failed: {str(e)}",
                details={"method": method_used.value}
            )
    
    def _detect_residual_pii(self, anonymized_data: List[Dict[str, Any]]) -> PIIDetectionResult:
        """Detect any remaining PII in anonymized data"""
        
        all_matches = []
        all_pii_types = set()
        
        for record in anonymized_data:
            detection_results = self.pii_detector.detect_pii_in_document(record)
            
            for field_name, result in detection_results.items():
                all_matches.extend(result.matches)
                all_pii_types.update(result.pii_types_found)
        
        # Create combined result
        risk_level = "CRITICAL" if len(all_matches) > 0 else "LOW"
        
        combined_result = PIIDetectionResult(
            text="Combined anonymized data",
            matches=all_matches,
            total_matches=len(all_matches),
            pii_types_found=all_pii_types,
            has_pii=len(all_matches) > 0,
            risk_level=risk_level
        )
        
        return combined_result
    
    def _assess_reidentification_risk(
        self,
        original_data: List[Dict[str, Any]],
        anonymized_data: List[Dict[str, Any]],
        method_used: AnonymizationMethod
    ) -> ReidentificationRisk:
        """Assess the risk of re-identification"""
        
        # Calculate k-anonymity
        k_anonymity = self._calculate_k_anonymity(anonymized_data)
        
        # Calculate l-diversity
        l_diversity = self._calculate_l_diversity(anonymized_data)
        
        # Calculate t-closeness
        t_closeness = self._calculate_t_closeness(original_data, anonymized_data)
        
        # Count unique combinations
        unique_combinations = self._count_unique_combinations(anonymized_data)
        
        # Assess risk factors
        risk_factors = self._identify_risk_factors(
            anonymized_data, k_anonymity, l_diversity, method_used
        )
        
        # Calculate overall risk level and confidence
        risk_level, confidence = self._calculate_overall_risk(
            k_anonymity, l_diversity, t_closeness, unique_combinations, 
            len(anonymized_data), risk_factors
        )
        
        return ReidentificationRisk(
            risk_level=risk_level,
            confidence=confidence,
            risk_factors=risk_factors,
            unique_combinations=unique_combinations,
            k_anonymity=k_anonymity,
            l_diversity=l_diversity,
            t_closeness=t_closeness
        )
    
    def _calculate_k_anonymity(self, data: List[Dict[str, Any]]) -> int:
        """Calculate k-anonymity: minimum group size for any combination of quasi-identifiers"""
        
        if not data:
            return 0
        
        # Identify potential quasi-identifiers (non-sensitive attributes)
        quasi_identifiers = self._identify_quasi_identifiers(data)
        
        if not quasi_identifiers:
            return len(data)  # If no quasi-identifiers, all records are identical
        
        # Group records by quasi-identifier combinations
        groups = {}
        for record in data:
            key = tuple(str(record.get(qi, "")) for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = 0
            groups[key] += 1
        
        # Return minimum group size
        return min(groups.values()) if groups else 0
    
    def _calculate_l_diversity(self, data: List[Dict[str, Any]]) -> float:
        """Calculate l-diversity: diversity of sensitive attributes within equivalence classes"""
        
        if not data:
            return 0.0
        
        # Identify sensitive attributes
        sensitive_attrs = self._identify_sensitive_attributes(data)
        
        if not sensitive_attrs:
            return 1.0  # No sensitive attributes
        
        quasi_identifiers = self._identify_quasi_identifiers(data)
        
        # Group by quasi-identifiers
        groups = {}
        for record in data:
            qi_key = tuple(str(record.get(qi, "")) for qi in quasi_identifiers)
            if qi_key not in groups:
                groups[qi_key] = []
            groups[qi_key].append(record)
        
        # Calculate diversity for each group
        diversities = []
        for group in groups.values():
            for sensitive_attr in sensitive_attrs:
                values = [record.get(sensitive_attr) for record in group if record.get(sensitive_attr)]
                unique_values = len(set(values))
                if len(values) > 0:
                    diversities.append(unique_values)
        
        return statistics.mean(diversities) if diversities else 0.0
    
    def _calculate_t_closeness(
        self, 
        original_data: List[Dict[str, Any]], 
        anonymized_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate t-closeness: similarity between original and anonymized distributions"""
        
        if not original_data or not anonymized_data:
            return 1.0
        
        sensitive_attrs = self._identify_sensitive_attributes(original_data)
        
        if not sensitive_attrs:
            return 0.0
        
        closeness_scores = []
        
        for attr in sensitive_attrs:
            # Get distributions
            orig_values = [r.get(attr) for r in original_data if r.get(attr) is not None]
            anon_values = [r.get(attr) for r in anonymized_data if r.get(attr) is not None]
            
            if orig_values and anon_values:
                # Calculate distribution similarity (simplified)
                orig_counter = Counter(orig_values)
                anon_counter = Counter(anon_values)
                
                all_values = set(orig_values + anon_values)
                
                orig_dist = [orig_counter.get(v, 0) / len(orig_values) for v in all_values]
                anon_dist = [anon_counter.get(v, 0) / len(anon_values) for v in all_values]
                
                # Earth Mover's Distance approximation
                emd = sum(abs(o - a) for o, a in zip(orig_dist, anon_dist)) / 2
                closeness_scores.append(1 - emd)
        
        return statistics.mean(closeness_scores) if closeness_scores else 0.0
    
    def _count_unique_combinations(self, data: List[Dict[str, Any]]) -> int:
        """Count unique combinations of all attributes"""
        
        if not data:
            return 0
        
        unique_records = set()
        for record in data:
            # Create a hashable representation of the record
            record_tuple = tuple(sorted(record.items()))
            unique_records.add(record_tuple)
        
        return len(unique_records)
    
    def _identify_quasi_identifiers(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify potential quasi-identifier fields"""
        
        if not data:
            return []
        
        # Common quasi-identifier patterns
        qi_patterns = [
            r'.*age.*', r'.*zip.*', r'.*postal.*', r'.*location.*',
            r'.*city.*', r'.*state.*', r'.*country.*', r'.*gender.*',
            r'.*birth.*', r'.*date.*', r'.*year.*', r'.*education.*',
            r'.*occupation.*', r'.*job.*', r'.*income.*', r'.*salary.*'
        ]
        
        potential_qis = []
        sample_record = data[0]
        
        for field in sample_record.keys():
            field_lower = field.lower()
            if any(re.match(pattern, field_lower) for pattern in qi_patterns):
                potential_qis.append(field)
        
        return potential_qis
    
    def _identify_sensitive_attributes(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify sensitive attribute fields"""
        
        if not data:
            return []
        
        # Common sensitive attribute patterns
        sensitive_patterns = [
            r'.*ssn.*', r'.*social.*', r'.*credit.*', r'.*card.*',
            r'.*medical.*', r'.*health.*', r'.*diagnosis.*', r'.*treatment.*',
            r'.*financial.*', r'.*bank.*', r'.*account.*', r'.*salary.*',
            r'.*political.*', r'.*religion.*', r'.*orientation.*'
        ]
        
        sensitive_attrs = []
        sample_record = data[0]
        
        for field in sample_record.keys():
            field_lower = field.lower()
            if any(re.match(pattern, field_lower) for pattern in sensitive_patterns):
                sensitive_attrs.append(field)
        
        return sensitive_attrs
    
    def _identify_risk_factors(
        self,
        data: List[Dict[str, Any]],
        k_anonymity: int,
        l_diversity: float,
        method_used: AnonymizationMethod
    ) -> List[str]:
        """Identify specific re-identification risk factors"""
        
        risk_factors = []
        
        # Low k-anonymity
        if k_anonymity < 3:
            risk_factors.append(f"Low k-anonymity ({k_anonymity})")
        
        # Low l-diversity
        if l_diversity < 2:
            risk_factors.append(f"Low l-diversity ({l_diversity:.2f})")
        
        # Small dataset size
        if len(data) < 100:
            risk_factors.append("Small dataset size")
        
        # Method-specific risks
        if method_used == AnonymizationMethod.MASKING:
            risk_factors.append("Masking may preserve patterns")
        elif method_used == AnonymizationMethod.GENERALIZATION:
            risk_factors.append("Generalization may create outliers")
        
        # High uniqueness
        unique_ratio = self._count_unique_combinations(data) / len(data) if data else 0
        if unique_ratio > 0.8:
            risk_factors.append("High record uniqueness")
        
        return risk_factors
    
    def _calculate_overall_risk(
        self,
        k_anonymity: int,
        l_diversity: float,
        t_closeness: float,
        unique_combinations: int,
        total_records: int,
        risk_factors: List[str]
    ) -> Tuple[RiskLevel, float]:
        """Calculate overall re-identification risk level and confidence"""
        
        risk_score = 0.0
        
        # K-anonymity contribution (0-40 points)
        if k_anonymity == 0:
            risk_score += 40
        elif k_anonymity < 3:
            risk_score += 30
        elif k_anonymity < 5:
            risk_score += 20
        elif k_anonymity < 10:
            risk_score += 10
        
        # L-diversity contribution (0-30 points)
        if l_diversity < 1:
            risk_score += 30
        elif l_diversity < 2:
            risk_score += 20
        elif l_diversity < 3:
            risk_score += 10
        
        # T-closeness contribution (0-20 points)
        if t_closeness < 0.3:
            risk_score += 20
        elif t_closeness < 0.5:
            risk_score += 15
        elif t_closeness < 0.7:
            risk_score += 10
        
        # Uniqueness contribution (0-10 points)
        uniqueness_ratio = unique_combinations / total_records if total_records > 0 else 1
        if uniqueness_ratio > 0.9:
            risk_score += 10
        elif uniqueness_ratio > 0.7:
            risk_score += 5
        
        # Risk factors penalty
        risk_score += len(risk_factors) * 2
        
        # Convert to risk level
        if risk_score >= 80:
            risk_level = RiskLevel.VERY_HIGH
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 40:
            risk_level = RiskLevel.MEDIUM
        elif risk_score >= 20:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.VERY_LOW
        
        # Confidence based on data size and factors
        confidence = min(0.95, 0.5 + (min(total_records, 1000) / 2000) + (min(len(risk_factors), 5) / 10))
        
        return risk_level, confidence
    
    def _calculate_quality_metrics(
        self,
        original_data: List[Dict[str, Any]],
        anonymized_data: List[Dict[str, Any]]
    ) -> AnonymizationQuality:
        """Calculate data quality metrics after anonymization"""
        
        if not original_data or not anonymized_data:
            return AnonymizationQuality(0, 1, 0, 0, 0)
        
        # Data utility: how much useful information is preserved
        data_utility = self._calculate_data_utility(original_data, anonymized_data)
        
        # Information loss: how much information was removed
        information_loss = self._calculate_information_loss(original_data, anonymized_data)
        
        # Semantic similarity: how similar the data remains
        semantic_similarity = self._calculate_semantic_similarity(original_data, anonymized_data)
        
        # Structural preservation: how well structure is maintained
        structural_preservation = self._calculate_structural_preservation(original_data, anonymized_data)
        
        # Overall quality score
        overall_quality = statistics.mean([
            data_utility, 
            1 - information_loss, 
            semantic_similarity, 
            structural_preservation
        ])
        
        return AnonymizationQuality(
            data_utility=data_utility,
            information_loss=information_loss,
            semantic_similarity=semantic_similarity,
            structural_preservation=structural_preservation,
            overall_quality=overall_quality
        )
    
    def _calculate_data_utility(
        self, 
        original_data: List[Dict[str, Any]], 
        anonymized_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate how much useful information is preserved"""
        
        if len(original_data) != len(anonymized_data):
            return 0.0
        
        preserved_fields = 0
        total_fields = 0
        
        for orig, anon in zip(original_data, anonymized_data):
            for field in orig.keys():
                total_fields += 1
                if field in anon and anon[field] is not None and str(anon[field]).strip():
                    preserved_fields += 1
        
        return preserved_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_information_loss(
        self, 
        original_data: List[Dict[str, Any]], 
        anonymized_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate the amount of information lost"""
        
        if not original_data or not anonymized_data:
            return 1.0
        
        # Calculate entropy loss
        total_loss = 0.0
        field_count = 0
        
        # Get all fields
        all_fields = set()
        for record in original_data + anonymized_data:
            all_fields.update(record.keys())
        
        for field in all_fields:
            orig_values = [r.get(field) for r in original_data if r.get(field) is not None]
            anon_values = [r.get(field) for r in anonymized_data if r.get(field) is not None]
            
            if orig_values:
                field_count += 1
                
                # Calculate entropy
                orig_entropy = self._calculate_entropy(orig_values)
                anon_entropy = self._calculate_entropy(anon_values) if anon_values else 0
                
                # Information loss for this field
                if orig_entropy > 0:
                    field_loss = 1 - (anon_entropy / orig_entropy)
                    total_loss += max(0, field_loss)
        
        return total_loss / field_count if field_count > 0 else 0.0
    
    def _calculate_entropy(self, values: List[Any]) -> float:
        """Calculate Shannon entropy of a list of values"""
        
        if not values:
            return 0.0
        
        counter = Counter(values)
        probabilities = [count / len(values) for count in counter.values()]
        
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * (p.bit_length() - 1)  # Approximation of log2(p)
        
        return entropy
    
    def _calculate_semantic_similarity(
        self, 
        original_data: List[Dict[str, Any]], 
        anonymized_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate semantic similarity between original and anonymized data"""
        
        if len(original_data) != len(anonymized_data):
            return 0.0
        
        similarities = []
        
        for orig, anon in zip(original_data, anonymized_data):
            record_similarity = self._calculate_record_similarity(orig, anon)
            similarities.append(record_similarity)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    def _calculate_record_similarity(self, orig_record: Dict[str, Any], anon_record: Dict[str, Any]) -> float:
        """Calculate similarity between two records"""
        
        all_fields = set(orig_record.keys()) | set(anon_record.keys())
        if not all_fields:
            return 1.0
        
        field_similarities = []
        
        for field in all_fields:
            orig_value = str(orig_record.get(field, ""))
            anon_value = str(anon_record.get(field, ""))
            
            # Use string similarity
            similarity = difflib.SequenceMatcher(None, orig_value, anon_value).ratio()
            field_similarities.append(similarity)
        
        return statistics.mean(field_similarities)
    
    def _calculate_structural_preservation(
        self, 
        original_data: List[Dict[str, Any]], 
        anonymized_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well the data structure is preserved"""
        
        if not original_data or not anonymized_data:
            return 0.0
        
        # Check field preservation
        orig_fields = set(original_data[0].keys()) if original_data else set()
        anon_fields = set(anonymized_data[0].keys()) if anonymized_data else set()
        
        field_preservation = len(orig_fields & anon_fields) / len(orig_fields | anon_fields) if orig_fields or anon_fields else 1.0
        
        # Check record count preservation
        count_preservation = min(len(anonymized_data), len(original_data)) / max(len(original_data), 1)
        
        # Check data type preservation
        type_preservation = self._calculate_type_preservation(original_data, anonymized_data)
        
        return statistics.mean([field_preservation, count_preservation, type_preservation])
    
    def _calculate_type_preservation(
        self, 
        original_data: List[Dict[str, Any]], 
        anonymized_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well data types are preserved"""
        
        if not original_data or not anonymized_data:
            return 0.0
        
        type_matches = 0
        total_fields = 0
        
        orig_sample = original_data[0]
        anon_sample = anonymized_data[0]
        
        for field in orig_sample.keys():
            if field in anon_sample:
                total_fields += 1
                orig_type = type(orig_sample[field])
                anon_type = type(anon_sample[field])
                
                if orig_type == anon_type:
                    type_matches += 1
        
        return type_matches / total_fields if total_fields > 0 else 0.0
    
    def _check_compliance(
        self,
        reident_risk: ReidentificationRisk,
        residual_pii: PIIDetectionResult,
        standards: List[str]
    ) -> Dict[str, bool]:
        """Check compliance with privacy standards"""
        
        compliance_status = {}
        
        for standard in standards:
            if standard.lower() in self.risk_thresholds:
                thresholds = self.risk_thresholds[standard.lower()]
                
                # Check re-identification risk
                risk_acceptable = reident_risk.risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]
                
                # Check k-anonymity
                k_acceptable = reident_risk.k_anonymity >= thresholds["min_k_anonymity"]
                
                # Check for residual PII
                no_residual_pii = not residual_pii.has_pii
                
                compliance_status[standard] = risk_acceptable and k_acceptable and no_residual_pii
            else:
                compliance_status[standard] = False
        
        return compliance_status
    
    def _generate_recommendations(
        self,
        reident_risk: ReidentificationRisk,
        quality_metrics: AnonymizationQuality,
        residual_pii: PIIDetectionResult,
        compliance_status: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations for improving anonymization"""
        
        recommendations = []
        
        # Re-identification risk recommendations
        if reident_risk.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            recommendations.append("CRITICAL: High re-identification risk detected")
            recommendations.append("Consider stronger anonymization methods")
            
            if reident_risk.k_anonymity < 3:
                recommendations.append(f"Increase k-anonymity (current: {reident_risk.k_anonymity})")
            
            if reident_risk.l_diversity < 2:
                recommendations.append(f"Improve l-diversity (current: {reident_risk.l_diversity:.2f})")
        
        # Residual PII recommendations
        if residual_pii.has_pii:
            recommendations.append(f"Remove residual PII: {len(residual_pii.matches)} items detected")
            pii_types = [match.pii_type.value for match in residual_pii.matches]
            recommendations.append(f"PII types found: {', '.join(set(pii_types))}")
        
        # Quality recommendations
        if quality_metrics.data_utility < 0.7:
            recommendations.append("Low data utility - consider less aggressive anonymization")
        
        if quality_metrics.information_loss > 0.5:
            recommendations.append("High information loss - review anonymization strategy")
        
        # Compliance recommendations
        non_compliant = [std for std, compliant in compliance_status.items() if not compliant]
        if non_compliant:
            recommendations.append(f"Non-compliant with: {', '.join(non_compliant)}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Anonymization appears adequate")
            recommendations.append("Continue monitoring for emerging risks")
        
        return recommendations
    
    def _assess_overall_adequacy(
        self,
        reident_risk: ReidentificationRisk,
        residual_pii: PIIDetectionResult,
        compliance_status: Dict[str, bool]
    ) -> bool:
        """Assess overall adequacy of anonymization"""
        
        # Must not have high re-identification risk
        risk_acceptable = reident_risk.risk_level not in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        
        # Must not have residual PII
        no_residual_pii = not residual_pii.has_pii
        
        # Must be compliant with at least one standard
        has_compliance = any(compliance_status.values()) if compliance_status else True
        
        return risk_acceptable and no_residual_pii and has_compliance