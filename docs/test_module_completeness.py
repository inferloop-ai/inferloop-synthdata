#!/usr/bin/env python3
"""
Test to check completeness of all 12 privacy modules
"""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "docs/structured-documents-synthetic-data/src"))

def test_module_imports():
    """Test all module imports to identify missing components"""
    
    print("🔍 Testing Module Completeness")
    print("=" * 60)
    
    missing_components = []
    
    # Test 1: PII Protection Module
    print("\n🔹 Testing PII Protection Module")
    print("-" * 30)
    
    try:
        from structured_docs_synth.privacy.pii_protection import (
            PIIDetector, PIIType, PIIMatch, PIIDetectionResult,
            AnonymizationVerifier, AnonymizationMethod, RiskLevel, VerificationResult,
            MaskingStrategies, MaskingMethod, MaskingRule, MaskingResult
        )
        print("✅ PII Protection - All imports successful")
    except ImportError as e:
        print(f"❌ PII Protection - Missing: {e}")
        missing_components.append(f"PII Protection: {e}")
    
    # Test 2: Compliance Module
    print("\n🔹 Testing Compliance Module")
    print("-" * 30)
    
    try:
        from structured_docs_synth.privacy.compliance import (
            GDPREnforcer, GDPRAssessment, LawfulBasis, ProcessingPurpose,
            HIPAAEnforcer, HIPAAAssessment, PHIIdentifier, PHIAssessment,
            PCIDSSEnforcer, PCIDSSAssessment, CardholderDataElement, 
            SensitiveAuthenticationData, ComplianceLevel, CardholderDataAssessment,
            SOXEnforcer, SOXAssessment, FinancialDataType, SOXSection,
            DataSensitivityLevel, FinancialDataAssessment,
            AuditLogger, AuditRecord, AuditEventType, AuditSeverity,
            ComplianceFramework, AuditContext, DataSubject, AuditQuery, AuditSummary
        )
        print("✅ Compliance - All imports successful")
    except ImportError as e:
        print(f"❌ Compliance - Missing: {e}")
        missing_components.append(f"Compliance: {e}")
    
    # Test 3: Differential Privacy Module
    print("\n🔹 Testing Differential Privacy Module")
    print("-" * 30)
    
    try:
        from structured_docs_synth.privacy.differential_privacy import (
            LaplaceMechanism, LaplaceNoise, NoiseParameters,
            ExponentialMechanism, UtilityFunction, ExponentialResult,
            PrivacyBudgetTracker, BudgetAllocation, BudgetStatus,
            CompositionAnalyzer, CompositionResult, PrivacyAccountant
        )
        print("✅ Differential Privacy - All imports successful")
    except ImportError as e:
        print(f"❌ Differential Privacy - Missing: {e}")
        missing_components.append(f"Differential Privacy: {e}")
    
    # Test 4: Main Privacy Module
    print("\n🔹 Testing Main Privacy Module")
    print("-" * 30)
    
    try:
        from structured_docs_synth.privacy import (
            # PII Protection
            PIIDetector, PIIType, PIIMatch, PIIDetectionResult,
            AnonymizationVerifier, AnonymizationMethod, RiskLevel, VerificationResult,
            MaskingStrategies, MaskingMethod, MaskingRule, MaskingResult,
            
            # Compliance
            GDPREnforcer, GDPRAssessment, LawfulBasis, ProcessingPurpose,
            HIPAAEnforcer, HIPAAAssessment, PHIIdentifier, PHIAssessment,
            PCIDSSEnforcer, PCIDSSAssessment, CardholderDataElement, 
            SensitiveAuthenticationData, ComplianceLevel,
            SOXEnforcer, SOXAssessment, FinancialDataType, SOXSection, DataSensitivityLevel,
            AuditLogger, AuditRecord, AuditEventType, AuditSeverity, AuditContext, 
            DataSubject, AuditQuery, AuditSummary, ComplianceFramework,
            
            # Differential Privacy
            LaplaceMechanism, LaplaceNoise, NoiseParameters,
            ExponentialMechanism, UtilityFunction, ExponentialResult,
            PrivacyBudgetTracker, BudgetAllocation, BudgetStatus,
            CompositionAnalyzer, CompositionResult, PrivacyAccountant
        )
        print("✅ Main Privacy Module - All imports successful")
    except ImportError as e:
        print(f"❌ Main Privacy Module - Missing: {e}")
        missing_components.append(f"Main Privacy Module: {e}")
    
    # Test 5: Individual File Completeness
    print("\n🔹 Testing Individual File Completeness")
    print("-" * 30)
    
    file_tests = [
        ("PIIDetector", "structured_docs_synth.privacy.pii_protection.pii_detector", "PIIDetector"),
        ("AnonymizationVerifier", "structured_docs_synth.privacy.pii_protection.anonymization_verifier", "AnonymizationVerifier"),
        ("MaskingStrategies", "structured_docs_synth.privacy.pii_protection.masking_strategies", "MaskingStrategies"),
        ("GDPREnforcer", "structured_docs_synth.privacy.compliance.gdpr_enforcer", "GDPREnforcer"),
        ("HIPAAEnforcer", "structured_docs_synth.privacy.compliance.hipaa_enforcer", "HIPAAEnforcer"),
        ("PCIDSSEnforcer", "structured_docs_synth.privacy.compliance.pci_dss_enforcer", "PCIDSSEnforcer"),
        ("SOXEnforcer", "structured_docs_synth.privacy.compliance.sox_enforcer", "SOXEnforcer"),
        ("AuditLogger", "structured_docs_synth.privacy.compliance.audit_logger", "AuditLogger"),
        ("LaplaceMechanism", "structured_docs_synth.privacy.differential_privacy.laplace_mechanism", "LaplaceMechanism"),
        ("ExponentialMechanism", "structured_docs_synth.privacy.differential_privacy.exponential_mechanism", "ExponentialMechanism"),
        ("PrivacyBudgetTracker", "structured_docs_synth.privacy.differential_privacy.privacy_budget", "PrivacyBudgetTracker"),
        ("CompositionAnalyzer", "structured_docs_synth.privacy.differential_privacy.composition_analyzer", "CompositionAnalyzer"),
    ]
    
    for test_name, module_path, class_name in file_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {test_name} - Class available")
        except ImportError as e:
            print(f"❌ {test_name} - Import error: {e}")
            missing_components.append(f"{test_name}: Import error - {e}")
        except AttributeError as e:
            print(f"❌ {test_name} - Class missing: {e}")
            missing_components.append(f"{test_name}: Class missing - {e}")
    
    # Summary
    print(f"\n🎯 Module Completeness Summary")
    print("-" * 40)
    
    if not missing_components:
        print("🟢 ALL MODULES COMPLETE - No missing components found!")
        print("📊 Module Statistics:")
        print("  • PII Protection: 3 main classes + supporting types")
        print("  • Compliance: 4 enforcers + audit logger + supporting types")
        print("  • Differential Privacy: 4 mechanisms + supporting types")
        print("  • Total: 12 main modules with full type support")
        return True
    else:
        print(f"🔴 MISSING COMPONENTS FOUND: {len(missing_components)}")
        for i, component in enumerate(missing_components, 1):
            print(f"  {i}. {component}")
        return False

if __name__ == "__main__":
    success = test_module_imports()
    sys.exit(0 if success else 1)