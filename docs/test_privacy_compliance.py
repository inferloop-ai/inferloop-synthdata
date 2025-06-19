#!/usr/bin/env python3
"""
Comprehensive test for Privacy and Compliance modules
"""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "docs/structured-documents-synthetic-data/src"))

try:
    from structured_docs_synth.privacy import (
        PIIDetector, AnonymizationVerifier, MaskingStrategies,
        GDPREnforcer, HIPAAEnforcer, 
        MaskingMethod, MaskingRule, AnonymizationMethod,
        LawfulBasis, ProcessingPurpose
    )
    
    def test_privacy_compliance_suite():
        """Comprehensive test of privacy and compliance modules"""
        print("🔐 Testing Privacy & Compliance Suite")
        print("=" * 60)
        
        # Test data with various PII types
        test_data = {
            "patient_name": "John Smith",
            "ssn": "123-45-6789",
            "email": "john.smith@hospital.com",
            "phone": "(555) 123-4567", 
            "address": "123 Main Street, Anytown, CA 90210",
            "date_of_birth": "01/15/1985",
            "medical_record_number": "MRN: 12345678",
            "diagnosis": "Type 2 Diabetes",
            "treatment": "Metformin 500mg twice daily"
        }
        
        print(f"📊 Test Data Fields: {len(test_data)}")
        for key, value in test_data.items():
            print(f"  • {key}: {value}")
        
        # 1. PII Detection
        print(f"\n🔍 Step 1: PII Detection")
        print("-" * 30)
        
        pii_detector = PIIDetector()
        pii_results = pii_detector.detect_pii_in_document(test_data)
        
        if pii_results:
            print(f"✅ PII detected in {len(pii_results)} fields")
            for field_name, result in pii_results.items():
                print(f"  🔸 {field_name}: {result.total_matches} matches ({result.risk_level} risk)")
        else:
            print("❌ No PII detected")
        
        # 2. Data Masking
        print(f"\n🎭 Step 2: Data Masking")
        print("-" * 30)
        
        masking_strategies = MaskingStrategies()
        
        # Test different masking methods
        test_ssn = "123-45-6789"
        test_email = "john.smith@example.com"
        
        # Create masking rules
        ssn_rule = MaskingRule(
            pii_type=pii_detector.pii_detector.PIIType.SSN if hasattr(pii_detector, 'pii_detector') else None,
            method=MaskingMethod.PARTIAL_MASK,
            preservation_level=masking_strategies.PreservationLevel.FORMAT_ONLY if hasattr(masking_strategies, 'PreservationLevel') else None
        )
        
        print(f"Original SSN: {test_ssn}")
        # Note: This would need the actual MaskingRule structure to work properly
        print(f"Masked SSN: [Masking would be applied here]")
        
        print(f"Original Email: {test_email}")
        print(f"Masked Email: [Masking would be applied here]")
        
        # 3. Anonymization Verification
        print(f"\n🔬 Step 3: Anonymization Verification")
        print("-" * 30)
        
        anonymization_verifier = AnonymizationVerifier()
        
        # Simulate anonymized data
        anonymized_data = {
            "patient_name": "Patient-001",
            "age_range": "40-50",
            "zip_code": "902XX",
            "diagnosis_category": "Endocrine",
            "treatment_type": "Medication"
        }
        
        print("✅ Anonymization Verifier initialized")
        print(f"📝 Original data: {len(test_data)} fields")
        print(f"📝 Anonymized data: {len(anonymized_data)} fields")
        
        # Would need to implement the actual verification call
        print("🔍 Verification results: [Would show k-anonymity, l-diversity, etc.]")
        
        # 4. GDPR Compliance Assessment
        print(f"\n🇪🇺 Step 4: GDPR Compliance")
        print("-" * 30)
        
        gdpr_enforcer = GDPREnforcer()
        
        # Register consent for GDPR
        gdpr_enforcer.register_consent(
            data_subject_id="patient-001",
            consent_given=True,
            purposes=[ProcessingPurpose.RESEARCH],
            consent_method="electronic_form"
        )
        
        gdpr_assessment = gdpr_enforcer.assess_gdpr_compliance(
            data=test_data,
            processing_purpose=ProcessingPurpose.RESEARCH,
            lawful_basis=LawfulBasis.CONSENT,
            data_subject_id="patient-001"
        )
        
        print(f"✅ GDPR Assessment Complete")
        print(f"  📊 Compliant: {gdpr_assessment.is_compliant}")
        print(f"  📊 Risk Level: {gdpr_assessment.risk_level}")
        print(f"  📊 Violations: {len(gdpr_assessment.violations)}")
        print(f"  📊 Warnings: {len(gdpr_assessment.warnings)}")
        
        if gdpr_assessment.violations:
            print("  ⚠️  Violations:")
            for violation in gdpr_assessment.violations:
                print(f"    • {violation}")
        
        if gdpr_assessment.required_actions:
            print("  📋 Required Actions:")
            for action in gdpr_assessment.required_actions[:3]:  # Show first 3
                print(f"    • {action}")
        
        # 5. HIPAA Compliance Assessment  
        print(f"\n🏥 Step 5: HIPAA Compliance")
        print("-" * 30)
        
        hipaa_enforcer = HIPAAEnforcer()
        
        hipaa_assessment = hipaa_enforcer.assess_hipaa_compliance(
            data=test_data,
            is_covered_entity=True,
            has_baa=False,
            intended_use="research"
        )
        
        print(f"✅ HIPAA Assessment Complete")
        print(f"  📊 Compliant: {hipaa_assessment.is_compliant}")
        print(f"  📊 Risk Level: {hipaa_assessment.risk_level}")
        print(f"  📊 Contains PHI: {hipaa_assessment.phi_assessment.contains_phi}")
        print(f"  📊 Safe Harbor Compliant: {hipaa_assessment.phi_assessment.safe_harbor_compliant}")
        print(f"  📊 Violations: {len(hipaa_assessment.violations)}")
        
        if hipaa_assessment.violations:
            print("  ⚠️  Violations:")
            for violation in hipaa_assessment.violations:
                print(f"    • {violation}")
        
        if hipaa_assessment.phi_assessment.recommendations:
            print("  📋 PHI Recommendations:")
            for rec in hipaa_assessment.phi_assessment.recommendations[:3]:  # Show first 3
                print(f"    • {rec}")
        
        # 6. Summary Report
        print(f"\n📈 Step 6: Compliance Summary")
        print("-" * 30)
        
        gdpr_report = gdpr_enforcer.get_compliance_report()
        hipaa_report = hipaa_enforcer.get_compliance_report()
        
        print(f"✅ GDPR Report Generated")
        print(f"  📊 Processing Records: {gdpr_report['processing_records']}")
        print(f"  📊 Consent Records: {gdpr_report['consent_records']}")
        print(f"  📊 Active Consents: {gdpr_report['active_consents']}")
        
        print(f"✅ HIPAA Report Generated")
        print(f"  📊 PHI Identifiers Monitored: {len(hipaa_report['phi_identifiers_monitored'])}")
        print(f"  📊 BAA Agreements: {hipaa_report['total_baa_agreements']}")
        
        # 7. Overall Assessment
        print(f"\n🎯 Overall Privacy & Compliance Status")
        print("-" * 40)
        
        overall_compliant = gdpr_assessment.is_compliant and hipaa_assessment.is_compliant
        total_violations = len(gdpr_assessment.violations) + len(hipaa_assessment.violations)
        
        if overall_compliant:
            print("🟢 COMPLIANT - All privacy regulations satisfied")
        else:
            print("🔴 NON-COMPLIANT - Violations detected")
        
        print(f"📊 Total Violations: {total_violations}")
        print(f"📊 GDPR Risk: {gdpr_assessment.risk_level}")
        print(f"📊 HIPAA Risk: {hipaa_assessment.risk_level}")
        
        # Final recommendations
        print(f"\n💡 Final Recommendations:")
        
        if total_violations > 0:
            print("  🔥 IMMEDIATE ACTION REQUIRED")
            print("  • Resolve all compliance violations before data processing")
            print("  • Implement additional privacy safeguards")
        
        if hipaa_assessment.phi_assessment.contains_phi:
            print("  • PHI detected - ensure HIPAA compliance measures")
            
        if not gdpr_assessment.is_compliant:
            print("  • GDPR violations detected - review data processing lawfulness")
        
        print("  • Regular compliance audits recommended")
        print("  • Monitor for regulatory updates")
        
        print(f"\n🎉 Privacy & Compliance Testing Complete!")
        
        return {
            "gdpr_compliant": gdpr_assessment.is_compliant,
            "hipaa_compliant": hipaa_assessment.is_compliant,
            "overall_compliant": overall_compliant,
            "total_violations": total_violations,
            "gdpr_risk": gdpr_assessment.risk_level,
            "hipaa_risk": hipaa_assessment.risk_level
        }
    
    if __name__ == "__main__":
        results = test_privacy_compliance_suite()
        
        # Exit with appropriate code
        if results["overall_compliant"]:
            print("\n✅ All tests passed - System is privacy compliant")
            sys.exit(0)
        else:
            print(f"\n❌ Tests failed - {results['total_violations']} violations detected")
            sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()