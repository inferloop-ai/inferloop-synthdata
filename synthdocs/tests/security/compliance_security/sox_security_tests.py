"""Security tests for SOX compliance.

Tests security controls required for Sarbanes-Oxley Act including
internal controls, financial reporting integrity, audit trails, and
access controls for financial systems.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid

from structured_docs_synth.privacy.compliance.sox_enforcer import (
    SOXEnforcer,
    InternalControlsManager,
    FinancialReportingIntegrity,
    AuditTrailManager,
    AccessControlEnforcer,
    ChangeManagementSystem
)
from structured_docs_synth.core.exceptions import ComplianceError, SecurityError


class TestSOXInternalControls:
    """Test SOX internal controls implementation."""
    
    @pytest.fixture
    def sox_enforcer(self):
        """Provide SOX enforcer instance."""
        return SOXEnforcer()
    
    @pytest.fixture
    def controls_manager(self):
        """Provide internal controls manager."""
        return InternalControlsManager()
    
    @pytest.mark.security
    def test_segregation_of_duties(self, controls_manager):
        """Test segregation of duties enforcement."""
        # Define incompatible roles
        user_roles = {
            "user1": ["transaction_entry", "transaction_approval"],  # Violation
            "user2": ["payment_processing", "payment_reconciliation"],  # Violation
            "user3": ["data_entry", "data_review"]  # Acceptable
        }
        
        # Check for violations
        violations = controls_manager.check_segregation_violations(user_roles)
        
        assert len(violations) == 2
        assert "user1" in violations
        assert "user2" in violations
        assert "user3" not in violations
        
        # Verify specific violations
        assert violations["user1"]["incompatible_roles"] == [
            ("transaction_entry", "transaction_approval")
        ]
    
    @pytest.mark.security
    def test_authorization_matrix(self, controls_manager):
        """Test authorization matrix enforcement."""
        # Define authorization request
        auth_request = {
            "user_id": "FIN_USER_123",
            "role": "financial_analyst",
            "resource": "general_ledger",
            "action": "modify",
            "amount": 50000
        }
        
        # Check authorization
        auth_result = controls_manager.check_authorization(
            request=auth_request,
            enforce_limits=True
        )
        
        assert auth_result["authorized"] is False
        assert auth_result["reason"] == "exceeds_role_limit"
        assert auth_result["max_allowed_amount"] == 10000
        assert auth_result["requires_approval_from"] == ["finance_manager", "cfo"]
    
    @pytest.mark.security
    def test_control_effectiveness_monitoring(self, controls_manager):
        """Test monitoring of control effectiveness."""
        # Simulate control execution
        control_executions = [
            {"control_id": "CTRL_001", "result": "passed", "timestamp": datetime.now()},
            {"control_id": "CTRL_001", "result": "passed", "timestamp": datetime.now()},
            {"control_id": "CTRL_001", "result": "failed", "timestamp": datetime.now()},
            {"control_id": "CTRL_002", "result": "passed", "timestamp": datetime.now()}
        ]
        
        # Calculate effectiveness
        effectiveness = controls_manager.calculate_control_effectiveness(
            executions=control_executions,
            control_id="CTRL_001"
        )
        
        assert effectiveness["total_executions"] == 3
        assert effectiveness["passed"] == 2
        assert effectiveness["failed"] == 1
        assert effectiveness["effectiveness_rate"] == 0.67
        assert effectiveness["meets_threshold"] is False  # Assuming 80% threshold


class TestSOXFinancialReporting:
    """Test SOX financial reporting integrity."""
    
    @pytest.fixture
    def reporting_integrity(self):
        """Provide financial reporting integrity manager."""
        return FinancialReportingIntegrity()
    
    @pytest.mark.security
    def test_financial_data_integrity(self, reporting_integrity):
        """Test financial data integrity controls."""
        # Financial data entry
        financial_data = {
            "transaction_id": "TXN_2024_001",
            "amount": 100000.00,
            "account": "revenue",
            "period": "Q1_2024",
            "source_system": "ERP"
        }
        
        # Record with integrity controls
        record_result = reporting_integrity.record_financial_data(
            data=financial_data,
            user_id="FIN_USER_456",
            require_validation=True
        )
        
        assert record_result["recorded"] is True
        assert record_result["integrity_hash"] is not None
        assert record_result["validation_status"] == "pending"
        assert record_result["requires_review"] is True
        assert record_result["tamper_evident_seal"] is not None
    
    @pytest.mark.security
    def test_material_weakness_detection(self, reporting_integrity):
        """Test detection of material weaknesses."""
        # Simulate control test results
        control_tests = [
            {"control": "revenue_recognition", "failures": 5, "total": 100},
            {"control": "expense_approval", "failures": 1, "total": 100},
            {"control": "journal_entry_review", "failures": 15, "total": 100}  # Material
        ]
        
        # Analyze for material weaknesses
        weaknesses = reporting_integrity.analyze_material_weaknesses(
            control_tests=control_tests,
            materiality_threshold=0.05
        )
        
        assert len(weaknesses["material_weaknesses"]) == 2
        assert "revenue_recognition" in weaknesses["material_weaknesses"]
        assert "journal_entry_review" in weaknesses["material_weaknesses"]
        assert weaknesses["requires_disclosure"] is True
        assert weaknesses["severity"] == "high"
    
    @pytest.mark.security
    def test_financial_statement_certification(self, reporting_integrity):
        """Test financial statement certification process."""
        # Prepare certification request
        certification_request = {
            "statement_type": "10-K",
            "period": "FY2023",
            "certifying_officers": ["CEO", "CFO"],
            "internal_controls_tested": True,
            "material_weaknesses": []
        }
        
        # Process certification
        cert_result = reporting_integrity.process_certification(
            request=certification_request,
            sub_certifications=["controller", "treasurer", "chief_accounting_officer"]
        )
        
        assert cert_result["certification_id"] is not None
        assert cert_result["status"] == "pending_signatures"
        assert len(cert_result["required_signatures"]) == 2
        assert cert_result["sub_certifications_complete"] is True
        assert cert_result["302_compliant"] is True
        assert cert_result["404_compliant"] is True


class TestSOXAuditTrail:
    """Test SOX audit trail requirements."""
    
    @pytest.fixture
    def audit_manager(self):
        """Provide audit trail manager."""
        return AuditTrailManager()
    
    @pytest.mark.security
    def test_comprehensive_audit_logging(self, audit_manager):
        """Test comprehensive audit trail logging."""
        # Financial system event
        event = {
            "event_type": "gl_entry_modification",
            "user_id": "FIN_USER_789",
            "timestamp": datetime.now(),
            "system": "general_ledger",
            "before_value": {"amount": 10000, "account": "1234"},
            "after_value": {"amount": 15000, "account": "1234"},
            "reason": "correction",
            "approval_id": "APPR_001"
        }
        
        # Log event
        log_result = audit_manager.log_financial_event(
            event=event,
            compliance_flags=["sox_relevant", "financial_impact"]
        )
        
        assert log_result["logged"] is True
        assert log_result["audit_id"] is not None
        assert log_result["immutable"] is True
        assert log_result["blockchain_anchored"] is True
        assert log_result["retention_years"] == 7
    
    @pytest.mark.security
    def test_audit_trail_integrity(self, audit_manager):
        """Test audit trail tamper resistance."""
        # Create audit entries
        entries = []
        for i in range(5):
            entry = audit_manager.create_audit_entry(
                action=f"action_{i}",
                user_id=f"user_{i}",
                details={"value": i * 100}
            )
            entries.append(entry)
        
        # Verify chain integrity
        integrity_check = audit_manager.verify_audit_chain(entries)
        
        assert integrity_check["valid"] is True
        assert integrity_check["chain_intact"] is True
        assert len(integrity_check["verified_links"]) == 4
        
        # Attempt tampering
        entries[2]["details"]["value"] = 999999  # Tamper with data
        
        # Re-verify - should detect tampering
        tamper_check = audit_manager.verify_audit_chain(entries)
        
        assert tamper_check["valid"] is False
        assert tamper_check["tampered_entry"] == 2
        assert tamper_check["alert_raised"] is True
    
    @pytest.mark.security
    def test_audit_log_retention(self, audit_manager):
        """Test audit log retention policies."""
        # Check retention policy
        retention = audit_manager.get_retention_policy("financial_audit_logs")
        
        assert retention["minimum_years"] == 7
        assert retention["delete_after_minimum"] is False
        assert retention["archive_location"] == "secure_cold_storage"
        assert retention["encryption_required"] is True
        
        # Test retention enforcement
        old_logs = [
            {"id": "log1", "date": datetime.now() - timedelta(days=365*8)},
            {"id": "log2", "date": datetime.now() - timedelta(days=365*6)}
        ]
        
        retention_result = audit_manager.enforce_retention(old_logs)
        
        assert retention_result["archived"] == ["log1"]
        assert retention_result["retained"] == ["log2"]
        assert retention_result["deletion_blocked"] is True


class TestSOXAccessControl:
    """Test SOX access control requirements."""
    
    @pytest.fixture
    def access_enforcer(self):
        """Provide access control enforcer."""
        return AccessControlEnforcer()
    
    @pytest.mark.security
    def test_privileged_access_management(self, access_enforcer):
        """Test privileged access management for financial systems."""
        # Request privileged access
        access_request = {
            "user_id": "ADMIN_001",
            "requested_role": "gl_admin",
            "systems": ["general_ledger", "financial_reporting"],
            "justification": "quarterly_closing",
            "duration_hours": 4
        }
        
        # Process request
        access_result = access_enforcer.grant_privileged_access(
            request=access_request,
            approver_id="CFO_001"
        )
        
        assert access_result["granted"] is True
        assert access_result["expires_at"] is not None
        assert access_result["monitoring_enabled"] is True
        assert access_result["session_recording"] is True
        assert access_result["real_time_alerts"] is True
        assert access_result["break_glass_available"] is False
    
    @pytest.mark.security
    def test_access_review_and_recertification(self, access_enforcer):
        """Test periodic access review and recertification."""
        # Get users requiring recertification
        users_for_review = access_enforcer.get_users_for_recertification(
            system="financial_systems",
            review_cycle_days=90
        )
        
        assert len(users_for_review) > 0
        
        # Perform access review
        review_result = access_enforcer.perform_access_review(
            user_id=users_for_review[0]["user_id"],
            reviewer_id="MANAGER_001",
            decisions={
                "general_ledger": "maintain",
                "payment_system": "revoke",
                "reporting_tool": "modify"
            }
        )
        
        assert review_result["review_complete"] is True
        assert review_result["access_modified"] is True
        assert "payment_system" in review_result["revoked_access"]
        assert review_result["next_review_date"] is not None
        assert review_result["audit_logged"] is True
    
    @pytest.mark.security
    def test_emergency_access_procedures(self, access_enforcer):
        """Test emergency (break-glass) access procedures."""
        # Emergency access request
        emergency_request = {
            "user_id": "EMERGENCY_USER_001",
            "system": "critical_financial_system",
            "reason": "system_outage_recovery",
            "incident_id": "INC_2024_001"
        }
        
        # Grant emergency access
        emergency_access = access_enforcer.grant_emergency_access(
            request=emergency_request,
            authorization_code="BREAKGLASS2024"
        )
        
        assert emergency_access["granted"] is True
        assert emergency_access["time_limited"] is True
        assert emergency_access["duration_hours"] <= 4
        assert emergency_access["enhanced_logging"] is True
        assert emergency_access["real_time_monitoring"] is True
        assert emergency_access["post_incident_review_required"] is True


class TestSOXChangeManagement:
    """Test SOX change management requirements."""
    
    @pytest.fixture
    def change_management(self):
        """Provide change management system."""
        return ChangeManagementSystem()
    
    @pytest.mark.security
    def test_financial_system_change_control(self, change_management):
        """Test change control for financial systems."""
        # Submit change request
        change_request = {
            "change_id": "CHG_2024_001",
            "system": "general_ledger",
            "change_type": "configuration",
            "description": "Update revenue recognition rules",
            "impact": "high",
            "testing_required": True,
            "rollback_plan": "Revert configuration to previous version"
        }
        
        # Process change request
        change_result = change_management.submit_change_request(
            request=change_request,
            requestor_id="DEV_001"
        )
        
        assert change_result["status"] == "pending_approval"
        assert len(change_result["required_approvals"]) >= 2
        assert "change_advisory_board" in change_result["required_approvals"]
        assert "sox_compliance_officer" in change_result["required_approvals"]
        assert change_result["testing_phase"] == "required"
    
    @pytest.mark.security
    def test_production_deployment_controls(self, change_management):
        """Test production deployment controls."""
        # Deployment request
        deployment = {
            "change_id": "CHG_2024_001",
            "environment": "production",
            "deployment_window": "2024-01-15 02:00:00",
            "systems_affected": ["gl", "reporting", "consolidation"]
        }
        
        # Validate deployment
        validation = change_management.validate_deployment(
            deployment=deployment,
            check_approvals=True,
            check_testing=True
        )
        
        assert validation["can_deploy"] is False
        assert "missing_test_results" in validation["blockers"]
        assert "blackout_period" not in validation["blockers"]
        assert validation["risk_score"] == "high"
        assert validation["requires_cab_presence"] is True
    
    @pytest.mark.security
    def test_configuration_baseline_management(self, change_management):
        """Test configuration baseline management."""
        # Create baseline
        baseline = change_management.create_configuration_baseline(
            system="financial_reporting",
            components=[
                "report_templates",
                "calculation_rules",
                "data_mappings",
                "security_settings"
            ]
        )
        
        assert baseline["baseline_id"] is not None
        assert baseline["hash"] is not None
        assert baseline["timestamp"] is not None
        assert baseline["approved_by"] is not None
        
        # Detect drift
        current_config = {
            "report_templates": "v2.1",
            "calculation_rules": "v1.5_modified",  # Drift
            "data_mappings": "v3.0",
            "security_settings": "strict"
        }
        
        drift_check = change_management.check_configuration_drift(
            baseline_id=baseline["baseline_id"],
            current_config=current_config
        )
        
        assert drift_check["drift_detected"] is True
        assert "calculation_rules" in drift_check["drifted_components"]
        assert drift_check["requires_investigation"] is True
        assert drift_check["compliance_impact"] == "high"


class TestSOXReportingSecurity:
    """Test SOX reporting security requirements."""
    
    @pytest.fixture
    def sox_enforcer(self):
        """Provide SOX enforcer instance."""
        return SOXEnforcer()
    
    @pytest.mark.security
    def test_financial_report_security(self, sox_enforcer):
        """Test security of financial reporting process."""
        # Generate financial report
        report_data = {
            "type": "quarterly_earnings",
            "period": "Q4_2023",
            "revenue": 50000000,
            "net_income": 5000000,
            "eps": 2.50
        }
        
        # Secure report generation
        secure_report = sox_enforcer.generate_secure_report(
            data=report_data,
            preparer_id="ANALYST_001",
            reviewer_id="CONTROLLER_001"
        )
        
        assert secure_report["encrypted"] is True
        assert secure_report["digitally_signed"] is True
        assert secure_report["watermarked"] is True
        assert secure_report["version_controlled"] is True
        assert secure_report["distribution_controlled"] is True
        assert len(secure_report["authorized_recipients"]) > 0
    
    @pytest.mark.security
    def test_whistleblower_protection(self, sox_enforcer):
        """Test whistleblower reporting protection."""
        # Submit whistleblower report
        report = {
            "concern_type": "financial_irregularity",
            "description": "Potential revenue recognition issue",
            "department": "sales",
            "anonymous": True
        }
        
        # Process report
        result = sox_enforcer.process_whistleblower_report(
            report=report,
            channel="secure_hotline"
        )
        
        assert result["received"] is True
        assert result["encrypted"] is True
        assert result["identity_protected"] is True
        assert result["investigation_initiated"] is True
        assert result["audit_committee_notified"] is True
        assert result["retaliation_monitoring"] is True


if __name__ == "__main__":
    pytest.main(["-v", "-m", "security", __file__])