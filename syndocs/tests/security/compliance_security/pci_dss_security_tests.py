"""Security tests for PCI-DSS compliance.

Tests security controls required for Payment Card Industry Data Security Standard
including cardholder data protection, network security, access control, monitoring,
and security testing requirements.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import json
import re
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import ipaddress

from structured_docs_synth.privacy.compliance.pci_dss_enforcer import (
    PCIDSSEnforcer,
    CardholderDataProtection,
    NetworkSecurityManager,
    AccessControlManager,
    VulnerabilityManager,
    SecurityMonitoring
)
from structured_docs_synth.core.exceptions import ComplianceError, SecurityError


class TestPCIDSSCardholderData:
    """Test PCI-DSS cardholder data protection requirements."""
    
    @pytest.fixture
    def pci_enforcer(self):
        """Provide PCI-DSS enforcer instance."""
        return PCIDSSEnforcer()
    
    @pytest.fixture
    def data_protection(self):
        """Provide cardholder data protection manager."""
        return CardholderDataProtection()
    
    @pytest.mark.security
    def test_card_number_detection(self, data_protection):
        """Test detection of card numbers in various formats."""
        # Test data with various card number formats
        test_data = {
            "valid_card": "4111111111111111",  # Valid Visa
            "formatted_card": "4111-1111-1111-1111",
            "spaced_card": "4111 1111 1111 1111",
            "partial_card": "411111******1111",
            "not_card": "1234567890123456",  # Invalid Luhn
            "text": "Customer paid with card ending in 1111"
        }
        
        # Scan for card numbers
        scan_result = data_protection.scan_for_card_data(test_data)
        
        assert scan_result["found_card_numbers"] is True
        assert len(scan_result["detected_cards"]) >= 3
        assert scan_result["risk_level"] == "critical"
        assert "valid_card" in scan_result["fields_with_card_data"]
        assert scan_result["requires_immediate_action"] is True
    
    @pytest.mark.security
    def test_card_data_encryption(self, data_protection):
        """Test encryption of cardholder data."""
        # Card data to encrypt
        card_data = {
            "pan": "4111111111111111",
            "expiry": "12/25",
            "cvv": "123",
            "cardholder_name": "John Doe"
        }
        
        # Encrypt card data
        encrypted = data_protection.encrypt_card_data(
            data=card_data,
            encryption_method="AES256_GCM",
            key_management="HSM"
        )
        
        assert encrypted["encrypted"] is True
        assert encrypted["pan"] != card_data["pan"]
        assert "4111111111111111" not in str(encrypted)
        assert encrypted["encryption_metadata"]["algorithm"] == "AES256_GCM"
        assert encrypted["encryption_metadata"]["key_id"] is not None
        assert encrypted["encryption_metadata"]["hsm_protected"] is True
    
    @pytest.mark.security
    def test_tokenization(self, data_protection):
        """Test card data tokenization."""
        # Tokenize card number
        card_number = "4111111111111111"
        
        token_result = data_protection.tokenize_card(
            card_number=card_number,
            token_format="FORMAT_PRESERVING",
            vault_storage=True
        )
        
        assert token_result["token"] is not None
        assert token_result["token"] != card_number
        assert len(token_result["token"]) == len(card_number)
        assert token_result["token"][:6] == card_number[:6]  # Preserve BIN
        assert token_result["token"][-4:] == card_number[-4:]  # Preserve last 4
        assert token_result["vault_id"] is not None
        assert token_result["reversible"] is True
    
    @pytest.mark.security
    def test_secure_deletion(self, data_protection):
        """Test secure deletion of cardholder data."""
        # Data to be deleted
        sensitive_data = {
            "file_path": "/tmp/card_data.txt",
            "database_record": "payment_123",
            "cache_key": "card_cache_456"
        }
        
        # Perform secure deletion
        deletion_result = data_protection.secure_delete(
            data_references=sensitive_data,
            overwrite_passes=7,
            verify_deletion=True
        )
        
        assert deletion_result["deleted"] is True
        assert deletion_result["overwrite_completed"] is True
        assert deletion_result["verification_passed"] is True
        assert deletion_result["forensic_resistant"] is True
        assert deletion_result["audit_logged"] is True


class TestPCIDSSNetworkSecurity:
    """Test PCI-DSS network security requirements."""
    
    @pytest.fixture
    def network_security(self):
        """Provide network security manager."""
        return NetworkSecurityManager()
    
    @pytest.mark.security
    def test_network_segmentation(self, network_security):
        """Test network segmentation for cardholder data environment."""
        # Define network topology
        network_config = {
            "cde_subnet": "10.0.1.0/24",
            "dmz_subnet": "10.0.2.0/24",
            "corporate_subnet": "10.0.3.0/24",
            "public_subnet": "10.0.4.0/24"
        }
        
        # Validate segmentation
        segmentation = network_security.validate_network_segmentation(
            config=network_config,
            check_isolation=True
        )
        
        assert segmentation["properly_segmented"] is True
        assert segmentation["cde_isolated"] is True
        assert segmentation["firewall_rules_correct"] is True
        assert len(segmentation["allowed_connections"]) == 0
        assert segmentation["compliance_status"] == "compliant"
    
    @pytest.mark.security
    def test_firewall_configuration(self, network_security):
        """Test firewall configuration compliance."""
        # Firewall rules
        firewall_rules = [
            {"src": "0.0.0.0/0", "dst": "10.0.1.0/24", "action": "deny"},  # Good
            {"src": "10.0.3.0/24", "dst": "10.0.1.0/24", "action": "allow"},  # Bad
            {"src": "10.0.1.0/24", "dst": "0.0.0.0/0", "action": "allow", "port": 443}
        ]
        
        # Validate rules
        validation = network_security.validate_firewall_rules(
            rules=firewall_rules,
            cde_subnet="10.0.1.0/24"
        )
        
        assert validation["has_violations"] is True
        assert len(validation["violations"]) >= 1
        assert validation["violations"][0]["rule_index"] == 1
        assert validation["violations"][0]["reason"] == "corporate_to_cde_access"
        assert validation["recommendations"] is not None
    
    @pytest.mark.security
    def test_encryption_in_transit(self, network_security):
        """Test encryption requirements for data in transit."""
        # Connection configuration
        connection = {
            "protocol": "https",
            "tls_version": "1.2",
            "cipher_suite": "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            "certificate": {"valid": True, "expires": "2025-12-31"}
        }
        
        # Validate encryption
        encryption_check = network_security.validate_encryption_in_transit(
            connection=connection,
            data_type="cardholder_data"
        )
        
        assert encryption_check["compliant"] is False  # TLS 1.2 not sufficient
        assert encryption_check["minimum_tls_version"] == "1.3"
        assert encryption_check["strong_cipher"] is True
        assert encryption_check["certificate_valid"] is True
        
        # Test with TLS 1.3
        connection["tls_version"] = "1.3"
        encryption_check = network_security.validate_encryption_in_transit(
            connection=connection,
            data_type="cardholder_data"
        )
        
        assert encryption_check["compliant"] is True


class TestPCIDSSAccessControl:
    """Test PCI-DSS access control requirements."""
    
    @pytest.fixture
    def access_control(self):
        """Provide access control manager."""
        return AccessControlManager()
    
    @pytest.mark.security
    def test_unique_user_ids(self, access_control):
        """Test unique user ID requirements."""
        # User accounts
        users = [
            {"id": "user1", "name": "John Doe", "shared": False},
            {"id": "user2", "name": "Jane Smith", "shared": False},
            {"id": "admin", "name": "Admin Account", "shared": True},  # Violation
            {"id": "service", "name": "Service Account", "shared": True}  # Violation
        ]
        
        # Validate user IDs
        validation = access_control.validate_user_ids(users)
        
        assert validation["has_violations"] is True
        assert len(validation["shared_accounts"]) == 2
        assert "admin" in validation["shared_accounts"]
        assert validation["remediation_required"] is True
    
    @pytest.mark.security
    def test_password_requirements(self, access_control):
        """Test PCI-DSS password requirements."""
        # Test passwords
        passwords = {
            "weak": "password123",
            "no_special": "Password123",
            "short": "P@ss1",
            "strong": "C0mpl3x!P@ssw0rd#2024",
            "reused": "OldP@ssw0rd!123"
        }
        
        # Validate passwords
        for pwd_type, password in passwords.items():
            result = access_control.validate_password(
                password=password,
                user_id="test_user",
                check_history=True
            )
            
            if pwd_type == "strong":
                assert result["compliant"] is True
                assert result["strength_score"] >= 90
            else:
                assert result["compliant"] is False
                assert len(result["violations"]) > 0
    
    @pytest.mark.security
    def test_access_removal_procedures(self, access_control):
        """Test timely removal of access for terminated users."""
        # Termination event
        termination = {
            "user_id": "USER_789",
            "termination_time": datetime.now() - timedelta(hours=2),
            "systems": ["payment_gateway", "card_vault", "reporting"]
        }
        
        # Check access removal
        removal_status = access_control.check_access_removal(termination)
        
        assert removal_status["removal_complete"] is False  # > 24 hours
        assert removal_status["systems_with_access"] == ["payment_gateway"]
        assert removal_status["compliance_violation"] is True
        assert removal_status["time_exceeded_hours"] == 2
    
    @pytest.mark.security
    def test_physical_access_controls(self, access_control):
        """Test physical access controls for CDE."""
        # Physical access request
        access_request = {
            "user_id": "VISITOR_001",
            "area": "data_center_cde",
            "purpose": "maintenance",
            "escort_required": True,
            "duration_hours": 2
        }
        
        # Validate physical access
        access_result = access_control.grant_physical_access(
            request=access_request,
            approver_id="DC_MANAGER_001"
        )
        
        assert access_result["granted"] is True
        assert access_result["escort_assigned"] is True
        assert access_result["badge_type"] == "temporary_restricted"
        assert access_result["monitoring_enabled"] is True
        assert access_result["video_surveillance"] is True


class TestPCIDSSVulnerabilityManagement:
    """Test PCI-DSS vulnerability management requirements."""
    
    @pytest.fixture
    def vuln_manager(self):
        """Provide vulnerability manager."""
        return VulnerabilityManager()
    
    @pytest.mark.security
    def test_vulnerability_scanning(self, vuln_manager):
        """Test regular vulnerability scanning requirements."""
        # Scan configuration
        scan_config = {
            "scan_type": "authenticated",
            "targets": ["10.0.1.0/24"],
            "scan_profile": "pci_dss_quarterly"
        }
        
        # Execute scan
        scan_result = vuln_manager.execute_vulnerability_scan(
            config=scan_config,
            scanner="approved_asv"
        )
        
        assert scan_result["scan_completed"] is True
        assert scan_result["asv_approved"] is True
        assert scan_result["critical_vulnerabilities"] >= 0
        assert scan_result["pci_compliance_status"] is not None
        assert scan_result["next_scan_due"] is not None
    
    @pytest.mark.security
    def test_patch_management(self, vuln_manager):
        """Test security patch management."""
        # System inventory
        systems = [
            {"id": "web01", "os": "Linux", "last_patch": "2024-01-15"},
            {"id": "db01", "os": "Linux", "last_patch": "2023-12-01"},  # Overdue
            {"id": "app01", "os": "Windows", "last_patch": "2024-02-01"}
        ]
        
        # Check patch status
        patch_status = vuln_manager.check_patch_compliance(
            systems=systems,
            max_days_since_patch=30
        )
        
        assert patch_status["compliant"] is False
        assert len(patch_status["systems_needing_patches"]) == 1
        assert "db01" in patch_status["systems_needing_patches"]
        assert patch_status["critical_patches_missing"] > 0
    
    @pytest.mark.security
    def test_penetration_testing(self, vuln_manager):
        """Test penetration testing requirements."""
        # Last pentest info
        last_pentest = {
            "date": datetime.now() - timedelta(days=400),  # > 1 year
            "scope": ["external", "internal", "segmentation"],
            "findings": ["critical": 0, "high": 2, "medium": 5]
        }
        
        # Check pentest compliance
        pentest_status = vuln_manager.check_pentest_compliance(last_pentest)
        
        assert pentest_status["compliant"] is False
        assert pentest_status["overdue_days"] > 35  # 365 day requirement
        assert pentest_status["required_scope"] == ["external", "internal", "segmentation"]
        assert pentest_status["schedule_immediately"] is True


class TestPCIDSSMonitoring:
    """Test PCI-DSS monitoring and logging requirements."""
    
    @pytest.fixture
    def security_monitoring(self):
        """Provide security monitoring manager."""
        return SecurityMonitoring()
    
    @pytest.mark.security
    def test_log_monitoring(self, security_monitoring):
        """Test security log monitoring requirements."""
        # Log sources
        log_sources = [
            "firewall", "ids", "authentication", "access_control",
            "anti_virus", "application", "database"
        ]
        
        # Validate logging
        log_validation = security_monitoring.validate_log_sources(log_sources)
        
        assert log_validation["all_required_sources"] is True
        assert log_validation["centralized_logging"] is True
        assert log_validation["real_time_monitoring"] is True
        assert log_validation["retention_days"] >= 365
    
    @pytest.mark.security
    def test_file_integrity_monitoring(self, security_monitoring):
        """Test file integrity monitoring (FIM)."""
        # FIM configuration
        fim_config = {
            "monitored_paths": [
                "/etc/", "/usr/bin/", "/payment_app/",
                "/var/www/", "/database/config/"
            ],
            "check_frequency": "hourly",
            "alert_on_change": True
        }
        
        # Validate FIM
        fim_validation = security_monitoring.validate_fim_config(fim_config)
        
        assert fim_validation["covers_critical_files"] is True
        assert fim_validation["frequency_compliant"] is True
        assert fim_validation["real_time_alerts"] is True
        assert fim_validation["baseline_established"] is True
    
    @pytest.mark.security
    def test_intrusion_detection(self, security_monitoring):
        """Test intrusion detection system requirements."""
        # IDS alert
        ids_alert = {
            "timestamp": datetime.now(),
            "source_ip": "192.168.1.100",
            "destination_ip": "10.0.1.50",
            "alert_type": "sql_injection_attempt",
            "severity": "critical",
            "payload": "' OR '1'='1"
        }
        
        # Process alert
        alert_response = security_monitoring.process_ids_alert(
            alert=ids_alert,
            automated_response=True
        )
        
        assert alert_response["investigated"] is True
        assert alert_response["blocked"] is True
        assert alert_response["incident_created"] is True
        assert alert_response["notification_sent"] is True
        assert alert_response["forensics_initiated"] is True
    
    @pytest.mark.security
    def test_daily_log_review(self, security_monitoring):
        """Test daily log review process."""
        # Log review task
        review_date = datetime.now().date()
        
        # Perform review
        review_result = security_monitoring.perform_daily_log_review(
            date=review_date,
            reviewer_id="SEC_ANALYST_001"
        )
        
        assert review_result["review_completed"] is True
        assert review_result["anomalies_found"] >= 0
        assert review_result["incidents_created"] >= 0
        assert review_result["review_documented"] is True
        assert review_result["sign_off_recorded"] is True


class TestPCIDSSIncidentResponse:
    """Test PCI-DSS incident response requirements."""
    
    @pytest.fixture
    def pci_enforcer(self):
        """Provide PCI-DSS enforcer instance."""
        return PCIDSSEnforcer()
    
    @pytest.mark.security
    def test_incident_response_plan(self, pci_enforcer):
        """Test incident response plan requirements."""
        # Validate IR plan
        ir_plan = pci_enforcer.validate_incident_response_plan()
        
        assert ir_plan["plan_exists"] is True
        assert ir_plan["roles_defined"] is True
        assert ir_plan["contact_list_current"] is True
        assert ir_plan["tested_annually"] is True
        assert all(phase in ir_plan["phases"] for phase in [
            "preparation", "identification", "containment",
            "eradication", "recovery", "lessons_learned"
        ])
    
    @pytest.mark.security
    def test_data_breach_response(self, pci_enforcer):
        """Test data breach response procedures."""
        # Simulated breach
        breach_event = {
            "type": "unauthorized_access",
            "affected_records": 1000,
            "data_types": ["pan", "expiry", "cardholder_name"],
            "detection_time": datetime.now(),
            "source": "external_attacker"
        }
        
        # Initiate response
        response = pci_enforcer.initiate_breach_response(breach_event)
        
        assert response["containment_initiated"] is True
        assert response["forensics_engaged"] is True
        assert response["card_brands_notified"] is True
        assert response["law_enforcement_notified"] is True
        assert response["public_disclosure_required"] is True
        assert response["timeline_compliant"] is True


if __name__ == "__main__":
    pytest.main(["-v", "-m", "security", __file__])