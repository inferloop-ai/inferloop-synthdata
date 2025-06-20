"""
Unit tests for GDPR compliance enforcement.

Tests the GDPR enforcer for ensuring data privacy compliance including
consent management, data minimization, and right to erasure.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from structured_docs_synth.privacy.compliance.gdpr_enforcer import (
    GDPREnforcer,
    GDPRConfig,
    ConsentRecord,
    DataSubject,
    ProcessingActivity,
    DataCategory,
    LawfulBasis,
    RetentionPolicy
)
from structured_docs_synth.core.exceptions import PrivacyError, ComplianceError


class TestGDPRConfig:
    """Test GDPR configuration."""
    
    def test_default_config(self):
        """Test default GDPR configuration."""
        config = GDPRConfig()
        
        assert config.enforce_consent is True
        assert config.enforce_minimization is True
        assert config.enforce_retention is True
        assert config.default_retention_days == 365
        assert config.require_explicit_consent is True
        assert config.allow_profiling is False
        assert config.enable_audit_logging is True
    
    def test_custom_config(self):
        """Test custom GDPR configuration."""
        config = GDPRConfig(
            enforce_consent=False,
            default_retention_days=730,
            allow_profiling=True
        )
        
        assert config.enforce_consent is False
        assert config.default_retention_days == 730
        assert config.allow_profiling is True


class TestConsentRecord:
    """Test consent record management."""
    
    def test_consent_creation(self):
        """Test creating consent record."""
        consent = ConsentRecord(
            subject_id='user123',
            purpose='marketing',
            given_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),
            scope=['email', 'name'],
            withdrawal_method='email'
        )
        
        assert consent.subject_id == 'user123'
        assert consent.purpose == 'marketing'
        assert consent.is_valid()
        assert 'email' in consent.scope
    
    def test_consent_expiry(self):
        """Test consent expiration."""
        # Expired consent
        consent = ConsentRecord(
            subject_id='user123',
            purpose='marketing',
            given_at=datetime.now() - timedelta(days=400),
            expires_at=datetime.now() - timedelta(days=35)
        )
        
        assert not consent.is_valid()
        assert consent.is_expired()
    
    def test_consent_withdrawal(self):
        """Test consent withdrawal."""
        consent = ConsentRecord(
            subject_id='user123',
            purpose='marketing',
            given_at=datetime.now()
        )
        
        assert consent.is_valid()
        
        # Withdraw consent
        consent.withdraw()
        
        assert not consent.is_valid()
        assert consent.withdrawn is True
        assert consent.withdrawn_at is not None


class TestDataSubject:
    """Test data subject rights management."""
    
    def test_data_subject_creation(self):
        """Test creating data subject."""
        subject = DataSubject(
            subject_id='user123',
            email='user@example.com',
            created_at=datetime.now(),
            consents=[],
            data_categories=[DataCategory.PERSONAL, DataCategory.CONTACT]
        )
        
        assert subject.subject_id == 'user123'
        assert subject.email == 'user@example.com'
        assert DataCategory.PERSONAL in subject.data_categories
    
    def test_add_consent(self):
        """Test adding consent to subject."""
        subject = DataSubject('user123', 'user@example.com')
        
        consent = ConsentRecord(
            subject_id='user123',
            purpose='marketing',
            given_at=datetime.now()
        )
        
        subject.add_consent(consent)
        
        assert len(subject.consents) == 1
        assert subject.has_valid_consent('marketing')
    
    def test_right_to_access(self):
        """Test data subject's right to access."""
        subject = DataSubject('user123', 'user@example.com')
        subject.personal_data = {
            'name': 'John Doe',
            'phone': '+1234567890',
            'address': '123 Main St'
        }
        
        data_export = subject.export_personal_data()
        
        assert data_export['subject_id'] == 'user123'
        assert data_export['personal_data']['name'] == 'John Doe'
        assert 'export_date' in data_export
    
    def test_right_to_erasure(self):
        """Test data subject's right to be forgotten."""
        subject = DataSubject('user123', 'user@example.com')
        subject.personal_data = {'name': 'John Doe'}
        
        # Request erasure
        erasure_result = subject.request_erasure()
        
        assert erasure_result['erased'] is True
        assert subject.personal_data == {}
        assert subject.erasure_requested is True
        assert subject.erasure_date is not None


class TestGDPREnforcer:
    """Test GDPR compliance enforcement."""
    
    @pytest.fixture
    def enforcer(self):
        """Provide GDPR enforcer instance."""
        return GDPREnforcer()
    
    @pytest.fixture
    def mock_audit_logger(self):
        """Provide mock audit logger."""
        logger = AsyncMock()
        logger.log_activity = AsyncMock()
        return logger
    
    def test_process_with_consent(self, enforcer):
        """Test data processing with valid consent."""
        # Register data subject
        subject = enforcer.register_data_subject(
            subject_id='user123',
            email='user@example.com'
        )
        
        # Record consent
        enforcer.record_consent(
            subject_id='user123',
            purpose='analytics',
            scope=['usage_data', 'preferences']
        )
        
        # Process data
        result = enforcer.process_personal_data(
            subject_id='user123',
            purpose='analytics',
            data={'page_views': 10, 'preferences': {'theme': 'dark'}}
        )
        
        assert result['allowed'] is True
        assert result['lawful_basis'] == LawfulBasis.CONSENT
    
    def test_process_without_consent(self, enforcer):
        """Test data processing without consent."""
        enforcer.register_data_subject('user123', 'user@example.com')
        
        with pytest.raises(ComplianceError, match="No valid consent"):
            enforcer.process_personal_data(
                subject_id='user123',
                purpose='marketing',
                data={'email': 'user@example.com'}
            )
    
    def test_legitimate_interest(self, enforcer):
        """Test processing under legitimate interest."""
        enforcer.register_data_subject('user123', 'user@example.com')
        
        # Configure legitimate interest
        enforcer.configure_legitimate_interest(
            purpose='fraud_prevention',
            description='Preventing fraudulent activities',
            balancing_test_passed=True
        )
        
        result = enforcer.process_personal_data(
            subject_id='user123',
            purpose='fraud_prevention',
            data={'ip_address': '192.168.1.1'},
            lawful_basis=LawfulBasis.LEGITIMATE_INTEREST
        )
        
        assert result['allowed'] is True
        assert result['lawful_basis'] == LawfulBasis.LEGITIMATE_INTEREST
    
    def test_data_minimization(self, enforcer):
        """Test data minimization principle."""
        enforcer.config.enforce_minimization = True
        
        # Define minimal data requirements
        enforcer.define_minimal_data(
            purpose='registration',
            required_fields=['email', 'password'],
            optional_fields=['name']
        )
        
        # Test with excessive data
        with pytest.raises(ComplianceError, match="Data minimization"):
            enforcer.validate_data_minimization(
                purpose='registration',
                data={
                    'email': 'user@example.com',
                    'password': 'hash',
                    'name': 'John',
                    'phone': '1234567890',  # Not required
                    'address': '123 Main St'  # Not required
                }
            )
    
    def test_retention_policy(self, enforcer):
        """Test data retention enforcement."""
        # Set retention policy
        policy = RetentionPolicy(
            data_category=DataCategory.PERSONAL,
            retention_days=30,
            deletion_method='anonymize'
        )
        enforcer.set_retention_policy(policy)
        
        # Register old data
        old_date = datetime.now() - timedelta(days=45)
        enforcer.register_data_storage(
            subject_id='user123',
            data_category=DataCategory.PERSONAL,
            stored_at=old_date
        )
        
        # Check expired data
        expired = enforcer.get_expired_data()
        
        assert len(expired) == 1
        assert expired[0]['subject_id'] == 'user123'
        assert expired[0]['days_over'] > 14
    
    def test_consent_withdrawal(self, enforcer):
        """Test consent withdrawal handling."""
        # Setup subject with consent
        enforcer.register_data_subject('user123', 'user@example.com')
        enforcer.record_consent('user123', 'marketing', ['email'])
        
        # Verify consent exists
        assert enforcer.has_valid_consent('user123', 'marketing')
        
        # Withdraw consent
        result = enforcer.withdraw_consent('user123', 'marketing')
        
        assert result['success'] is True
        assert not enforcer.has_valid_consent('user123', 'marketing')
    
    def test_data_portability(self, enforcer):
        """Test right to data portability."""
        # Register subject with data
        subject = enforcer.register_data_subject('user123', 'user@example.com')
        subject.personal_data = {
            'profile': {'name': 'John Doe', 'age': 30},
            'preferences': {'newsletter': True}
        }
        
        # Request portable data
        export = enforcer.export_subject_data('user123', format='json')
        
        assert export['format'] == 'json'
        assert 'personal_data' in export['data']
        assert export['data']['personal_data']['profile']['name'] == 'John Doe'
        assert export['machine_readable'] is True
    
    def test_privacy_by_design(self, enforcer):
        """Test privacy by design implementation."""
        # Configure privacy settings
        privacy_settings = enforcer.get_privacy_by_design_settings()
        
        assert privacy_settings['data_minimization'] is True
        assert privacy_settings['encryption_at_rest'] is True
        assert privacy_settings['pseudonymization'] is True
        assert privacy_settings['access_controls'] is True
    
    def test_data_breach_notification(self, enforcer):
        """Test data breach handling."""
        # Report breach
        breach_report = enforcer.report_data_breach(
            breach_type='unauthorized_access',
            affected_subjects=['user123', 'user456'],
            data_categories=[DataCategory.PERSONAL, DataCategory.SENSITIVE],
            breach_date=datetime.now(),
            description='Unauthorized database access'
        )
        
        assert breach_report['id'] is not None
        assert breach_report['severity'] == 'high'  # Due to sensitive data
        assert breach_report['notification_required'] is True
        assert breach_report['notification_deadline'] is not None
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, enforcer, mock_audit_logger):
        """Test GDPR audit logging."""
        enforcer.audit_logger = mock_audit_logger
        
        # Perform auditable action
        enforcer.register_data_subject('user123', 'user@example.com')
        enforcer.record_consent('user123', 'analytics', ['usage_data'])
        
        # Verify audit logs
        mock_audit_logger.log_activity.assert_called()
        call_args = mock_audit_logger.log_activity.call_args
        assert 'consent_recorded' in str(call_args)
    
    def test_cross_border_transfer(self, enforcer):
        """Test cross-border data transfer compliance."""
        # Configure transfer
        transfer = enforcer.validate_cross_border_transfer(
            source_country='DE',  # Germany (EU)
            destination_country='US',  # United States
            transfer_mechanism='standard_contractual_clauses',
            data_categories=[DataCategory.PERSONAL]
        )
        
        assert transfer['allowed'] is True
        assert transfer['mechanism'] == 'standard_contractual_clauses'
        assert 'safeguards' in transfer
        
        # Test transfer without safeguards
        with pytest.raises(ComplianceError, match="adequate safeguards"):
            enforcer.validate_cross_border_transfer(
                source_country='DE',
                destination_country='CN',  # China
                transfer_mechanism=None
            )
    
    def test_children_data_protection(self, enforcer):
        """Test special protection for children's data."""
        # Register child subject
        child_subject = enforcer.register_data_subject(
            subject_id='child123',
            email='parent@example.com',
            age=14
        )
        
        # Verify parental consent requirement
        with pytest.raises(ComplianceError, match="parental consent"):
            enforcer.record_consent(
                subject_id='child123',
                purpose='social_media',
                scope=['profile_data'],
                parental_consent=False
            )
        
        # Record with parental consent
        result = enforcer.record_consent(
            subject_id='child123',
            purpose='educational',
            scope=['learning_progress'],
            parental_consent=True,
            parent_id='parent123'
        )
        
        assert result['success'] is True
        assert result['requires_parental_consent'] is True
    
    def test_automated_decision_making(self, enforcer):
        """Test restrictions on automated decision-making."""
        enforcer.register_data_subject('user123', 'user@example.com')
        
        # Test automated profiling without consent
        with pytest.raises(ComplianceError, match="automated decision-making"):
            enforcer.validate_automated_processing(
                subject_id='user123',
                processing_type='profiling',
                has_human_intervention=False,
                has_explicit_consent=False
            )
        
        # Test with safeguards
        result = enforcer.validate_automated_processing(
            subject_id='user123',
            processing_type='credit_scoring',
            has_human_intervention=True,
            has_explicit_consent=True,
            explanation_provided=True
        )
        
        assert result['allowed'] is True
        assert result['safeguards_met'] is True
    
    def test_data_protection_impact_assessment(self, enforcer):
        """Test DPIA (Data Protection Impact Assessment) requirement."""
        # Check if DPIA is required
        dpia_required = enforcer.requires_dpia(
            processing_type='large_scale_profiling',
            data_categories=[DataCategory.SENSITIVE, DataCategory.BIOMETRIC],
            uses_new_technology=True
        )
        
        assert dpia_required is True
        
        # Conduct DPIA
        dpia_result = enforcer.conduct_dpia(
            project_name='Biometric Authentication System',
            risks_identified=['unauthorized_access', 'data_breach'],
            mitigation_measures=['encryption', 'access_control', 'monitoring']
        )
        
        assert dpia_result['risk_level'] == 'high'
        assert dpia_result['approval_required'] is True
        assert len(dpia_result['recommendations']) > 0