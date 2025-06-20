"""
Unit tests for domain-specific data generation.

Tests the DomainDataGenerator class for creating realistic synthetic data
for various business domains including banking, healthcare, legal, etc.
"""

import pytest
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import Mock, patch

from structured_docs_synth.generation.content.domain_data_generator import (
    DomainDataGenerator,
    BankingDataGenerator,
    HealthcareDataGenerator,
    LegalDataGenerator,
    InsuranceDataGenerator,
    GovernmentDataGenerator,
    DomainType
)
from structured_docs_synth.core.exceptions import GenerationError


class TestDomainDataGenerator:
    """Test base domain data generator."""
    
    def test_create_banking_generator(self):
        """Test creating banking data generator."""
        generator = DomainDataGenerator.create(DomainType.BANKING)
        assert isinstance(generator, BankingDataGenerator)
    
    def test_create_healthcare_generator(self):
        """Test creating healthcare data generator."""
        generator = DomainDataGenerator.create(DomainType.HEALTHCARE)
        assert isinstance(generator, HealthcareDataGenerator)
    
    def test_create_legal_generator(self):
        """Test creating legal data generator."""
        generator = DomainDataGenerator.create(DomainType.LEGAL)
        assert isinstance(generator, LegalDataGenerator)
    
    def test_create_invalid_domain(self):
        """Test creating generator with invalid domain."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            DomainDataGenerator.create("invalid_domain")
    
    def test_base_generator_seed(self):
        """Test generator with seed for reproducibility."""
        gen1 = DomainDataGenerator.create(DomainType.BANKING, seed=42)
        gen2 = DomainDataGenerator.create(DomainType.BANKING, seed=42)
        
        # Should produce same results with same seed
        data1 = gen1.generate_account_number()
        data2 = gen2.generate_account_number()
        
        assert data1 == data2


class TestBankingDataGenerator:
    """Test banking data generation."""
    
    @pytest.fixture
    def banking_gen(self):
        """Provide banking data generator."""
        return BankingDataGenerator(seed=42)
    
    def test_generate_account_number(self, banking_gen):
        """Test account number generation."""
        account = banking_gen.generate_account_number()
        
        assert len(account) == 10
        assert account.isdigit()
        assert account[0] != '0'  # Should not start with 0
    
    def test_generate_routing_number(self, banking_gen):
        """Test routing number generation."""
        routing = banking_gen.generate_routing_number()
        
        assert len(routing) == 9
        assert routing.isdigit()
        # Verify checksum (last digit)
        assert banking_gen._verify_routing_checksum(routing)
    
    def test_generate_iban(self, banking_gen):
        """Test IBAN generation."""
        iban = banking_gen.generate_iban(country_code='GB')
        
        assert iban.startswith('GB')
        assert len(iban) == 22  # GB IBAN length
        assert iban[4:].isdigit()
    
    def test_generate_swift_code(self, banking_gen):
        """Test SWIFT/BIC code generation."""
        swift = banking_gen.generate_swift_code()
        
        assert len(swift) in [8, 11]
        assert swift[:4].isalpha()  # Bank code
        assert swift[4:6].isalpha()  # Country code
    
    def test_generate_transaction(self, banking_gen):
        """Test transaction data generation."""
        transaction = banking_gen.generate_transaction()
        
        assert 'transaction_id' in transaction
        assert 'date' in transaction
        assert 'amount' in transaction
        assert 'type' in transaction
        assert 'description' in transaction
        assert 'balance' in transaction
        
        # Validate transaction types
        assert transaction['type'] in ['debit', 'credit', 'transfer', 'fee']
        
        # Validate amount
        assert isinstance(transaction['amount'], Decimal)
        assert transaction['amount'] > 0
    
    def test_generate_bank_statement(self, banking_gen):
        """Test bank statement generation."""
        statement = banking_gen.generate_bank_statement(
            account_number='1234567890',
            num_transactions=10
        )
        
        assert statement['account_number'] == '1234567890'
        assert 'statement_period' in statement
        assert 'opening_balance' in statement
        assert 'closing_balance' in statement
        assert len(statement['transactions']) == 10
        
        # Verify running balance calculation
        balance = statement['opening_balance']
        for txn in statement['transactions']:
            if txn['type'] == 'credit':
                balance += txn['amount']
            else:
                balance -= txn['amount']
            assert abs(txn['balance'] - balance) < Decimal('0.01')
    
    def test_generate_loan_application(self, banking_gen):
        """Test loan application data generation."""
        loan = banking_gen.generate_loan_application()
        
        assert 'application_id' in loan
        assert 'applicant' in loan
        assert 'loan_type' in loan
        assert 'amount_requested' in loan
        assert 'term_months' in loan
        assert 'interest_rate' in loan
        assert 'credit_score' in loan
        assert 'income' in loan
        assert 'employment' in loan
        
        # Validate loan types
        assert loan['loan_type'] in ['personal', 'mortgage', 'auto', 'business']
        
        # Validate credit score
        assert 300 <= loan['credit_score'] <= 850
    
    def test_generate_credit_card(self, banking_gen):
        """Test credit card data generation."""
        card = banking_gen.generate_credit_card()
        
        assert 'card_number' in card
        assert 'card_type' in card
        assert 'expiry_date' in card
        assert 'cvv' in card
        assert 'cardholder_name' in card
        
        # Validate card number (Luhn algorithm)
        assert banking_gen._validate_luhn(card['card_number'])
        
        # Validate card type prefixes
        if card['card_type'] == 'visa':
            assert card['card_number'].startswith('4')
        elif card['card_type'] == 'mastercard':
            assert card['card_number'][0] in ['5', '2']


class TestHealthcareDataGenerator:
    """Test healthcare data generation."""
    
    @pytest.fixture
    def healthcare_gen(self):
        """Provide healthcare data generator."""
        return HealthcareDataGenerator(seed=42)
    
    def test_generate_patient_record(self, healthcare_gen):
        """Test patient record generation."""
        patient = healthcare_gen.generate_patient_record()
        
        assert 'patient_id' in patient
        assert 'mrn' in patient  # Medical Record Number
        assert 'demographics' in patient
        assert 'insurance' in patient
        assert 'emergency_contact' in patient
        
        # Validate demographics
        demo = patient['demographics']
        assert 'first_name' in demo
        assert 'last_name' in demo
        assert 'date_of_birth' in demo
        assert 'gender' in demo
        assert 'ssn' in demo
    
    def test_generate_medical_history(self, healthcare_gen):
        """Test medical history generation."""
        history = healthcare_gen.generate_medical_history()
        
        assert 'conditions' in history
        assert 'medications' in history
        assert 'allergies' in history
        assert 'procedures' in history
        assert 'immunizations' in history
        
        # Validate conditions
        for condition in history['conditions']:
            assert 'icd10_code' in condition
            assert 'description' in condition
            assert 'onset_date' in condition
            assert 'status' in condition
    
    def test_generate_prescription(self, healthcare_gen):
        """Test prescription generation."""
        rx = healthcare_gen.generate_prescription()
        
        assert 'prescription_id' in rx
        assert 'medication' in rx
        assert 'dosage' in rx
        assert 'frequency' in rx
        assert 'duration' in rx
        assert 'refills' in rx
        assert 'prescriber' in rx
        assert 'dea_number' in rx
        
        # Validate DEA number format
        dea = rx['dea_number']
        assert len(dea) == 9
        assert dea[0].isalpha()
        assert dea[1].isalpha()
        assert dea[2:].isdigit()
    
    def test_generate_lab_results(self, healthcare_gen):
        """Test lab results generation."""
        lab = healthcare_gen.generate_lab_results()
        
        assert 'lab_id' in lab
        assert 'patient_id' in lab
        assert 'order_date' in lab
        assert 'collection_date' in lab
        assert 'results' in lab
        
        # Validate results
        for result in lab['results']:
            assert 'test_name' in result
            assert 'value' in result
            assert 'unit' in result
            assert 'reference_range' in result
            assert 'flag' in result  # Normal, High, Low
    
    def test_generate_insurance_claim(self, healthcare_gen):
        """Test insurance claim generation."""
        claim = healthcare_gen.generate_insurance_claim()
        
        assert 'claim_id' in claim
        assert 'patient' in claim
        assert 'provider' in claim
        assert 'diagnosis_codes' in claim
        assert 'procedure_codes' in claim
        assert 'charges' in claim
        assert 'insurance_info' in claim
        
        # Validate procedure codes (CPT)
        for proc in claim['procedure_codes']:
            assert len(proc) == 5
            assert proc.isdigit()
    
    def test_generate_encounter_note(self, healthcare_gen):
        """Test clinical encounter note generation."""
        note = healthcare_gen.generate_encounter_note()
        
        assert 'encounter_id' in note
        assert 'date' in note
        assert 'chief_complaint' in note
        assert 'history_present_illness' in note
        assert 'physical_exam' in note
        assert 'assessment' in note
        assert 'plan' in note
        assert 'provider' in note


class TestLegalDataGenerator:
    """Test legal data generation."""
    
    @pytest.fixture
    def legal_gen(self):
        """Provide legal data generator."""
        return LegalDataGenerator(seed=42)
    
    def test_generate_contract(self, legal_gen):
        """Test contract generation."""
        contract = legal_gen.generate_contract()
        
        assert 'contract_id' in contract
        assert 'type' in contract
        assert 'parties' in contract
        assert 'effective_date' in contract
        assert 'terms' in contract
        assert 'governing_law' in contract
        assert 'signatures' in contract
        
        # Validate parties
        assert len(contract['parties']) >= 2
        for party in contract['parties']:
            assert 'name' in party
            assert 'role' in party
            assert 'address' in party
    
    def test_generate_case_filing(self, legal_gen):
        """Test case filing generation."""
        filing = legal_gen.generate_case_filing()
        
        assert 'case_number' in filing
        assert 'court' in filing
        assert 'filing_date' in filing
        assert 'case_type' in filing
        assert 'plaintiff' in filing
        assert 'defendant' in filing
        assert 'claims' in filing
        assert 'relief_sought' in filing
    
    def test_generate_patent_application(self, legal_gen):
        """Test patent application generation."""
        patent = legal_gen.generate_patent_application()
        
        assert 'application_number' in patent
        assert 'title' in patent
        assert 'inventors' in patent
        assert 'abstract' in patent
        assert 'claims' in patent
        assert 'description' in patent
        assert 'filing_date' in patent
        
        # Validate application number format
        app_num = patent['application_number']
        assert len(app_num) == 8
        assert app_num.isdigit()
    
    def test_generate_legal_memo(self, legal_gen):
        """Test legal memorandum generation."""
        memo = legal_gen.generate_legal_memo()
        
        assert 'to' in memo
        assert 'from' in memo
        assert 'date' in memo
        assert 'subject' in memo
        assert 'issue' in memo
        assert 'brief_answer' in memo
        assert 'facts' in memo
        assert 'discussion' in memo
        assert 'conclusion' in memo


class TestInsuranceDataGenerator:
    """Test insurance data generation."""
    
    @pytest.fixture
    def insurance_gen(self):
        """Provide insurance data generator."""
        return InsuranceDataGenerator(seed=42)
    
    def test_generate_policy(self, insurance_gen):
        """Test insurance policy generation."""
        policy = insurance_gen.generate_policy()
        
        assert 'policy_number' in policy
        assert 'type' in policy
        assert 'policyholder' in policy
        assert 'coverage' in policy
        assert 'premium' in policy
        assert 'deductible' in policy
        assert 'effective_date' in policy
        assert 'expiration_date' in policy
        
        # Validate policy types
        assert policy['type'] in ['auto', 'home', 'life', 'health', 'business']
    
    def test_generate_claim(self, insurance_gen):
        """Test insurance claim generation."""
        claim = insurance_gen.generate_claim()
        
        assert 'claim_number' in claim
        assert 'policy_number' in claim
        assert 'date_of_loss' in claim
        assert 'type' in claim
        assert 'description' in claim
        assert 'amount_claimed' in claim
        assert 'status' in claim
        assert 'adjuster' in claim
    
    def test_generate_quote(self, insurance_gen):
        """Test insurance quote generation."""
        quote = insurance_gen.generate_quote()
        
        assert 'quote_id' in quote
        assert 'type' in quote
        assert 'applicant' in quote
        assert 'coverage_options' in quote
        assert 'premium_estimate' in quote
        assert 'valid_until' in quote


class TestGovernmentDataGenerator:
    """Test government data generation."""
    
    @pytest.fixture
    def gov_gen(self):
        """Provide government data generator."""
        return GovernmentDataGenerator(seed=42)
    
    def test_generate_permit_application(self, gov_gen):
        """Test permit application generation."""
        permit = gov_gen.generate_permit_application()
        
        assert 'application_id' in permit
        assert 'permit_type' in permit
        assert 'applicant' in permit
        assert 'property_address' in permit
        assert 'description' in permit
        assert 'estimated_value' in permit
        assert 'status' in permit
    
    def test_generate_tax_form(self, gov_gen):
        """Test tax form generation."""
        tax_form = gov_gen.generate_tax_form()
        
        assert 'form_type' in tax_form
        assert 'tax_year' in tax_form
        assert 'taxpayer' in tax_form
        assert 'income' in tax_form
        assert 'deductions' in tax_form
        assert 'tax_owed' in tax_form
        
        # Validate SSN/EIN format
        if 'ssn' in tax_form['taxpayer']:
            ssn = tax_form['taxpayer']['ssn']
            assert len(ssn) == 11  # XXX-XX-XXXX
            assert ssn[3] == '-' and ssn[6] == '-'
    
    def test_generate_business_license(self, gov_gen):
        """Test business license generation."""
        license = gov_gen.generate_business_license()
        
        assert 'license_number' in license
        assert 'business_name' in license
        assert 'business_type' in license
        assert 'owner' in license
        assert 'address' in license
        assert 'issue_date' in license
        assert 'expiration_date' in license
        assert 'fee' in license