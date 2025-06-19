#!/usr/bin/env python3
"""
Domain Data Generator for domain-specific synthetic data
"""

import json
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date, timedelta
import random

from pydantic import BaseModel, Field
from faker import Faker

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class DataDomain(Enum):
    """Data generation domains"""
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    GOVERNMENT = "government"
    EDUCATION = "education"
    RETAIL = "retail"
    TECHNOLOGY = "technology"
    REAL_ESTATE = "real_estate"
    INSURANCE = "insurance"
    MANUFACTURING = "manufacturing"


class GenerationMode(Enum):
    """Data generation modes"""
    REALISTIC = "realistic"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"
    ANONYMIZED = "anonymized"


@dataclass
class GeneratedData:
    """Generated domain data"""
    domain: DataDomain
    data_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_timestamp: datetime = field(default_factory=datetime.now)


class DomainDataConfig(BaseModel):
    """Domain data generator configuration"""
    
    # Generation settings
    domain: DataDomain = Field(DataDomain.FINANCIAL, description="Data domain")
    generation_mode: GenerationMode = Field(GenerationMode.REALISTIC, description="Generation mode")
    num_records: int = Field(100, description="Number of records to generate")
    
    # Realism settings
    use_realistic_patterns: bool = Field(True, description="Use realistic data patterns")
    include_edge_cases: bool = Field(False, description="Include edge cases")
    consistency_level: float = Field(0.8, description="Data consistency level (0-1)")
    
    # Privacy settings
    anonymize_data: bool = Field(True, description="Anonymize generated data")
    privacy_level: str = Field("medium", description="Privacy level: low, medium, high")
    
    # Localization
    locale: str = Field("en_US", description="Data locale")
    currency: str = Field("USD", description="Currency for financial data")
    date_format: str = Field("%Y-%m-%d", description="Date format")
    
    # Domain-specific settings
    domain_config: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific configuration")


class DomainDataGenerator:
    """
    Domain Data Generator for creating domain-specific synthetic data
    
    Generates realistic data for various domains including financial,
    healthcare, legal, and government sectors.
    """
    
    def __init__(self, config: Optional[DomainDataConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or DomainDataConfig()
        
        # Initialize Faker with locale
        self.faker = Faker(self.config.locale)
        Faker.seed(42)  # For reproducible results
        
        # Domain-specific generators
        self.domain_generators = {
            DataDomain.FINANCIAL: self._generate_financial_data,
            DataDomain.HEALTHCARE: self._generate_healthcare_data,
            DataDomain.LEGAL: self._generate_legal_data,
            DataDomain.GOVERNMENT: self._generate_government_data,
            DataDomain.EDUCATION: self._generate_education_data,
            DataDomain.RETAIL: self._generate_retail_data,
            DataDomain.TECHNOLOGY: self._generate_technology_data,
            DataDomain.REAL_ESTATE: self._generate_real_estate_data,
            DataDomain.INSURANCE: self._generate_insurance_data,
            DataDomain.MANUFACTURING: self._generate_manufacturing_data
        }
        
        self.logger.info(f"Domain Data Generator initialized for {self.config.domain.value}")
    
    def generate_data(self, data_type: str, count: Optional[int] = None) -> List[GeneratedData]:
        """Generate domain-specific data"""
        start_time = time.time()
        count = count or self.config.num_records
        
        try:
            generator = self.domain_generators.get(self.config.domain)
            if not generator:
                raise ValueError(f"No generator available for domain: {self.config.domain}")
            
            generated_data = []
            for i in range(count):
                data = generator(data_type, i)
                generated_data.append(data)
            
            generation_time = time.time() - start_time
            self.logger.info(f"Generated {count} {data_type} records in {generation_time:.2f}s")
            
            return generated_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate {data_type} data: {e}")
            raise ProcessingError(f"Data generation error: {e}")
    
    def _generate_financial_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate financial domain data"""
        if data_type == "transaction":
            content = {
                "transaction_id": f"TXN_{index:08d}",
                "account_number": self._generate_account_number(),
                "amount": round(random.uniform(1.00, 10000.00), 2),
                "transaction_type": random.choice(["debit", "credit", "transfer"]),
                "merchant": self.faker.company(),
                "category": random.choice(["groceries", "gas", "dining", "shopping", "utilities"]),
                "date": self.faker.date_between(start_date="-1y", end_date="today").strftime(self.config.date_format),
                "description": self.faker.sentence(nb_words=6),
                "currency": self.config.currency,
                "balance": round(random.uniform(100.00, 50000.00), 2)
            }
        elif data_type == "loan_application":
            content = {
                "application_id": f"LOAN_{index:08d}",
                "applicant_name": self.faker.name(),
                "loan_amount": random.randint(5000, 500000),
                "loan_purpose": random.choice(["home", "auto", "personal", "business", "education"]),
                "annual_income": random.randint(30000, 200000),
                "credit_score": random.randint(300, 850),
                "employment_status": random.choice(["employed", "self_employed", "retired", "unemployed"]),
                "loan_term_months": random.choice([12, 24, 36, 48, 60, 84, 120, 180, 360]),
                "interest_rate": round(random.uniform(2.5, 18.0), 2),
                "collateral": self.faker.sentence() if random.random() < 0.6 else None,
                "application_date": self.faker.date_between(start_date="-6m", end_date="today").strftime(self.config.date_format)
            }
        elif data_type == "investment_portfolio":
            content = {
                "portfolio_id": f"PORT_{index:08d}",
                "account_holder": self.faker.name(),
                "total_value": round(random.uniform(1000, 1000000), 2),
                "holdings": self._generate_investment_holdings(),
                "risk_profile": random.choice(["conservative", "moderate", "aggressive"]),
                "last_updated": self.faker.date_time_between(start_date="-1m", end_date="now").isoformat(),
                "advisor": self.faker.name(),
                "fees_annual": round(random.uniform(0.1, 2.0), 2)
            }
        else:
            content = {"type": data_type, "data": f"Generated financial data {index}"}
        
        return GeneratedData(
            domain=DataDomain.FINANCIAL,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_healthcare_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate healthcare domain data"""
        if data_type == "patient_record":
            content = {
                "patient_id": f"PT_{index:08d}",
                "name": self.faker.name(),
                "date_of_birth": self.faker.date_of_birth(minimum_age=0, maximum_age=100).strftime(self.config.date_format),
                "gender": random.choice(["M", "F", "O"]),
                "medical_record_number": f"MRN{random.randint(100000, 999999)}",
                "primary_physician": f"Dr. {self.faker.name()}",
                "insurance_provider": random.choice(["Blue Cross", "Aetna", "Cigna", "UnitedHealth", "Kaiser"]),
                "emergency_contact": self.faker.name(),
                "allergies": random.sample(["Penicillin", "Shellfish", "Nuts", "Latex", "None"], k=random.randint(0, 3)),
                "medical_conditions": random.sample(["Hypertension", "Diabetes", "Asthma", "Arthritis"], k=random.randint(0, 2))
            }
        elif data_type == "prescription":
            content = {
                "prescription_id": f"RX_{index:08d}",
                "patient_id": f"PT_{random.randint(1, 1000):08d}",
                "medication": random.choice(["Lisinopril", "Metformin", "Atorvastatin", "Amlodipine", "Metoprolol"]),
                "dosage": f"{random.choice([5, 10, 20, 25, 50, 100])}mg",
                "frequency": random.choice(["Once daily", "Twice daily", "Three times daily", "As needed"]),
                "quantity": random.choice([30, 60, 90]),
                "refills": random.randint(0, 5),
                "prescribing_physician": f"Dr. {self.faker.name()}",
                "date_prescribed": self.faker.date_between(start_date="-1y", end_date="today").strftime(self.config.date_format),
                "pharmacy": self.faker.company() + " Pharmacy"
            }
        else:
            content = {"type": data_type, "data": f"Generated healthcare data {index}"}
        
        return GeneratedData(
            domain=DataDomain.HEALTHCARE,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_legal_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate legal domain data"""
        if data_type == "contract":
            content = {
                "contract_id": f"CONT_{index:08d}",
                "contract_type": random.choice(["Service Agreement", "NDA", "Employment", "Lease", "Purchase"]),
                "parties": [self.faker.company(), self.faker.company()],
                "effective_date": self.faker.date_between(start_date="-2y", end_date="today").strftime(self.config.date_format),
                "expiration_date": self.faker.date_between(start_date="today", end_date="+5y").strftime(self.config.date_format),
                "contract_value": random.randint(1000, 1000000),
                "jurisdiction": self.faker.state(),
                "governing_law": self.faker.state(),
                "attorney": f"{self.faker.name()}, Esq.",
                "status": random.choice(["Active", "Pending", "Expired", "Terminated"])
            }
        elif data_type == "case_filing":
            content = {
                "case_number": f"CASE_{index:08d}",
                "case_type": random.choice(["Civil", "Criminal", "Family", "Probate", "Bankruptcy"]),
                "plaintiff": self.faker.name(),
                "defendant": self.faker.name(),
                "filing_date": self.faker.date_between(start_date="-2y", end_date="today").strftime(self.config.date_format),
                "court": f"{self.faker.city()} Superior Court",
                "judge": f"Judge {self.faker.name()}",
                "attorney_plaintiff": f"{self.faker.name()}, Esq.",
                "attorney_defendant": f"{self.faker.name()}, Esq.",
                "case_status": random.choice(["Pending", "In Progress", "Closed", "Settled"])
            }
        else:
            content = {"type": data_type, "data": f"Generated legal data {index}"}
        
        return GeneratedData(
            domain=DataDomain.LEGAL,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_government_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate government domain data"""
        if data_type == "tax_return":
            content = {
                "return_id": f"TAX_{index:08d}",
                "taxpayer_name": self.faker.name(),
                "ssn": "XXX-XX-" + str(random.randint(1000, 9999)),
                "tax_year": random.randint(2020, 2024),
                "filing_status": random.choice(["Single", "Married Filing Jointly", "Married Filing Separately", "Head of Household"]),
                "total_income": random.randint(20000, 200000),
                "federal_tax_withheld": random.randint(2000, 40000),
                "deductions": random.randint(5000, 25000),
                "tax_owed": random.randint(0, 15000),
                "refund_amount": random.randint(0, 8000),
                "filing_date": self.faker.date_between(start_date="-1y", end_date="today").strftime(self.config.date_format)
            }
        elif data_type == "permit_application":
            content = {
                "permit_id": f"PERMIT_{index:08d}",
                "applicant_name": self.faker.name(),
                "permit_type": random.choice(["Building", "Business", "Environmental", "Special Event", "Zoning"]),
                "application_date": self.faker.date_between(start_date="-6m", end_date="today").strftime(self.config.date_format),
                "project_description": self.faker.text(max_nb_chars=200),
                "estimated_cost": random.randint(1000, 500000),
                "approval_status": random.choice(["Pending", "Approved", "Denied", "Under Review"]),
                "inspector": self.faker.name(),
                "fees_paid": random.randint(50, 5000)
            }
        else:
            content = {"type": data_type, "data": f"Generated government data {index}"}
        
        return GeneratedData(
            domain=DataDomain.GOVERNMENT,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_education_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate education domain data"""
        if data_type == "student_record":
            content = {
                "student_id": f"STU_{index:08d}",
                "name": self.faker.name(),
                "date_of_birth": self.faker.date_of_birth(minimum_age=18, maximum_age=30).strftime(self.config.date_format),
                "enrollment_date": self.faker.date_between(start_date="-4y", end_date="today").strftime(self.config.date_format),
                "major": random.choice(["Computer Science", "Business", "Engineering", "Biology", "Psychology", "English"]),
                "gpa": round(random.uniform(2.0, 4.0), 2),
                "credits_earned": random.randint(0, 120),
                "academic_standing": random.choice(["Good Standing", "Probation", "Dean's List", "Honor Roll"]),
                "advisor": self.faker.name(),
                "expected_graduation": self.faker.date_between(start_date="today", end_date="+4y").strftime(self.config.date_format)
            }
        elif data_type == "transcript":
            content = {
                "transcript_id": f"TRANS_{index:08d}",
                "student_id": f"STU_{random.randint(1, 1000):08d}",
                "courses": self._generate_course_list(),
                "total_credits": random.randint(60, 120),
                "cumulative_gpa": round(random.uniform(2.0, 4.0), 2),
                "degree_awarded": random.choice(["Bachelor of Science", "Bachelor of Arts", "None"]),
                "honors": random.choice(["Summa Cum Laude", "Magna Cum Laude", "Cum Laude", "None"]),
                "issue_date": self.faker.date_between(start_date="-1y", end_date="today").strftime(self.config.date_format)
            }
        elif data_type == "course_enrollment":
            content = {
                "enrollment_id": f"ENR_{index:08d}",
                "student_id": f"STU_{random.randint(1, 1000):08d}",
                "course_code": f"{random.choice(['CS', 'MATH', 'ENG', 'BIO', 'HIST'])}{random.randint(100, 400)}",
                "course_name": self.faker.catch_phrase(),
                "instructor": f"Prof. {self.faker.name()}",
                "semester": random.choice(["Fall 2023", "Spring 2024", "Summer 2024"]),
                "credits": random.choice([1, 2, 3, 4]),
                "section": f"{random.randint(1, 5):02d}",
                "enrollment_status": random.choice(["Enrolled", "Waitlisted", "Dropped"])
            }
        else:
            content = {"type": data_type, "data": f"Generated education data {index}"}
        
        return GeneratedData(
            domain=DataDomain.EDUCATION,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_retail_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate retail domain data"""
        if data_type == "order":
            content = {
                "order_id": f"ORD_{index:08d}",
                "customer_id": f"CUST_{random.randint(1, 10000):08d}",
                "customer_name": self.faker.name(),
                "order_date": self.faker.date_time_between(start_date="-1y", end_date="now").isoformat(),
                "items": self._generate_order_items(),
                "subtotal": round(random.uniform(10, 1000), 2),
                "tax": round(random.uniform(1, 100), 2),
                "shipping": round(random.uniform(0, 50), 2),
                "total": round(random.uniform(11, 1150), 2),
                "payment_method": random.choice(["Credit Card", "Debit Card", "PayPal", "Apple Pay", "Google Pay"]),
                "shipping_address": self.faker.address(),
                "order_status": random.choice(["Pending", "Processing", "Shipped", "Delivered", "Cancelled"])
            }
        elif data_type == "inventory":
            content = {
                "sku": f"SKU_{index:06d}",
                "product_name": self.faker.catch_phrase(),
                "category": random.choice(["Electronics", "Clothing", "Home", "Books", "Sports", "Toys"]),
                "quantity_in_stock": random.randint(0, 1000),
                "unit_cost": round(random.uniform(1, 500), 2),
                "retail_price": round(random.uniform(2, 1000), 2),
                "supplier": self.faker.company(),
                "reorder_level": random.randint(10, 100),
                "last_restocked": self.faker.date_between(start_date="-6m", end_date="today").strftime(self.config.date_format),
                "warehouse_location": f"{random.choice(['A', 'B', 'C'])}-{random.randint(1, 20)}-{random.randint(1, 10)}"
            }
        elif data_type == "customer":
            content = {
                "customer_id": f"CUST_{index:08d}",
                "name": self.faker.name(),
                "email": self.faker.email(),
                "phone": self.faker.phone_number(),
                "date_joined": self.faker.date_between(start_date="-5y", end_date="today").strftime(self.config.date_format),
                "loyalty_tier": random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
                "total_purchases": random.randint(0, 100),
                "lifetime_value": round(random.uniform(0, 10000), 2),
                "preferred_categories": random.sample(["Electronics", "Clothing", "Home", "Books"], k=random.randint(1, 3))
            }
        else:
            content = {"type": data_type, "data": f"Generated retail data {index}"}
        
        return GeneratedData(
            domain=DataDomain.RETAIL,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_technology_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate technology domain data"""
        if data_type == "server_log":
            content = {
                "log_id": f"LOG_{index:012d}",
                "timestamp": self.faker.date_time_between(start_date="-30d", end_date="now").isoformat(),
                "server_id": f"SRV_{random.randint(1, 50):03d}",
                "ip_address": self.faker.ipv4(),
                "request_method": random.choice(["GET", "POST", "PUT", "DELETE", "PATCH"]),
                "endpoint": f"/api/v1/{random.choice(['users', 'products', 'orders', 'auth'])}/{random.randint(1, 1000)}",
                "status_code": random.choice([200, 201, 204, 400, 401, 403, 404, 500]),
                "response_time_ms": random.randint(10, 2000),
                "user_agent": self.faker.user_agent(),
                "error_message": self.faker.sentence() if random.random() < 0.1 else None
            }
        elif data_type == "bug_report":
            content = {
                "ticket_id": f"BUG_{index:06d}",
                "title": self.faker.catch_phrase(),
                "description": self.faker.text(max_nb_chars=500),
                "severity": random.choice(["Critical", "High", "Medium", "Low"]),
                "priority": random.choice(["P0", "P1", "P2", "P3"]),
                "status": random.choice(["Open", "In Progress", "Code Review", "Testing", "Closed"]),
                "reporter": self.faker.email(),
                "assignee": self.faker.email() if random.random() < 0.8 else None,
                "created_date": self.faker.date_time_between(start_date="-6m", end_date="now").isoformat(),
                "affected_version": f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 10)}",
                "component": random.choice(["Frontend", "Backend", "Database", "API", "Mobile", "Infrastructure"])
            }
        elif data_type == "deployment":
            content = {
                "deployment_id": f"DEPLOY_{index:08d}",
                "application": random.choice(["web-app", "mobile-api", "auth-service", "payment-gateway"]),
                "version": f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 10)}",
                "environment": random.choice(["development", "staging", "production"]),
                "deployment_time": self.faker.date_time_between(start_date="-1y", end_date="now").isoformat(),
                "deployed_by": self.faker.email(),
                "commit_hash": self.faker.sha1()[:8],
                "status": random.choice(["Success", "Failed", "Rolled Back", "In Progress"]),
                "duration_seconds": random.randint(30, 600),
                "rollback_available": random.choice([True, False])
            }
        else:
            content = {"type": data_type, "data": f"Generated technology data {index}"}
        
        return GeneratedData(
            domain=DataDomain.TECHNOLOGY,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_real_estate_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate real estate domain data"""
        if data_type == "property_listing":
            content = {
                "listing_id": f"MLS_{index:08d}",
                "address": self.faker.address(),
                "property_type": random.choice(["Single Family", "Condo", "Townhouse", "Multi-Family", "Land"]),
                "bedrooms": random.randint(0, 6),
                "bathrooms": round(random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4]), 1),
                "square_feet": random.randint(500, 5000),
                "lot_size_acres": round(random.uniform(0.1, 5.0), 2),
                "year_built": random.randint(1900, 2024),
                "list_price": random.randint(100000, 2000000),
                "list_date": self.faker.date_between(start_date="-6m", end_date="today").strftime(self.config.date_format),
                "status": random.choice(["Active", "Pending", "Sold", "Withdrawn", "Expired"]),
                "days_on_market": random.randint(0, 365),
                "listing_agent": self.faker.name(),
                "broker": self.faker.company() + " Realty"
            }
        elif data_type == "lease_agreement":
            content = {
                "lease_id": f"LEASE_{index:08d}",
                "property_address": self.faker.address(),
                "landlord": self.faker.name(),
                "tenant": self.faker.name(),
                "lease_start": self.faker.date_between(start_date="-2y", end_date="today").strftime(self.config.date_format),
                "lease_end": self.faker.date_between(start_date="today", end_date="+2y").strftime(self.config.date_format),
                "monthly_rent": random.randint(500, 5000),
                "security_deposit": random.randint(500, 10000),
                "pet_deposit": random.choice([0, 250, 500]),
                "utilities_included": random.sample(["Water", "Trash", "Gas", "Electric", "Internet"], k=random.randint(0, 3)),
                "lease_type": random.choice(["Fixed Term", "Month-to-Month", "Sublease"])
            }
        elif data_type == "property_appraisal":
            content = {
                "appraisal_id": f"APP_{index:08d}",
                "property_address": self.faker.address(),
                "appraiser": self.faker.name(),
                "appraisal_date": self.faker.date_between(start_date="-6m", end_date="today").strftime(self.config.date_format),
                "property_type": random.choice(["Single Family", "Condo", "Townhouse"]),
                "market_value": random.randint(100000, 1500000),
                "land_value": random.randint(20000, 500000),
                "improvement_value": random.randint(80000, 1000000),
                "condition": random.choice(["Excellent", "Good", "Fair", "Poor"]),
                "comparable_sales": random.randint(3, 6),
                "appraisal_purpose": random.choice(["Purchase", "Refinance", "Estate", "Tax Assessment"])
            }
        else:
            content = {"type": data_type, "data": f"Generated real estate data {index}"}
        
        return GeneratedData(
            domain=DataDomain.REAL_ESTATE,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_insurance_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate insurance domain data"""
        if data_type == "policy":
            content = {
                "policy_number": f"POL_{index:08d}",
                "policy_type": random.choice(["Auto", "Home", "Life", "Health", "Disability"]),
                "policyholder": self.faker.name(),
                "effective_date": self.faker.date_between(start_date="-2y", end_date="today").strftime(self.config.date_format),
                "expiration_date": self.faker.date_between(start_date="today", end_date="+1y").strftime(self.config.date_format),
                "premium_amount": round(random.uniform(50, 5000), 2),
                "premium_frequency": random.choice(["Monthly", "Quarterly", "Semi-Annual", "Annual"]),
                "coverage_limit": random.randint(10000, 1000000),
                "deductible": random.choice([250, 500, 1000, 2500, 5000]),
                "agent": self.faker.name(),
                "status": random.choice(["Active", "Lapsed", "Cancelled", "Expired"])
            }
        elif data_type == "claim":
            content = {
                "claim_id": f"CLM_{index:08d}",
                "policy_number": f"POL_{random.randint(1, 10000):08d}",
                "claimant": self.faker.name(),
                "claim_date": self.faker.date_between(start_date="-1y", end_date="today").strftime(self.config.date_format),
                "incident_date": self.faker.date_between(start_date="-1y", end_date="today").strftime(self.config.date_format),
                "claim_type": random.choice(["Collision", "Theft", "Fire", "Water Damage", "Medical", "Liability"]),
                "claim_amount": round(random.uniform(100, 50000), 2),
                "approved_amount": round(random.uniform(0, 45000), 2),
                "status": random.choice(["Submitted", "Under Review", "Approved", "Denied", "Paid", "Closed"]),
                "adjuster": self.faker.name(),
                "description": self.faker.text(max_nb_chars=200)
            }
        elif data_type == "quote":
            content = {
                "quote_id": f"QTE_{index:08d}",
                "applicant_name": self.faker.name(),
                "quote_date": self.faker.date_time_between(start_date="-30d", end_date="now").isoformat(),
                "policy_type": random.choice(["Auto", "Home", "Life", "Health"]),
                "coverage_options": random.sample(["Basic", "Standard", "Premium", "Comprehensive"], k=random.randint(1, 3)),
                "annual_premium": round(random.uniform(500, 10000), 2),
                "valid_until": self.faker.date_between(start_date="today", end_date="+30d").strftime(self.config.date_format),
                "risk_score": random.randint(1, 100),
                "discount_applied": round(random.uniform(0, 25), 2),
                "agent": self.faker.name()
            }
        else:
            content = {"type": data_type, "data": f"Generated insurance data {index}"}
        
        return GeneratedData(
            domain=DataDomain.INSURANCE,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_manufacturing_data(self, data_type: str, index: int) -> GeneratedData:
        """Generate manufacturing domain data"""
        if data_type == "work_order":
            content = {
                "work_order_id": f"WO_{index:08d}",
                "product_id": f"PROD_{random.randint(1, 1000):06d}",
                "product_name": self.faker.catch_phrase(),
                "quantity": random.randint(10, 10000),
                "priority": random.choice(["Low", "Medium", "High", "Urgent"]),
                "start_date": self.faker.date_between(start_date="-30d", end_date="today").strftime(self.config.date_format),
                "due_date": self.faker.date_between(start_date="today", end_date="+60d").strftime(self.config.date_format),
                "production_line": f"Line-{random.randint(1, 10)}",
                "assigned_team": f"Team-{random.choice(['A', 'B', 'C', 'D'])}",
                "status": random.choice(["Scheduled", "In Progress", "Completed", "On Hold", "Cancelled"]),
                "completion_percentage": random.randint(0, 100)
            }
        elif data_type == "quality_inspection":
            content = {
                "inspection_id": f"QC_{index:08d}",
                "batch_number": f"BATCH_{random.randint(1000, 9999)}",
                "product_id": f"PROD_{random.randint(1, 1000):06d}",
                "inspection_date": self.faker.date_time_between(start_date="-7d", end_date="now").isoformat(),
                "inspector": self.faker.name(),
                "sample_size": random.randint(10, 100),
                "defects_found": random.randint(0, 10),
                "defect_rate": round(random.uniform(0, 5), 2),
                "passed": random.choice([True, False]),
                "measurements": {
                    "dimension_accuracy": round(random.uniform(95, 100), 2),
                    "weight_variance": round(random.uniform(0, 2), 2),
                    "surface_quality": random.choice(["Excellent", "Good", "Acceptable", "Poor"])
                },
                "notes": self.faker.text(max_nb_chars=200) if random.random() < 0.3 else None
            }
        elif data_type == "inventory_movement":
            content = {
                "movement_id": f"MOV_{index:08d}",
                "material_id": f"MAT_{random.randint(1, 5000):06d}",
                "material_name": self.faker.catch_phrase(),
                "movement_type": random.choice(["Receipt", "Issue", "Transfer", "Adjustment"]),
                "quantity": random.randint(1, 1000),
                "unit_of_measure": random.choice(["EA", "KG", "LB", "M", "L", "BOX"]),
                "from_location": f"WH-{random.randint(1, 5)}-{random.choice(['A', 'B', 'C'])}{random.randint(1, 20)}",
                "to_location": f"WH-{random.randint(1, 5)}-{random.choice(['A', 'B', 'C'])}{random.randint(1, 20)}",
                "movement_date": self.faker.date_time_between(start_date="-30d", end_date="now").isoformat(),
                "performed_by": self.faker.name(),
                "reference_doc": f"{random.choice(['PO', 'SO', 'WO'])}_{random.randint(10000, 99999)}"
            }
        else:
            content = {"type": data_type, "data": f"Generated manufacturing data {index}"}
        
        return GeneratedData(
            domain=DataDomain.MANUFACTURING,
            data_type=data_type,
            content=content,
            metadata={"generation_mode": self.config.generation_mode.value}
        )
    
    def _generate_account_number(self) -> str:
        """Generate realistic account number"""
        if self.config.anonymize_data:
            return f"****{random.randint(1000, 9999)}"
        else:
            return f"{random.randint(1000, 9999)}{random.randint(10000000, 99999999)}"
    
    def _generate_investment_holdings(self) -> List[Dict[str, Any]]:
        """Generate investment portfolio holdings"""
        stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        holdings = []
        
        for _ in range(random.randint(3, 8)):
            holding = {
                "symbol": random.choice(stocks),
                "shares": random.randint(1, 1000),
                "purchase_price": round(random.uniform(10, 500), 2),
                "current_price": round(random.uniform(10, 500), 2),
                "purchase_date": self.faker.date_between(start_date="-2y", end_date="today").strftime(self.config.date_format)
            }
            holdings.append(holding)
        
        return holdings
    
    def _generate_course_list(self) -> List[Dict[str, Any]]:
        """Generate list of courses for transcript"""
        courses = []
        departments = ['CS', 'MATH', 'ENG', 'BIO', 'HIST', 'PSYC', 'CHEM', 'PHYS']
        
        for _ in range(random.randint(20, 40)):
            course = {
                "course_code": f"{random.choice(departments)}{random.randint(100, 400)}",
                "course_name": self.faker.catch_phrase(),
                "credits": random.choice([1, 2, 3, 4]),
                "grade": random.choice(['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']),
                "semester": f"{random.choice(['Fall', 'Spring', 'Summer'])} {random.randint(2020, 2024)}"
            }
            courses.append(course)
        
        return courses
    
    def _generate_order_items(self) -> List[Dict[str, Any]]:
        """Generate order items for retail order"""
        items = []
        num_items = random.randint(1, 10)
        
        for _ in range(num_items):
            item = {
                "sku": f"SKU_{random.randint(1000, 9999)}",
                "name": self.faker.catch_phrase(),
                "quantity": random.randint(1, 5),
                "unit_price": round(random.uniform(5, 500), 2),
                "discount": round(random.uniform(0, 20), 2),
                "total": round(random.uniform(5, 1500), 2)
            }
            items.append(item)
        
        return items
    
    def get_available_data_types(self, domain: Optional[DataDomain] = None) -> Dict[str, List[str]]:
        """Get available data types for domain(s)"""
        domain = domain or self.config.domain
        
        data_types = {
            DataDomain.FINANCIAL: ["transaction", "loan_application", "investment_portfolio", "bank_statement"],
            DataDomain.HEALTHCARE: ["patient_record", "prescription", "lab_result", "insurance_claim"],
            DataDomain.LEGAL: ["contract", "case_filing", "legal_brief", "settlement_agreement"],
            DataDomain.GOVERNMENT: ["tax_return", "permit_application", "license_application", "court_filing"],
            DataDomain.EDUCATION: ["student_record", "transcript", "course_enrollment", "financial_aid"],
            DataDomain.RETAIL: ["order", "inventory", "customer", "return"],
            DataDomain.TECHNOLOGY: ["server_log", "bug_report", "deployment", "api_usage"],
            DataDomain.REAL_ESTATE: ["property_listing", "lease_agreement", "property_appraisal", "mortgage"],
            DataDomain.INSURANCE: ["policy", "claim", "quote", "risk_assessment"],
            DataDomain.MANUFACTURING: ["work_order", "quality_inspection", "inventory_movement", "production_report"]
        }
        
        if domain in data_types:
            return {domain.value: data_types[domain]}
        else:
            return {d.value: types for d, types in data_types.items()}


# Factory function
def create_domain_generator(**config_kwargs) -> DomainDataGenerator:
    """Factory function to create domain data generator"""
    config = DomainDataConfig(**config_kwargs)
    return DomainDataGenerator(config)