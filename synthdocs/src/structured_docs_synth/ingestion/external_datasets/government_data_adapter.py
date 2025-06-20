#!/usr/bin/env python3
"""
Government Data Adapter for government forms and documents
"""

import json
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date, timedelta
import random

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class GovernmentDocumentType(Enum):
    """Government document types"""
    TAX_FORM = "tax_form"
    PASSPORT_APPLICATION = "passport_application"
    VISA_APPLICATION = "visa_application"
    DRIVERS_LICENSE = "drivers_license"
    VOTER_REGISTRATION = "voter_registration"
    SOCIAL_SECURITY = "social_security"
    BIRTH_CERTIFICATE = "birth_certificate"
    DEATH_CERTIFICATE = "death_certificate"
    MARRIAGE_LICENSE = "marriage_license"
    BUSINESS_LICENSE = "business_license"
    PERMIT_APPLICATION = "permit_application"
    IMMIGRATION_FORM = "immigration_form"


class GovernmentAgency(Enum):
    """Government agencies"""
    IRS = "irs"
    STATE_DEPT = "state_department"
    DHS = "department_homeland_security"
    DOT = "department_transportation"
    SSA = "social_security_administration"
    USCIS = "us_citizenship_immigration"
    STATE_LOCAL = "state_local_government"
    COUNTY_CLERK = "county_clerk"
    CITY_HALL = "city_hall"


@dataclass
class PersonInfo:
    """Personal information (anonymized)"""
    first_name: str
    last_name: str
    middle_initial: str = ""
    date_of_birth: date = field(default_factory=lambda: date(1980, 1, 1))
    ssn_last4: str = "0000"  # Only last 4 digits
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    phone: str = ""
    email: str = ""


@dataclass
class GovernmentDocument:
    """Government document record"""
    document_id: str
    document_type: GovernmentDocumentType
    agency: GovernmentAgency
    form_number: str
    applicant_info: PersonInfo
    submission_date: datetime
    form_data: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GovernmentDataConfig(BaseModel):
    """Government data adapter configuration"""
    
    # Document generation
    num_documents: int = Field(500, description="Number of documents to generate")
    document_types: List[GovernmentDocumentType] = Field(
        default=[GovernmentDocumentType.TAX_FORM, GovernmentDocumentType.PASSPORT_APPLICATION],
        description="Document types to generate"
    )
    
    # Privacy settings
    anonymize_data: bool = Field(True, description="Anonymize personal data")
    use_fake_names: bool = Field(True, description="Use fake names")
    mask_ssn: bool = Field(True, description="Mask SSN numbers")
    
    # Realism settings
    realistic_addresses: bool = Field(True, description="Generate realistic addresses")
    realistic_dates: bool = Field(True, description="Use realistic date ranges")
    include_validation_errors: bool = Field(False, description="Include common form errors")
    
    # Content settings
    include_text_content: bool = Field(True, description="Generate text content")
    include_structured_data: bool = Field(True, description="Include structured data")
    
    # Geographic settings
    states: List[str] = Field(
        default=["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
        description="US states to use"
    )
    
    # Date ranges
    start_year: int = Field(2020, description="Start year for documents")
    end_year: int = Field(2024, description="End year for documents")


class GovernmentDataAdapter:
    """
    Government Data Adapter for government forms and documents
    
    Generates realistic government forms including tax returns, passport
    applications, licenses, and permits with proper anonymization.
    """
    
    def __init__(self, config: Optional[GovernmentDataConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or GovernmentDataConfig()
        
        # Sample data for generation
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Mary",
            "James", "Jennifer", "William", "Linda", "Richard", "Elizabeth", "Joseph"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"
        ]
        
        self.cities = {
            "CA": ["Los Angeles", "San Francisco", "San Diego", "Sacramento"],
            "NY": ["New York", "Buffalo", "Rochester", "Albany"],
            "TX": ["Houston", "Dallas", "Austin", "San Antonio"],
            "FL": ["Miami", "Tampa", "Orlando", "Jacksonville"]
        }
        
        self.form_numbers = {
            GovernmentDocumentType.TAX_FORM: ["1040", "1040EZ", "1040A", "1099"],
            GovernmentDocumentType.PASSPORT_APPLICATION: ["DS-11", "DS-82"],
            GovernmentDocumentType.VISA_APPLICATION: ["DS-160", "DS-156"],
            GovernmentDocumentType.DRIVERS_LICENSE: ["DL-44", "DL-180"],
            GovernmentDocumentType.BUSINESS_LICENSE: ["BL-100", "BL-200"]
        }
        
        self.logger.info("Government Data Adapter initialized")
    
    def load_data(self, **kwargs) -> List[GovernmentDocument]:
        """Load government document data"""
        start_time = time.time()
        
        try:
            documents = self._generate_government_documents()
            
            loading_time = time.time() - start_time
            self.logger.info(f"Generated {len(documents)} government documents in {loading_time:.2f}s")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to generate government data: {e}")
            raise ProcessingError(f"Government data generation error: {e}")
    
    def _generate_government_documents(self) -> List[GovernmentDocument]:
        """Generate government documents"""
        documents = []
        
        for i in range(self.config.num_documents):
            doc_type = random.choice(self.config.document_types)
            document = self._generate_document(i, doc_type)
            documents.append(document)
        
        return documents
    
    def _generate_document(self, index: int, doc_type: GovernmentDocumentType) -> GovernmentDocument:
        """Generate a single government document"""
        # Generate applicant info
        applicant = self._generate_person_info()
        
        # Determine agency
        agency = self._get_agency_for_document_type(doc_type)
        
        # Generate form number
        form_numbers = self.form_numbers.get(doc_type, ["FORM-001"])
        form_number = random.choice(form_numbers)
        
        # Generate submission date
        submission_date = datetime(
            random.randint(self.config.start_year, self.config.end_year),
            random.randint(1, 12),
            random.randint(1, 28),
            random.randint(9, 17),  # Business hours
            random.randint(0, 59)
        )
        
        # Generate form data
        form_data = self._generate_form_data(doc_type, applicant)
        
        document = GovernmentDocument(
            document_id=f"{doc_type.value}_{index:06d}",
            document_type=doc_type,
            agency=agency,
            form_number=form_number,
            applicant_info=applicant,
            submission_date=submission_date,
            form_data=form_data,
            metadata={
                "processing_status": random.choice(["pending", "approved", "under_review"]),
                "filing_method": random.choice(["online", "mail", "in_person"]),
                "language": "English"
            }
        )
        
        # Generate content
        if self.config.include_text_content:
            document.text_content = self._generate_text_content(document)
        
        if self.config.include_structured_data:
            document.structured_data = self._generate_structured_data(document)
        
        return document
    
    def _generate_person_info(self) -> PersonInfo:
        """Generate person information"""
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        
        # Generate address
        state = random.choice(self.config.states)
        cities = self.cities.get(state, ["Unknown City"])
        city = random.choice(cities)
        
        address = f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'First', 'Second', 'Park'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr'])}"
        zip_code = f"{random.randint(10000, 99999)}"
        
        # Generate contact info
        phone = f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
        email = f"{first_name.lower()}.{last_name.lower()}@example.com" if not self.config.anonymize_data else "user@example.com"
        
        # Generate birth date
        birth_year = random.randint(1950, 2000)
        birth_date = date(birth_year, random.randint(1, 12), random.randint(1, 28))
        
        return PersonInfo(
            first_name=first_name,
            last_name=last_name,
            middle_initial=random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            date_of_birth=birth_date,
            ssn_last4=f"{random.randint(1000, 9999)}" if not self.config.mask_ssn else "XXXX",
            address=address,
            city=city,
            state=state,
            zip_code=zip_code,
            phone=phone,
            email=email
        )
    
    def _get_agency_for_document_type(self, doc_type: GovernmentDocumentType) -> GovernmentAgency:
        """Get appropriate agency for document type"""
        agency_mapping = {
            GovernmentDocumentType.TAX_FORM: GovernmentAgency.IRS,
            GovernmentDocumentType.PASSPORT_APPLICATION: GovernmentAgency.STATE_DEPT,
            GovernmentDocumentType.VISA_APPLICATION: GovernmentAgency.STATE_DEPT,
            GovernmentDocumentType.DRIVERS_LICENSE: GovernmentAgency.DOT,
            GovernmentDocumentType.SOCIAL_SECURITY: GovernmentAgency.SSA,
            GovernmentDocumentType.IMMIGRATION_FORM: GovernmentAgency.USCIS,
            GovernmentDocumentType.BIRTH_CERTIFICATE: GovernmentAgency.COUNTY_CLERK,
            GovernmentDocumentType.MARRIAGE_LICENSE: GovernmentAgency.COUNTY_CLERK,
            GovernmentDocumentType.BUSINESS_LICENSE: GovernmentAgency.CITY_HALL
        }
        
        return agency_mapping.get(doc_type, GovernmentAgency.STATE_LOCAL)
    
    def _generate_form_data(self, doc_type: GovernmentDocumentType, applicant: PersonInfo) -> Dict[str, Any]:
        """Generate form-specific data"""
        base_data = {
            "first_name": applicant.first_name,
            "last_name": applicant.last_name,
            "date_of_birth": applicant.date_of_birth.isoformat(),
            "address": applicant.address,
            "city": applicant.city,
            "state": applicant.state,
            "zip_code": applicant.zip_code
        }
        
        if doc_type == GovernmentDocumentType.TAX_FORM:
            return {**base_data, **self._generate_tax_form_data()}
        elif doc_type == GovernmentDocumentType.PASSPORT_APPLICATION:
            return {**base_data, **self._generate_passport_data()}
        elif doc_type == GovernmentDocumentType.DRIVERS_LICENSE:
            return {**base_data, **self._generate_drivers_license_data()}
        else:
            return base_data
    
    def _generate_tax_form_data(self) -> Dict[str, Any]:
        """Generate tax form specific data"""
        return {
            "filing_status": random.choice(["single", "married_filing_jointly", "head_of_household"]),
            "wages": random.randint(30000, 120000),
            "tax_withheld": random.randint(3000, 15000),
            "dependents": random.randint(0, 3),
            "agi": random.randint(25000, 100000),
            "tax_owed": random.randint(0, 5000),
            "refund": random.randint(0, 3000)
        }
    
    def _generate_passport_data(self) -> Dict[str, Any]:
        """Generate passport application data"""
        return {
            "place_of_birth": random.choice(["New York, NY", "Los Angeles, CA", "Chicago, IL"]),
            "citizenship": "United States",
            "travel_date": (datetime.now().date() + timedelta(days=random.randint(30, 365))).isoformat(),
            "destination_country": random.choice(["France", "Germany", "Japan", "Canada", "Mexico"]),
            "emergency_contact": f"{random.choice(self.first_names)} {random.choice(self.last_names)}",
            "previous_passport": random.choice([True, False]),
            "expedition_service": random.choice(["standard", "expedited"])
        }
    
    def _generate_drivers_license_data(self) -> Dict[str, Any]:
        """Generate drivers license data"""
        return {
            "license_class": random.choice(["Class C", "Class B", "Class A"]),
            "height": f"5'{random.randint(4, 11)}\"",
            "weight": random.randint(120, 250),
            "eye_color": random.choice(["Brown", "Blue", "Green", "Hazel"]),
            "hair_color": random.choice(["Brown", "Black", "Blonde", "Red", "Gray"]),
            "restrictions": random.choice(["None", "Corrective Lenses", "Daylight Only"]),
            "organ_donor": random.choice([True, False]),
            "motorcycle_endorsement": random.choice([True, False])
        }
    
    def _generate_text_content(self, document: GovernmentDocument) -> str:
        """Generate text content for document"""
        if document.document_type == GovernmentDocumentType.TAX_FORM:
            return self._generate_tax_form_text(document)
        elif document.document_type == GovernmentDocumentType.PASSPORT_APPLICATION:
            return self._generate_passport_text(document)
        else:
            return self._generate_generic_form_text(document)
    
    def _generate_tax_form_text(self, document: GovernmentDocument) -> str:
        """Generate tax form text"""
        applicant = document.applicant_info
        form_data = document.form_data
        
        text = f"""DEPARTMENT OF THE TREASURY - INTERNAL REVENUE SERVICE
U.S. Individual Income Tax Return - Form {document.form_number}
Tax Year: {document.submission_date.year - 1}

TAXPAYER INFORMATION:
Name: {applicant.first_name} {applicant.middle_initial} {applicant.last_name}
Address: {applicant.address}
City, State, ZIP: {applicant.city}, {applicant.state} {applicant.zip_code}
SSN: XXX-XX-{applicant.ssn_last4}

FILING STATUS: {form_data.get('filing_status', 'Unknown').replace('_', ' ').title()}

INCOME:
Wages, salaries, tips: ${form_data.get('wages', 0):,}
Adjusted Gross Income: ${form_data.get('agi', 0):,}

TAX COMPUTATION:
Total Tax: ${form_data.get('tax_owed', 0):,}
Federal Tax Withheld: ${form_data.get('tax_withheld', 0):,}
Refund: ${form_data.get('refund', 0):,}

Date Filed: {document.submission_date.strftime('%m/%d/%Y')}"""
        
        return text
    
    def _generate_passport_text(self, document: GovernmentDocument) -> str:
        """Generate passport application text"""
        applicant = document.applicant_info
        form_data = document.form_data
        
        text = f"""U.S. DEPARTMENT OF STATE
APPLICALION FOR A U.S. PASSPORT - Form {document.form_number}

APPLICANT INFORMATION:
Name: {applicant.first_name} {applicant.middle_initial} {applicant.last_name}
Date of Birth: {applicant.date_of_birth.strftime('%m/%d/%Y')}
Place of Birth: {form_data.get('place_of_birth', 'Unknown')}
Citizenship: {form_data.get('citizenship', 'United States')}

CONTACT INFORMATION:
Mailing Address: {applicant.address}
City, State, ZIP: {applicant.city}, {applicant.state} {applicant.zip_code}
Phone: {applicant.phone}
Email: {applicant.email}

TRAVEL INFORMATION:
Intended Travel Date: {form_data.get('travel_date', 'Unknown')}
Destination: {form_data.get('destination_country', 'Unknown')}
Expedited Service: {form_data.get('expedition_service', 'standard').title()}

EMERGENCY CONTACT: {form_data.get('emergency_contact', 'Not provided')}

Application Date: {document.submission_date.strftime('%m/%d/%Y')}"""
        
        return text
    
    def _generate_generic_form_text(self, document: GovernmentDocument) -> str:
        """Generate generic government form text"""
        applicant = document.applicant_info
        
        text = f"""GOVERNMENT FORM - {document.form_number}
{document.agency.value.replace('_', ' ').upper()}

Form Type: {document.document_type.value.replace('_', ' ').title()}
Submission Date: {document.submission_date.strftime('%m/%d/%Y')}

APPLICANT INFORMATION:
Name: {applicant.first_name} {applicant.middle_initial} {applicant.last_name}
Date of Birth: {applicant.date_of_birth.strftime('%m/%d/%Y')}
Address: {applicant.address}
City, State, ZIP: {applicant.city}, {applicant.state} {applicant.zip_code}

Status: {document.metadata.get('processing_status', 'Unknown').title()}"""
        
        return text
    
    def _generate_structured_data(self, document: GovernmentDocument) -> Dict[str, Any]:
        """Generate structured data representation"""
        return {
            "document_info": {
                "type": document.document_type.value,
                "form_number": document.form_number,
                "agency": document.agency.value,
                "submission_date": document.submission_date.isoformat()
            },
            "applicant": {
                "name": f"{document.applicant_info.first_name} {document.applicant_info.last_name}",
                "date_of_birth": document.applicant_info.date_of_birth.isoformat(),
                "location": f"{document.applicant_info.city}, {document.applicant_info.state}"
            },
            "form_data": document.form_data,
            "metadata": document.metadata
        }
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information"""
        return {
            "adapter_type": "government_data",
            "supported_document_types": [doc_type.value for doc_type in GovernmentDocumentType],
            "supported_agencies": [agency.value for agency in GovernmentAgency],
            "num_documents": self.config.num_documents,
            "anonymized": self.config.anonymize_data,
            "year_range": (self.config.start_year, self.config.end_year)
        }


# Factory function
def create_government_adapter(**config_kwargs) -> GovernmentDataAdapter:
    """Factory function to create government data adapter"""
    config = GovernmentDataConfig(**config_kwargs)
    return GovernmentDataAdapter(config)