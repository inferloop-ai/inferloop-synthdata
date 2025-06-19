#!/usr/bin/env python3
"""
Entity Generator for creating realistic entities (names, addresses, etc.)
"""

import random
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from faker import Faker
from faker.providers import BaseProvider

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger


class EntityType(Enum):
    """Entity types for generation"""
    PERSON = "person"
    COMPANY = "company"
    ADDRESS = "address"
    FINANCIAL_ACCOUNT = "financial_account"
    MEDICAL_IDENTIFIER = "medical_identifier"
    LEGAL_ENTITY = "legal_entity"
    GOVERNMENT_ID = "government_id"
    CONTACT_INFO = "contact_info"


@dataclass
class GeneratedEntity:
    """Generated entity data"""
    entity_type: EntityType
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class EntityGeneratorConfig(BaseModel):
    """Entity generator configuration"""
    locale: str = Field("en_US", description="Locale for generation")
    anonymize: bool = Field(True, description="Anonymize sensitive data")
    consistency_seed: Optional[int] = Field(None, description="Seed for consistent generation")
    domain_specific: bool = Field(True, description="Use domain-specific patterns")


class CustomFinancialProvider(BaseProvider):
    """Custom provider for financial data"""
    
    def routing_number(self):
        """Generate valid routing number"""
        # US routing numbers are 9 digits with specific check digit algorithm
        return f"{random.randint(100000000, 999999999)}"
    
    def account_number(self, length=10):
        """Generate account number"""
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])
    
    def credit_card_security_code(self):
        """Generate CVV code"""
        return f"{random.randint(100, 999)}"


class CustomMedicalProvider(BaseProvider):
    """Custom provider for medical data"""
    
    def medical_record_number(self):
        """Generate medical record number"""
        return f"MRN{random.randint(100000, 999999)}"
    
    def npi_number(self):
        """Generate National Provider Identifier"""
        return f"{random.randint(1000000000, 9999999999)}"
    
    def insurance_group_number(self):
        """Generate insurance group number"""
        return f"{random.choice(['GRP', 'PLN', 'POL'])}{random.randint(100000, 999999)}"


class EntityGenerator:
    """
    Entity Generator for creating realistic entities
    
    Generates various types of entities including people, companies,
    addresses, and domain-specific identifiers.
    """
    
    def __init__(self, config: Optional[EntityGeneratorConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or EntityGeneratorConfig()
        
        # Initialize Faker
        self.faker = Faker(self.config.locale)
        if self.config.consistency_seed:
            Faker.seed(self.config.consistency_seed)
        
        # Add custom providers
        self.faker.add_provider(CustomFinancialProvider)
        self.faker.add_provider(CustomMedicalProvider)
        
        # Entity generators
        self.generators = {
            EntityType.PERSON: self._generate_person,
            EntityType.COMPANY: self._generate_company,
            EntityType.ADDRESS: self._generate_address,
            EntityType.FINANCIAL_ACCOUNT: self._generate_financial_account,
            EntityType.MEDICAL_IDENTIFIER: self._generate_medical_identifier,
            EntityType.LEGAL_ENTITY: self._generate_legal_entity,
            EntityType.GOVERNMENT_ID: self._generate_government_id,
            EntityType.CONTACT_INFO: self._generate_contact_info
        }
        
        self.logger.info("Entity Generator initialized")
    
    def generate_entity(self, entity_type: EntityType, **kwargs) -> GeneratedEntity:
        """Generate a single entity"""
        generator = self.generators.get(entity_type)
        if not generator:
            raise ValueError(f"No generator for entity type: {entity_type}")
        
        data = generator(**kwargs)
        
        return GeneratedEntity(
            entity_type=entity_type,
            data=data,
            metadata={
                "generated_with": "EntityGenerator",
                "locale": self.config.locale,
                "anonymized": self.config.anonymize
            }
        )
    
    def generate_entities(self, entity_type: EntityType, count: int, **kwargs) -> List[GeneratedEntity]:
        """Generate multiple entities"""
        return [self.generate_entity(entity_type, **kwargs) for _ in range(count)]
    
    def _generate_person(self, **kwargs) -> Dict[str, Any]:
        """Generate person entity"""
        gender = kwargs.get('gender', random.choice(['M', 'F']))
        
        if gender == 'M':
            first_name = self.faker.first_name_male()
            prefix = random.choice(['Mr.', 'Dr.', ''])
        else:
            first_name = self.faker.first_name_female()
            prefix = random.choice(['Ms.', 'Mrs.', 'Dr.', ''])
        
        person_data = {
            'prefix': prefix,
            'first_name': first_name,
            'middle_initial': random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            'last_name': self.faker.last_name(),
            'suffix': random.choice(['', 'Jr.', 'Sr.', 'II', 'III']) if random.random() < 0.1 else '',
            'gender': gender,
            'date_of_birth': self.faker.date_of_birth(minimum_age=18, maximum_age=80),
            'nationality': 'US',
            'preferred_name': first_name if random.random() < 0.8 else self.faker.first_name()
        }
        
        # Add full name
        name_parts = [p for p in [person_data['prefix'], person_data['first_name'], 
                                 person_data['middle_initial'] + '.', person_data['last_name'], 
                                 person_data['suffix']] if p]
        person_data['full_name'] = ' '.join(name_parts)
        
        return person_data
    
    def _generate_company(self, **kwargs) -> Dict[str, Any]:
        """Generate company entity"""
        company_type = kwargs.get('company_type', random.choice(['corporation', 'llc', 'partnership', 'sole_proprietorship']))
        
        base_name = self.faker.company()
        
        # Add entity type suffix
        if company_type == 'corporation':
            legal_name = f"{base_name} {'Inc.' if random.random() < 0.7 else 'Corp.'}"
        elif company_type == 'llc':
            legal_name = f"{base_name} LLC"
        else:
            legal_name = base_name
        
        return {
            'legal_name': legal_name,
            'dba_name': base_name if random.random() < 0.3 else legal_name,
            'company_type': company_type,
            'industry': self.faker.bs(),
            'ein': f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}",
            'state_of_incorporation': self.faker.state_abbr(),
            'founded_year': random.randint(1950, 2023),
            'website': f"www.{base_name.lower().replace(' ', '').replace(',', '').replace('.', '')}.com",
            'stock_symbol': ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4)) if company_type == 'corporation' and random.random() < 0.1 else None
        }
    
    def _generate_address(self, **kwargs) -> Dict[str, Any]:
        """Generate address entity"""
        address_type = kwargs.get('address_type', random.choice(['residential', 'commercial', 'po_box']))
        
        if address_type == 'po_box':
            return {
                'address_type': address_type,
                'po_box': f"PO Box {random.randint(1, 99999)}",
                'city': self.faker.city(),
                'state': self.faker.state_abbr(),
                'zip_code': self.faker.zipcode(),
                'country': 'USA'
            }
        else:
            addr = {
                'address_type': address_type,
                'street_number': str(random.randint(1, 9999)),
                'street_name': self.faker.street_name(),
                'street_suffix': self.faker.street_suffix(),
                'unit_number': f"#{random.randint(1, 999)}" if random.random() < 0.3 else None,
                'city': self.faker.city(),
                'state': self.faker.state(),
                'state_abbr': self.faker.state_abbr(),
                'zip_code': self.faker.zipcode(),
                'zip_plus_four': self.faker.zipcode_plus4(),
                'country': 'USA',
                'county': f"{self.faker.city()} County",
                'formatted_address': None
            }
            
            # Set formatted address
            unit = f" {addr['unit_number']}" if addr.get('unit_number') else ""
            addr['formatted_address'] = f"{addr['street_number']} {addr['street_name']} {addr['street_suffix']}{unit}, {addr['city']}, {addr['state_abbr']} {addr['zip_code']}"
            
            return addr
    
    def _generate_financial_account(self, **kwargs) -> Dict[str, Any]:
        """Generate financial account entity"""
        account_type = kwargs.get('account_type', random.choice(['checking', 'savings', 'credit_card', 'loan', 'investment']))
        
        base_data = {
            'account_type': account_type,
            'institution_name': random.choice(['First National Bank', 'Community Bank', 'Regional Credit Union', 'Metro Bank']),
            'routing_number': self.faker.routing_number(),
            'account_number': f"****{random.randint(1000, 9999)}" if self.config.anonymize else self.faker.account_number(),
            'account_status': random.choice(['active', 'closed', 'frozen']),
            'opening_date': self.faker.date_between(start_date='-10y', end_date='today')
        }
        
        if account_type == 'credit_card':
            base_data.update({
                'card_number': self.faker.credit_card_number(),
                'cvv': self.faker.credit_card_security_code(),
                'expiration_date': self.faker.credit_card_expire(),
                'credit_limit': random.randint(500, 50000),
                'card_type': random.choice(['Visa', 'Mastercard', 'American Express', 'Discover'])
            })
        
        return base_data
    
    def _generate_medical_identifier(self, **kwargs) -> Dict[str, Any]:
        """Generate medical identifier entity"""
        identifier_type = kwargs.get('identifier_type', random.choice(['mrn', 'npi', 'insurance', 'dea']))
        
        if identifier_type == 'mrn':
            return {
                'identifier_type': 'medical_record_number',
                'mrn': self.faker.medical_record_number(),
                'issuing_facility': f"{self.faker.city()} Medical Center",
                'issue_date': self.faker.date_between(start_date='-5y', end_date='today')
            }
        elif identifier_type == 'npi':
            return {
                'identifier_type': 'national_provider_identifier',
                'npi': self.faker.npi_number(),
                'provider_type': random.choice(['individual', 'organization']),
                'specialty': random.choice(['Family Medicine', 'Internal Medicine', 'Cardiology', 'Orthopedics'])
            }
        elif identifier_type == 'insurance':
            return {
                'identifier_type': 'insurance_identifier',
                'member_id': f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=9))}",
                'group_number': self.faker.insurance_group_number(),
                'plan_name': random.choice(['Premium Health', 'Basic Care', 'Family Plan', 'Individual Plan']),
                'insurance_company': random.choice(['Blue Cross Blue Shield', 'Aetna', 'Cigna', 'UnitedHealth'])
            }
        else:
            return {'identifier_type': identifier_type, 'value': f"{random.randint(100000000, 999999999)}"}
    
    def _generate_legal_entity(self, **kwargs) -> Dict[str, Any]:
        """Generate legal entity"""
        return {
            'entity_type': 'legal_entity',
            'bar_number': f"BAR{random.randint(100000, 999999)}",
            'attorney_name': f"{self.faker.name()}, Esq.",
            'law_firm': f"{self.faker.last_name()} & Associates",
            'practice_areas': random.sample(['Corporate', 'Litigation', 'Real Estate', 'Family', 'Criminal'], k=random.randint(1, 3)),
            'jurisdiction': self.faker.state(),
            'admission_date': self.faker.date_between(start_date='-20y', end_date='-1y')
        }
    
    def _generate_government_id(self, **kwargs) -> Dict[str, Any]:
        """Generate government identifier"""
        id_type = kwargs.get('id_type', random.choice(['ssn', 'ein', 'passport', 'drivers_license']))
        
        if id_type == 'ssn':
            ssn = self.faker.ssn()
            return {
                'id_type': 'social_security_number',
                'ssn': f"XXX-XX-{ssn[-4:]}" if self.config.anonymize else ssn,
                'issuing_authority': 'Social Security Administration'
            }
        elif id_type == 'passport':
            return {
                'id_type': 'passport',
                'passport_number': f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=9))}",
                'country': 'USA',
                'issue_date': self.faker.date_between(start_date='-10y', end_date='today'),
                'expiration_date': self.faker.date_between(start_date='today', end_date='+10y')
            }
        elif id_type == 'drivers_license':
            return {
                'id_type': 'drivers_license',
                'license_number': f"DL{random.randint(10000000, 99999999)}",
                'state': self.faker.state_abbr(),
                'class': random.choice(['Class A', 'Class B', 'Class C']),
                'issue_date': self.faker.date_between(start_date='-5y', end_date='today'),
                'expiration_date': self.faker.date_between(start_date='today', end_date='+5y')
            }
        else:
            return {'id_type': id_type, 'value': f"{random.randint(100000000, 999999999)}"}
    
    def _generate_contact_info(self, **kwargs) -> Dict[str, Any]:
        """Generate contact information"""
        return {
            'primary_phone': self.faker.phone_number(),
            'secondary_phone': self.faker.phone_number() if random.random() < 0.4 else None,
            'email': self.faker.email(),
            'work_email': self.faker.company_email() if random.random() < 0.6 else None,
            'preferred_contact': random.choice(['email', 'phone', 'mail']),
            'emergency_contact_name': self.faker.name(),
            'emergency_contact_phone': self.faker.phone_number(),
            'emergency_contact_relationship': random.choice(['spouse', 'parent', 'sibling', 'friend', 'other'])
        }


# Factory function
def create_entity_generator(**config_kwargs) -> EntityGenerator:
    """Factory function to create entity generator"""
    config = EntityGeneratorConfig(**config_kwargs)
    return EntityGenerator(config)