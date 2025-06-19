#!/usr/bin/env python3
"""
Legal Data Adapter for legal documents and contracts
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


class LegalDocumentType(Enum):
    """Legal document types"""
    CONTRACT = "contract"
    LEASE_AGREEMENT = "lease_agreement"
    NDA = "non_disclosure_agreement"
    EMPLOYMENT_CONTRACT = "employment_contract"
    POWER_OF_ATTORNEY = "power_of_attorney"
    WILL_TESTAMENT = "will_testament"
    COURT_FILING = "court_filing"
    LEGAL_BRIEF = "legal_brief"
    PATENT_APPLICATION = "patent_application"
    CORPORATE_BYLAWS = "corporate_bylaws"
    MERGER_AGREEMENT = "merger_agreement"
    SETTLEMENT_AGREEMENT = "settlement_agreement"


class LegalPracticeArea(Enum):
    """Legal practice areas"""
    CORPORATE_LAW = "corporate_law"
    REAL_ESTATE = "real_estate"
    EMPLOYMENT_LAW = "employment_law"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    LITIGATION = "litigation"
    FAMILY_LAW = "family_law"
    CRIMINAL_LAW = "criminal_law"
    TAX_LAW = "tax_law"
    IMMIGRATION_LAW = "immigration_law"
    ENVIRONMENTAL_LAW = "environmental_law"


@dataclass
class LegalParty:
    """Legal party information"""
    party_id: str
    name: str
    entity_type: str  # individual, corporation, llc, etc.
    address: str = ""
    state_of_incorporation: str = ""
    represented_by: str = ""


@dataclass
class LegalDocument:
    """Legal document record"""
    document_id: str
    document_type: LegalDocumentType
    practice_area: LegalPracticeArea
    parties: List[LegalParty]
    execution_date: datetime
    jurisdiction: str
    legal_data: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LegalDataConfig(BaseModel):
    """Legal data adapter configuration"""
    
    # Document generation
    num_documents: int = Field(300, description="Number of documents to generate")
    document_types: List[LegalDocumentType] = Field(
        default=[LegalDocumentType.CONTRACT, LegalDocumentType.NDA],
        description="Document types to generate"
    )
    
    # Privacy settings
    anonymize_data: bool = Field(True, description="Anonymize party data")
    use_fake_entities: bool = Field(True, description="Use fake entity names")
    
    # Legal realism
    realistic_terms: bool = Field(True, description="Use realistic legal terms")
    realistic_jurisdictions: bool = Field(True, description="Use real jurisdictions")
    include_standard_clauses: bool = Field(True, description="Include standard legal clauses")
    
    # Content settings
    include_text_content: bool = Field(True, description="Generate text content")
    include_structured_data: bool = Field(True, description="Include structured data")
    
    # Legal settings
    practice_areas: List[LegalPracticeArea] = Field(
        default=[LegalPracticeArea.CORPORATE_LAW, LegalPracticeArea.REAL_ESTATE],
        description="Legal practice areas to include"
    )
    
    # Geographic settings
    jurisdictions: List[str] = Field(
        default=["Delaware", "New York", "California", "Texas", "Florida"],
        description="Legal jurisdictions"
    )
    
    # Date ranges
    start_year: int = Field(2020, description="Start year for documents")
    end_year: int = Field(2024, description="End year for documents")


class LegalDataAdapter:
    """
    Legal Data Adapter for legal documents and contracts
    
    Generates realistic legal documents including contracts, NDAs,
    court filings, and corporate documents with proper legal formatting.
    """
    
    def __init__(self, config: Optional[LegalDataConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or LegalDataConfig()
        
        # Legal data templates
        self.company_names = [
            "Tech Solutions Inc.", "Global Industries LLC", "Innovative Systems Corp.",
            "Advanced Technologies Ltd.", "Strategic Partners Group", "Digital Ventures Inc.",
            "Premier Services Corporation", "Integrated Solutions LLC", "Dynamic Enterprises",
            "Synergy Partners Inc.", "Alpha Corporation", "Beta Industries LLC"
        ]
        
        self.individual_names = [
            "John Smith", "Jane Johnson", "Michael Brown", "Sarah Davis", "Robert Miller",
            "Lisa Wilson", "David Garcia", "Maria Rodriguez", "James Martinez", "Jennifer Lopez"
        ]
        
        self.law_firms = [
            "Smith & Associates", "Johnson Legal Group", "Brown, Davis & Partners",
            "Miller Law Firm", "Wilson & Garcia LLP", "Rodriguez Legal Services",
            "Martinez & Lopez PC", "Premier Legal Group", "Corporate Law Partners"
        ]
        
        self.contract_types = {
            LegalDocumentType.CONTRACT: [
                "Service Agreement", "Supply Agreement", "Consulting Agreement",
                "License Agreement", "Distribution Agreement"
            ],
            LegalDocumentType.EMPLOYMENT_CONTRACT: [
                "Executive Employment Agreement", "At-Will Employment Contract",
                "Contractor Agreement", "Consulting Agreement"
            ],
            LegalDocumentType.LEASE_AGREEMENT: [
                "Commercial Lease", "Residential Lease", "Equipment Lease",
                "Office Space Lease"
            ]
        }
        
        self.standard_clauses = [
            "Force Majeure", "Confidentiality", "Indemnification", "Limitation of Liability",
            "Governing Law", "Dispute Resolution", "Termination", "Assignment",
            "Entire Agreement", "Severability", "Amendment", "Notice Provisions"
        ]
        
        self.logger.info("Legal Data Adapter initialized")
    
    def load_data(self, **kwargs) -> List[LegalDocument]:
        """Load legal document data"""
        start_time = time.time()
        
        try:
            documents = self._generate_legal_documents()
            
            loading_time = time.time() - start_time
            self.logger.info(f"Generated {len(documents)} legal documents in {loading_time:.2f}s")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to generate legal data: {e}")
            raise ProcessingError(f"Legal data generation error: {e}")
    
    def _generate_legal_documents(self) -> List[LegalDocument]:
        """Generate legal documents"""
        documents = []
        
        for i in range(self.config.num_documents):
            doc_type = random.choice(self.config.document_types)
            document = self._generate_document(i, doc_type)
            documents.append(document)
        
        return documents
    
    def _generate_document(self, index: int, doc_type: LegalDocumentType) -> LegalDocument:
        """Generate a single legal document"""
        # Generate parties
        parties = self._generate_parties(doc_type)
        
        # Determine practice area
        practice_area = self._get_practice_area_for_document_type(doc_type)
        
        # Generate execution date
        execution_date = datetime(
            random.randint(self.config.start_year, self.config.end_year),
            random.randint(1, 12),
            random.randint(1, 28),
            random.randint(9, 17),  # Business hours
            random.randint(0, 59)
        )
        
        # Select jurisdiction
        jurisdiction = random.choice(self.config.jurisdictions)
        
        # Generate legal data
        legal_data = self._generate_legal_data(doc_type, parties)
        
        document = LegalDocument(
            document_id=f"{doc_type.value}_{index:06d}",
            document_type=doc_type,
            practice_area=practice_area,
            parties=parties,
            execution_date=execution_date,
            jurisdiction=jurisdiction,
            legal_data=legal_data,
            metadata={
                "document_status": random.choice(["executed", "draft", "under_review", "expired"]),
                "confidentiality_level": random.choice(["public", "confidential", "attorney_client_privileged"]),
                "pages": random.randint(1, 50),
                "legal_review": random.choice(["complete", "pending", "not_required"])
            }
        )
        
        # Generate content
        if self.config.include_text_content:
            document.text_content = self._generate_text_content(document)
        
        if self.config.include_structured_data:
            document.structured_data = self._generate_structured_data(document)
        
        return document
    
    def _generate_parties(self, doc_type: LegalDocumentType) -> List[LegalParty]:
        """Generate parties for legal document"""
        parties = []
        
        # Determine number of parties based on document type
        if doc_type in [LegalDocumentType.CONTRACT, LegalDocumentType.NDA, LegalDocumentType.LEASE_AGREEMENT]:
            num_parties = 2
        elif doc_type in [LegalDocumentType.MERGER_AGREEMENT, LegalDocumentType.SETTLEMENT_AGREEMENT]:
            num_parties = random.randint(2, 4)
        else:
            num_parties = random.randint(1, 3)
        
        for i in range(num_parties):
            # Determine if party is individual or entity
            is_entity = random.random() < 0.7  # 70% chance of entity
            
            if is_entity:
                name = random.choice(self.company_names)
                entity_type = random.choice(["Corporation", "LLC", "Partnership", "LLP"])
                state_of_incorporation = random.choice(self.config.jurisdictions)
            else:
                name = random.choice(self.individual_names)
                entity_type = "Individual"
                state_of_incorporation = ""
            
            party = LegalParty(
                party_id=f"PARTY_{i+1}",
                name=name,
                entity_type=entity_type,
                address=f"{random.randint(100, 9999)} Business Ave, City, State 12345",
                state_of_incorporation=state_of_incorporation,
                represented_by=random.choice(self.law_firms) if random.random() < 0.6 else ""
            )
            
            parties.append(party)
        
        return parties
    
    def _get_practice_area_for_document_type(self, doc_type: LegalDocumentType) -> LegalPracticeArea:
        """Get practice area for document type"""
        practice_area_mapping = {
            LegalDocumentType.CONTRACT: LegalPracticeArea.CORPORATE_LAW,
            LegalDocumentType.LEASE_AGREEMENT: LegalPracticeArea.REAL_ESTATE,
            LegalDocumentType.EMPLOYMENT_CONTRACT: LegalPracticeArea.EMPLOYMENT_LAW,
            LegalDocumentType.NDA: LegalPracticeArea.CORPORATE_LAW,
            LegalDocumentType.PATENT_APPLICATION: LegalPracticeArea.INTELLECTUAL_PROPERTY,
            LegalDocumentType.COURT_FILING: LegalPracticeArea.LITIGATION,
            LegalDocumentType.WILL_TESTAMENT: LegalPracticeArea.FAMILY_LAW,
            LegalDocumentType.MERGER_AGREEMENT: LegalPracticeArea.CORPORATE_LAW
        }
        
        return practice_area_mapping.get(doc_type, random.choice(self.config.practice_areas))
    
    def _generate_legal_data(self, doc_type: LegalDocumentType, parties: List[LegalParty]) -> Dict[str, Any]:
        """Generate legal data based on document type"""
        base_data = {
            "parties": [party.name for party in parties],
            "clauses": random.sample(self.standard_clauses, random.randint(3, 8)),
            "effective_date": (datetime.now() + timedelta(days=random.randint(0, 30))).date().isoformat(),
            "expiration_date": (datetime.now() + timedelta(days=random.randint(365, 1095))).date().isoformat()
        }
        
        if doc_type == LegalDocumentType.CONTRACT:
            return {**base_data, **self._generate_contract_data()}
        elif doc_type == LegalDocumentType.NDA:
            return {**base_data, **self._generate_nda_data()}
        elif doc_type == LegalDocumentType.EMPLOYMENT_CONTRACT:
            return {**base_data, **self._generate_employment_data()}
        elif doc_type == LegalDocumentType.LEASE_AGREEMENT:
            return {**base_data, **self._generate_lease_data()}
        else:
            return base_data
    
    def _generate_contract_data(self) -> Dict[str, Any]:
        """Generate contract-specific data"""
        return {
            "contract_value": random.randint(10000, 1000000),
            "payment_terms": random.choice(["Net 30", "Net 60", "Upon completion", "Monthly"]),
            "deliverables": [
                "Software development services",
                "Technical documentation",
                "Training materials",
                "Support services"
            ][:random.randint(1, 4)],
            "intellectual_property": random.choice(["Client retains", "Contractor retains", "Shared ownership"]),
            "termination_clause": "Either party may terminate with 30 days written notice"
        }
    
    def _generate_nda_data(self) -> Dict[str, Any]:
        """Generate NDA-specific data"""
        return {
            "confidentiality_period": f"{random.randint(2, 10)} years",
            "permitted_disclosures": [
                "To employees with need to know",
                "To legal advisors",
                "As required by law"
            ],
            "return_of_materials": "Within 30 days of termination",
            "remedies": ["Injunctive relief", "Monetary damages"],
            "mutual_nda": random.choice([True, False])
        }
    
    def _generate_employment_data(self) -> Dict[str, Any]:
        """Generate employment contract data"""
        return {
            "position": random.choice(["Software Engineer", "Manager", "Director", "Consultant"]),
            "salary": random.randint(50000, 300000),
            "benefits": [
                "Health insurance", "401(k) matching", "Vacation time", "Stock options"
            ][:random.randint(2, 4)],
            "non_compete_period": f"{random.randint(6, 24)} months",
            "at_will_employment": random.choice([True, False]),
            "stock_options": random.randint(0, 10000)
        }
    
    def _generate_lease_data(self) -> Dict[str, Any]:
        """Generate lease agreement data"""
        return {
            "property_address": f"{random.randint(100, 9999)} Commercial Blvd, City, State",
            "lease_term": f"{random.randint(1, 10)} years",
            "monthly_rent": random.randint(1000, 20000),
            "security_deposit": random.randint(1000, 10000),
            "utilities_included": random.choice([True, False]),
            "renewal_option": random.choice([True, False]),
            "permitted_use": random.choice(["Office space", "Retail", "Warehouse", "Mixed use"])
        }
    
    def _generate_text_content(self, document: LegalDocument) -> str:
        """Generate text content for document"""
        if document.document_type == LegalDocumentType.CONTRACT:
            return self._generate_contract_text(document)
        elif document.document_type == LegalDocumentType.NDA:
            return self._generate_nda_text(document)
        elif document.document_type == LegalDocumentType.EMPLOYMENT_CONTRACT:
            return self._generate_employment_text(document)
        else:
            return self._generate_generic_legal_text(document)
    
    def _generate_contract_text(self, document: LegalDocument) -> str:
        """Generate contract text"""
        parties = document.parties
        legal_data = document.legal_data
        
        text = f"""SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into on {document.execution_date.strftime('%B %d, %Y')}, by and between:

{parties[0].name}, a {parties[0].entity_type} ("Client")
and
{parties[1].name}, a {parties[1].entity_type} ("Contractor")

WHEREAS, Client desires to engage Contractor to provide certain services;
WHEREAS, Contractor agrees to provide such services under the terms set forth herein;

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

1. SERVICES
Contractor shall provide the following services: {', '.join(legal_data.get('deliverables', []))}

2. COMPENSATION
Client shall pay Contractor ${legal_data.get('contract_value', 0):,} for the services.
Payment Terms: {legal_data.get('payment_terms', 'Net 30')}

3. INTELLECTUAL PROPERTY
{legal_data.get('intellectual_property', 'As specified in Schedule A')}

4. TERMINATION
{legal_data.get('termination_clause', 'Standard termination provisions apply')}

5. GOVERNING LAW
This Agreement shall be governed by the laws of {document.jurisdiction}.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

{parties[0].name}                    {parties[1].name}

By: _________________               By: _________________
Name:                              Name:
Title:                             Title:
Date:                              Date:"""
        
        return text
    
    def _generate_nda_text(self, document: LegalDocument) -> str:
        """Generate NDA text"""
        parties = document.parties
        legal_data = document.legal_data
        
        text = f"""NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into on {document.execution_date.strftime('%B %d, %Y')}, by and between:

{parties[0].name} ("Disclosing Party")
and
{parties[1].name} ("Receiving Party")

WHEREAS, the parties wish to explore a potential business relationship;
WHEREAS, in connection with such discussions, Disclosing Party may share confidential information;

NOW, THEREFORE, the parties agree as follows:

1. CONFIDENTIAL INFORMATION
All information disclosed by Disclosing Party shall be deemed confidential.

2. OBLIGATIONS
Receiving Party agrees to:
a) Maintain confidentiality of all disclosed information
b) Use information solely for evaluation purposes
c) Limit access to employees with need to know

3. PERMITTED DISCLOSURES
Information may be disclosed:
{chr(10).join(f'- {disclosure}' for disclosure in legal_data.get('permitted_disclosures', []))}

4. TERM
This Agreement shall remain in effect for {legal_data.get('confidentiality_period', '5 years')}.

5. RETURN OF MATERIALS
All materials shall be returned {legal_data.get('return_of_materials', 'upon request')}.

6. REMEDIES
Breach may result in: {', '.join(legal_data.get('remedies', ['damages']))}

7. GOVERNING LAW
This Agreement shall be governed by the laws of {document.jurisdiction}.

IN WITNESS WHEREOF, the parties have executed this Agreement.

{parties[0].name}                    {parties[1].name}

Signature: _______________           Signature: _______________
Date: _______________               Date: _______________"""
        
        return text
    
    def _generate_employment_text(self, document: LegalDocument) -> str:
        """Generate employment contract text"""
        parties = document.parties
        legal_data = document.legal_data
        
        text = f"""EMPLOYMENT AGREEMENT

This Employment Agreement is entered into on {document.execution_date.strftime('%B %d, %Y')}, between:

{parties[0].name} ("Company")
and
{parties[1].name} ("Employee")

1. POSITION AND DUTIES
Employee shall serve as {legal_data.get('position', 'Employee')} and perform duties as assigned.

2. COMPENSATION
Base Salary: ${legal_data.get('salary', 0):,} per year
Benefits: {', '.join(legal_data.get('benefits', []))}

3. AT-WILL EMPLOYMENT
Employment is at-will: {legal_data.get('at_will_employment', True)}

4. NON-COMPETE
Employee agrees not to compete for {legal_data.get('non_compete_period', '12 months')} after termination.

5. STOCK OPTIONS
Employee granted {legal_data.get('stock_options', 0)} stock options subject to vesting.

6. GOVERNING LAW
This Agreement is governed by {document.jurisdiction} law.

Employee Signature: _______________    Company Signature: _______________
Date: _______________               Date: _______________"""
        
        return text
    
    def _generate_generic_legal_text(self, document: LegalDocument) -> str:
        """Generate generic legal document text"""
        parties_text = "\n".join([f"{party.name} - {party.entity_type}" for party in document.parties])
        
        text = f"""LEGAL DOCUMENT
{document.document_type.value.replace('_', ' ').upper()}

Document Date: {document.execution_date.strftime('%B %d, %Y')}
Jurisdiction: {document.jurisdiction}
Practice Area: {document.practice_area.value.replace('_', ' ').title()}

PARTIES:
{parties_text}

Document Status: {document.metadata.get('document_status', 'Unknown')}
Confidentiality: {document.metadata.get('confidentiality_level', 'Standard')}
Pages: {document.metadata.get('pages', 'Unknown')}

This document contains legal provisions binding upon all parties."""
        
        return text
    
    def _generate_structured_data(self, document: LegalDocument) -> Dict[str, Any]:
        """Generate structured data representation"""
        return {
            "document_info": {
                "type": document.document_type.value,
                "practice_area": document.practice_area.value,
                "jurisdiction": document.jurisdiction,
                "execution_date": document.execution_date.isoformat()
            },
            "parties": [
                {
                    "id": party.party_id,
                    "name": party.name,
                    "entity_type": party.entity_type,
                    "represented_by": party.represented_by
                }
                for party in document.parties
            ],
            "legal_data": document.legal_data,
            "metadata": document.metadata
        }
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information"""
        return {
            "adapter_type": "legal_data",
            "supported_document_types": [doc_type.value for doc_type in LegalDocumentType],
            "supported_practice_areas": [area.value for area in LegalPracticeArea],
            "num_documents": self.config.num_documents,
            "jurisdictions": self.config.jurisdictions,
            "anonymized": self.config.anonymize_data,
            "year_range": (self.config.start_year, self.config.end_year)
        }


# Factory function
def create_legal_adapter(**config_kwargs) -> LegalDataAdapter:
    """Factory function to create legal data adapter"""
    config = LegalDataConfig(**config_kwargs)
    return LegalDataAdapter(config)