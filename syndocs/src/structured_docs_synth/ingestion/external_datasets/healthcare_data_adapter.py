#!/usr/bin/env python3
"""
Healthcare Data Adapter for medical documents and records
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


class HealthcareDocumentType(Enum):
    """Healthcare document types"""
    MEDICAL_RECORD = "medical_record"
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    INSURANCE_CLAIM = "insurance_claim"
    DISCHARGE_SUMMARY = "discharge_summary"
    CONSENT_FORM = "consent_form"
    REFERRAL = "referral"
    VACCINATION_RECORD = "vaccination_record"
    MEDICAL_CERTIFICATE = "medical_certificate"
    HIPAA_FORM = "hipaa_form"


class MedicalSpecialty(Enum):
    """Medical specialties"""
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    PEDIATRICS = "pediatrics"
    PSYCHIATRY = "psychiatry"
    ORTHOPEDICS = "orthopedics"
    DERMATOLOGY = "dermatology"
    GENERAL_PRACTICE = "general_practice"
    EMERGENCY_MEDICINE = "emergency_medicine"
    RADIOLOGY = "radiology"


@dataclass
class PatientInfo:
    """Patient information (anonymized)"""
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: str
    mrn: str  # Medical Record Number
    insurance_id: str = ""
    emergency_contact: str = ""


@dataclass
class ProviderInfo:
    """Healthcare provider information"""
    provider_id: str
    name: str
    specialty: MedicalSpecialty
    npi: str  # National Provider Identifier
    facility: str
    address: str = ""
    phone: str = ""


@dataclass
class HealthcareDocument:
    """Healthcare document record"""
    document_id: str
    document_type: HealthcareDocumentType
    patient_info: PatientInfo
    provider_info: ProviderInfo
    visit_date: datetime
    medical_data: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthcareDataConfig(BaseModel):
    """Healthcare data adapter configuration"""
    
    # Document generation
    num_documents: int = Field(500, description="Number of documents to generate")
    num_patients: int = Field(100, description="Number of unique patients")
    document_types: List[HealthcareDocumentType] = Field(
        default=[HealthcareDocumentType.MEDICAL_RECORD, HealthcareDocumentType.LAB_REPORT],
        description="Document types to generate"
    )
    
    # Privacy settings (HIPAA compliance)
    anonymize_data: bool = Field(True, description="Anonymize patient data")
    use_fake_names: bool = Field(True, description="Use fake patient names")
    mask_identifiers: bool = Field(True, description="Mask medical identifiers")
    
    # Medical realism
    realistic_conditions: bool = Field(True, description="Use realistic medical conditions")
    realistic_medications: bool = Field(True, description="Use realistic medication names")
    realistic_procedures: bool = Field(True, description="Use realistic medical procedures")
    
    # Content settings
    include_text_content: bool = Field(True, description="Generate text content")
    include_structured_data: bool = Field(True, description="Include structured data")
    
    # Medical settings
    specialties: List[MedicalSpecialty] = Field(
        default=[MedicalSpecialty.GENERAL_PRACTICE, MedicalSpecialty.CARDIOLOGY],
        description="Medical specialties to include"
    )
    
    # Date ranges
    start_year: int = Field(2020, description="Start year for medical records")
    end_year: int = Field(2024, description="End year for medical records")


class HealthcareDataAdapter:
    """
    Healthcare Data Adapter for medical documents and records
    
    Generates realistic healthcare documents including medical records,
    prescriptions, lab reports, and insurance claims with HIPAA compliance.
    """
    
    def __init__(self, config: Optional[HealthcareDataConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or HealthcareDataConfig()
        
        # Medical data templates
        self.patient_first_names = [
            "John", "Jane", "Michael", "Sarah", "Robert", "Mary", "William", "Linda",
            "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas"
        ]
        
        self.patient_last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson"
        ]
        
        self.provider_names = [
            "Dr. Emily Carter", "Dr. James Wilson", "Dr. Sarah Johnson", "Dr. Michael Brown",
            "Dr. Lisa Davis", "Dr. Robert Miller", "Dr. Jennifer Garcia", "Dr. David Lee",
            "Dr. Maria Rodriguez", "Dr. Christopher Taylor", "Dr. Amanda White", "Dr. Kevin Martinez"
        ]
        
        self.facilities = [
            "General Hospital", "Medical Center", "Community Health Center", "Regional Medical Center",
            "University Hospital", "Memorial Hospital", "St. Mary's Hospital", "City Medical Center"
        ]
        
        self.medical_conditions = [
            "Hypertension", "Diabetes Type 2", "Hyperlipidemia", "Asthma", "Depression",
            "Anxiety", "Arthritis", "Migraine", "GERD", "Allergic Rhinitis", "Obesity",
            "Insomnia", "Chronic Pain", "Hypothyroidism", "Anemia"
        ]
        
        self.medications = [
            "Lisinopril", "Metformin", "Atorvastatin", "Albuterol", "Sertraline",
            "Lorazepam", "Ibuprofen", "Sumatriptan", "Omeprazole", "Cetirizine",
            "Levothyroxine", "Gabapentin", "Hydrocodone", "Amoxicillin", "Prednisone"
        ]
        
        self.lab_tests = [
            "Complete Blood Count", "Basic Metabolic Panel", "Lipid Panel", "HbA1c",
            "TSH", "Vitamin D", "PSA", "Urinalysis", "Liver Function Tests", "ESR",
            "CRP", "Troponin", "BNP", "PT/INR", "Blood Culture"
        ]
        
        self.logger.info("Healthcare Data Adapter initialized")
    
    def load_data(self, **kwargs) -> List[HealthcareDocument]:
        """Load healthcare document data"""
        start_time = time.time()
        
        try:
            # Generate patients first
            patients = self._generate_patients()
            
            # Generate providers
            providers = self._generate_providers()
            
            # Generate documents
            documents = self._generate_healthcare_documents(patients, providers)
            
            loading_time = time.time() - start_time
            self.logger.info(f"Generated {len(documents)} healthcare documents in {loading_time:.2f}s")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to generate healthcare data: {e}")
            raise ProcessingError(f"Healthcare data generation error: {e}")
    
    def _generate_patients(self) -> List[PatientInfo]:
        """Generate patient records"""
        patients = []
        
        for i in range(self.config.num_patients):
            patient = PatientInfo(
                patient_id=f"PT{i:06d}",
                first_name=random.choice(self.patient_first_names),
                last_name=random.choice(self.patient_last_names),
                date_of_birth=date(
                    random.randint(1940, 2010),
                    random.randint(1, 12),
                    random.randint(1, 28)
                ),
                gender=random.choice(["Male", "Female", "Other"]),
                mrn=f"MRN{random.randint(100000, 999999)}",
                insurance_id=f"INS{random.randint(10000, 99999)}" if not self.config.mask_identifiers else "INS****",
                emergency_contact=f"{random.choice(self.patient_first_names)} {random.choice(self.patient_last_names)}"
            )
            patients.append(patient)
        
        return patients
    
    def _generate_providers(self) -> List[ProviderInfo]:
        """Generate healthcare provider records"""
        providers = []
        
        for i, specialty in enumerate(self.config.specialties * 3):  # Multiple providers per specialty
            provider = ProviderInfo(
                provider_id=f"PR{i:04d}",
                name=random.choice(self.provider_names),
                specialty=specialty,
                npi=f"{random.randint(1000000000, 9999999999)}",
                facility=random.choice(self.facilities),
                address=f"{random.randint(100, 9999)} Medical Dr, City, State 12345",
                phone=f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
            )
            providers.append(provider)
        
        return providers
    
    def _generate_healthcare_documents(self, patients: List[PatientInfo], 
                                     providers: List[ProviderInfo]) -> List[HealthcareDocument]:
        """Generate healthcare documents"""
        documents = []
        
        for i in range(self.config.num_documents):
            doc_type = random.choice(self.config.document_types)
            patient = random.choice(patients)
            provider = random.choice(providers)
            
            document = self._generate_document(i, doc_type, patient, provider)
            documents.append(document)
        
        return documents
    
    def _generate_document(self, index: int, doc_type: HealthcareDocumentType,
                          patient: PatientInfo, provider: ProviderInfo) -> HealthcareDocument:
        """Generate a single healthcare document"""
        # Generate visit date
        visit_date = datetime(
            random.randint(self.config.start_year, self.config.end_year),
            random.randint(1, 12),
            random.randint(1, 28),
            random.randint(8, 18),  # Medical hours
            random.randint(0, 59)
        )
        
        # Generate medical data based on document type
        medical_data = self._generate_medical_data(doc_type, provider.specialty)
        
        document = HealthcareDocument(
            document_id=f"{doc_type.value}_{index:06d}",
            document_type=doc_type,
            patient_info=patient,
            provider_info=provider,
            visit_date=visit_date,
            medical_data=medical_data,
            metadata={
                "visit_type": random.choice(["office_visit", "follow_up", "consultation", "emergency"]),
                "insurance_verification": random.choice(["verified", "pending", "not_verified"]),
                "document_status": random.choice(["final", "preliminary", "amended"]),
                "confidentiality": "restricted"
            }
        )
        
        # Generate content
        if self.config.include_text_content:
            document.text_content = self._generate_text_content(document)
        
        if self.config.include_structured_data:
            document.structured_data = self._generate_structured_data(document)
        
        return document
    
    def _generate_medical_data(self, doc_type: HealthcareDocumentType, 
                              specialty: MedicalSpecialty) -> Dict[str, Any]:
        """Generate medical data based on document type"""
        if doc_type == HealthcareDocumentType.MEDICAL_RECORD:
            return self._generate_medical_record_data(specialty)
        elif doc_type == HealthcareDocumentType.LAB_REPORT:
            return self._generate_lab_report_data()
        elif doc_type == HealthcareDocumentType.PRESCRIPTION:
            return self._generate_prescription_data()
        elif doc_type == HealthcareDocumentType.INSURANCE_CLAIM:
            return self._generate_insurance_claim_data()
        else:
            return self._generate_generic_medical_data()
    
    def _generate_medical_record_data(self, specialty: MedicalSpecialty) -> Dict[str, Any]:
        """Generate medical record data"""
        return {
            "chief_complaint": random.choice([
                "Chest pain", "Shortness of breath", "Headache", "Abdominal pain",
                "Fatigue", "Dizziness", "Back pain", "Joint pain", "Cough", "Fever"
            ]),
            "diagnosis": random.sample(self.medical_conditions, random.randint(1, 3)),
            "medications": random.sample(self.medications, random.randint(1, 4)),
            "vital_signs": {
                "blood_pressure": f"{random.randint(90, 180)}/{random.randint(60, 120)}",
                "heart_rate": random.randint(60, 100),
                "temperature": round(random.uniform(97.0, 101.0), 1),
                "weight": random.randint(100, 300),
                "height": f"5'{random.randint(2, 11)}\""
            },
            "allergies": random.choice(["NKDA", "Penicillin", "Shellfish", "Latex", "None known"]),
            "treatment_plan": "Continue current medications, follow-up in 3 months"
        }
    
    def _generate_lab_report_data(self) -> Dict[str, Any]:
        """Generate lab report data"""
        return {
            "tests_ordered": random.sample(self.lab_tests, random.randint(1, 4)),
            "results": {
                "glucose": random.randint(70, 200),
                "hemoglobin": round(random.uniform(10.0, 16.0), 1),
                "white_blood_cells": round(random.uniform(3.5, 11.0), 1),
                "cholesterol": random.randint(120, 300),
                "creatinine": round(random.uniform(0.6, 1.5), 2)
            },
            "reference_ranges": {
                "glucose": "70-100 mg/dL",
                "hemoglobin": "12.0-15.5 g/dL",
                "white_blood_cells": "3.5-10.5 K/uL"
            },
            "abnormal_flags": random.choice(["None", "High glucose", "Low hemoglobin"])
        }
    
    def _generate_prescription_data(self) -> Dict[str, Any]:
        """Generate prescription data"""
        medication = random.choice(self.medications)
        return {
            "medication": medication,
            "dosage": f"{random.choice([5, 10, 20, 25, 50, 100])}mg",
            "frequency": random.choice(["Once daily", "Twice daily", "Three times daily", "As needed"]),
            "quantity": random.choice([30, 60, 90]),
            "refills": random.randint(0, 5),
            "dea_number": f"AB{random.randint(1000000, 9999999)}" if not self.config.mask_identifiers else "AB*******",
            "generic_substitution": random.choice(["Permitted", "Not permitted"]),
            "instructions": "Take with food. Do not exceed recommended dose."
        }
    
    def _generate_insurance_claim_data(self) -> Dict[str, Any]:
        """Generate insurance claim data"""
        return {
            "claim_number": f"CLM{random.randint(1000000, 9999999)}",
            "procedure_codes": [f"CPT{random.randint(10000, 99999)}" for _ in range(random.randint(1, 3))],
            "diagnosis_codes": [f"ICD{random.randint(100, 999)}.{random.randint(0, 9)}" for _ in range(random.randint(1, 2))],
            "charges": round(random.uniform(100, 5000), 2),
            "insurance_payment": round(random.uniform(50, 4000), 2),
            "patient_responsibility": round(random.uniform(0, 500), 2),
            "claim_status": random.choice(["Paid", "Pending", "Denied", "Under Review"])
        }
    
    def _generate_generic_medical_data(self) -> Dict[str, Any]:
        """Generate generic medical data"""
        return {
            "notes": "Standard medical documentation",
            "follow_up": random.choice(["1 week", "2 weeks", "1 month", "3 months", "6 months"]),
            "priority": random.choice(["routine", "urgent", "stat"])
        }
    
    def _generate_text_content(self, document: HealthcareDocument) -> str:
        """Generate text content for document"""
        if document.document_type == HealthcareDocumentType.MEDICAL_RECORD:
            return self._generate_medical_record_text(document)
        elif document.document_type == HealthcareDocumentType.LAB_REPORT:
            return self._generate_lab_report_text(document)
        elif document.document_type == HealthcareDocumentType.PRESCRIPTION:
            return self._generate_prescription_text(document)
        else:
            return self._generate_generic_medical_text(document)
    
    def _generate_medical_record_text(self, document: HealthcareDocument) -> str:
        """Generate medical record text"""
        patient = document.patient_info
        provider = document.provider_info
        medical_data = document.medical_data
        
        text = f"""MEDICAL RECORD
{provider.facility.upper()}
{provider.specialty.value.replace('_', ' ').title()} Department

PATIENT INFORMATION:
Name: {patient.first_name} {patient.last_name}
MRN: {patient.mrn}
DOB: {patient.date_of_birth.strftime('%m/%d/%Y')}
Gender: {patient.gender}

PROVIDER: {provider.name}
NPI: {provider.npi}
Date of Service: {document.visit_date.strftime('%m/%d/%Y %H:%M')}

CHIEF COMPLAINT: {medical_data.get('chief_complaint', 'Not specified')}

VITAL SIGNS:
BP: {medical_data['vital_signs']['blood_pressure']} mmHg
HR: {medical_data['vital_signs']['heart_rate']} bpm
Temp: {medical_data['vital_signs']['temperature']}°F
Weight: {medical_data['vital_signs']['weight']} lbs
Height: {medical_data['vital_signs']['height']}

ASSESSMENT AND PLAN:
Diagnosis: {', '.join(medical_data.get('diagnosis', []))}
Medications: {', '.join(medical_data.get('medications', []))}
Allergies: {medical_data.get('allergies', 'None known')}

PLAN: {medical_data.get('treatment_plan', 'Continue current treatment')}

Electronically signed by {provider.name}"""
        
        return text
    
    def _generate_lab_report_text(self, document: HealthcareDocument) -> str:
        """Generate lab report text"""
        patient = document.patient_info
        provider = document.provider_info
        medical_data = document.medical_data
        
        text = f"""LABORATORY REPORT
{provider.facility.upper()}

PATIENT: {patient.first_name} {patient.last_name}
MRN: {patient.mrn}
DOB: {patient.date_of_birth.strftime('%m/%d/%Y')}
Collection Date: {document.visit_date.strftime('%m/%d/%Y %H:%M')}
Ordering Physician: {provider.name}

TESTS ORDERED: {', '.join(medical_data.get('tests_ordered', []))}

RESULTS:
"""
        
        results = medical_data.get('results', {})
        for test, value in results.items():
            text += f"{test.replace('_', ' ').title()}: {value}\n"
        
        text += f"\nABNORMAL FLAGS: {medical_data.get('abnormal_flags', 'None')}\n"
        text += f"\nReport generated: {document.visit_date.strftime('%m/%d/%Y %H:%M')}"
        
        return text
    
    def _generate_prescription_text(self, document: HealthcareDocument) -> str:
        """Generate prescription text"""
        patient = document.patient_info
        provider = document.provider_info
        medical_data = document.medical_data
        
        text = f"""PRESCRIPTION
{provider.facility.upper()}

PATIENT: {patient.first_name} {patient.last_name}
DOB: {patient.date_of_birth.strftime('%m/%d/%Y')}
Address: [Patient Address]

PRESCRIBER: {provider.name}
NPI: {provider.npi}
DEA: {medical_data.get('dea_number', 'Not provided')}

Rx: {medical_data.get('medication', 'Unknown medication')}
Strength: {medical_data.get('dosage', 'Unknown')}
Quantity: {medical_data.get('quantity', 'Unknown')}
Refills: {medical_data.get('refills', 0)}
Instructions: {medical_data.get('frequency', 'As directed')}

Generic Substitution: {medical_data.get('generic_substitution', 'Permitted')}

Date Prescribed: {document.visit_date.strftime('%m/%d/%Y')}
Signature: {provider.name}"""
        
        return text
    
    def _generate_generic_medical_text(self, document: HealthcareDocument) -> str:
        """Generate generic medical document text"""
        patient = document.patient_info
        provider = document.provider_info
        
        text = f"""MEDICAL DOCUMENT
{document.document_type.value.replace('_', ' ').upper()}
{provider.facility.upper()}

Patient: {patient.first_name} {patient.last_name}
MRN: {patient.mrn}
Provider: {provider.name}
Date: {document.visit_date.strftime('%m/%d/%Y')}

Document Status: {document.metadata.get('document_status', 'Final')}
Confidentiality: {document.metadata.get('confidentiality', 'Restricted')}"""
        
        return text
    
    def _generate_structured_data(self, document: HealthcareDocument) -> Dict[str, Any]:
        """Generate structured data representation"""
        return {
            "document_info": {
                "type": document.document_type.value,
                "visit_date": document.visit_date.isoformat(),
                "status": document.metadata.get('document_status')
            },
            "patient": {
                "id": document.patient_info.patient_id,
                "mrn": document.patient_info.mrn,
                "age": (date.today() - document.patient_info.date_of_birth).days // 365,
                "gender": document.patient_info.gender
            },
            "provider": {
                "id": document.provider_info.provider_id,
                "name": document.provider_info.name,
                "specialty": document.provider_info.specialty.value,
                "npi": document.provider_info.npi,
                "facility": document.provider_info.facility
            },
            "medical_data": document.medical_data,
            "metadata": document.metadata
        }
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information"""
        return {
            "adapter_type": "healthcare_data",
            "supported_document_types": [doc_type.value for doc_type in HealthcareDocumentType],
            "supported_specialties": [specialty.value for specialty in MedicalSpecialty],
            "num_documents": self.config.num_documents,
            "num_patients": self.config.num_patients,
            "hipaa_compliant": self.config.anonymize_data,
            "year_range": (self.config.start_year, self.config.end_year)
        }


# Factory function
def create_healthcare_adapter(**config_kwargs) -> HealthcareDataAdapter:
    """Factory function to create healthcare data adapter"""
    config = HealthcareDataConfig(**config_kwargs)
    return HealthcareDataAdapter(config)