"""
Form layout generator for creating structured forms with various field types.
Generates forms for applications, surveys, registrations, and data collection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

from pydantic import BaseModel, Field

from ...core.config import BaseConfig
from ...core.exceptions import GenerationError
from ...core.logging import get_logger
from .layout_engine import LayoutEngine, LayoutConfig, LayoutType, Page, LayoutElement, ElementType

logger = get_logger(__name__)


class FieldType(Enum):
    """Form field types"""
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    TEXTAREA = "textarea"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SELECT = "select"
    MULTISELECT = "multiselect"
    FILE = "file"
    SIGNATURE = "signature"
    RATING = "rating"
    SLIDER = "slider"
    COLOR = "color"
    URL = "url"
    PASSWORD = "password"


class FormType(Enum):
    """Common form types"""
    APPLICATION = "application"
    REGISTRATION = "registration"
    SURVEY = "survey"
    QUESTIONNAIRE = "questionnaire"
    ORDER = "order"
    CONTACT = "contact"
    FEEDBACK = "feedback"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    INSURANCE = "insurance"
    TAX = "tax"
    EMPLOYMENT = "employment"
    EDUCATION = "education"
    GOVERNMENT = "government"


@dataclass
class FormField:
    """Form field definition"""
    name: str
    label: str
    field_type: FieldType
    required: bool = False
    placeholder: Optional[str] = None
    default_value: Optional[Any] = None
    options: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None
    help_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "label": self.label,
            "type": self.field_type.value,
            "required": self.required,
            "placeholder": self.placeholder,
            "default": self.default_value,
            "options": self.options,
            "validation": self.validation,
            "help_text": self.help_text,
            "metadata": self.metadata
        }


@dataclass
class FormSection:
    """Form section grouping related fields"""
    title: str
    description: Optional[str] = None
    fields: List[FormField] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
            "metadata": self.metadata
        }


@dataclass
class FormLayout:
    """Complete form layout"""
    title: str
    form_type: FormType
    sections: List[FormSection]
    instructions: Optional[str] = None
    disclaimer: Optional[str] = None
    submit_label: str = "Submit"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "type": self.form_type.value,
            "sections": [s.to_dict() for s in self.sections],
            "instructions": self.instructions,
            "disclaimer": self.disclaimer,
            "submit_label": self.submit_label,
            "metadata": self.metadata
        }


class FormConfig(BaseConfig):
    """Form generation configuration"""
    form_type: FormType = Field(default=FormType.APPLICATION, description="Type of form to generate")
    include_instructions: bool = Field(default=True, description="Include form instructions")
    include_disclaimers: bool = Field(default=True, description="Include legal disclaimers")
    include_signatures: bool = Field(default=True, description="Include signature fields")
    field_layout: str = Field(default="vertical", description="Field layout style (vertical/horizontal)")
    group_related_fields: bool = Field(default=True, description="Group related fields in sections")
    max_fields_per_section: int = Field(default=8, description="Maximum fields per section")
    randomize_optional_fields: bool = Field(default=True, description="Randomly include optional fields")


class FormGenerator:
    """
    Form layout generator for creating structured forms.
    Generates various types of forms with appropriate fields and layouts.
    """
    
    # Form templates for different types
    FORM_TEMPLATES = {
        FormType.APPLICATION: {
            "title": "Application Form",
            "sections": [
                {
                    "title": "Personal Information",
                    "fields": [
                        ("first_name", "First Name", FieldType.TEXT, True),
                        ("last_name", "Last Name", FieldType.TEXT, True),
                        ("middle_name", "Middle Name", FieldType.TEXT, False),
                        ("date_of_birth", "Date of Birth", FieldType.DATE, True),
                        ("ssn", "Social Security Number", FieldType.TEXT, True),
                        ("gender", "Gender", FieldType.SELECT, False, ["Male", "Female", "Other", "Prefer not to say"]),
                    ]
                },
                {
                    "title": "Contact Information",
                    "fields": [
                        ("email", "Email Address", FieldType.EMAIL, True),
                        ("phone", "Phone Number", FieldType.PHONE, True),
                        ("alt_phone", "Alternative Phone", FieldType.PHONE, False),
                        ("address", "Street Address", FieldType.TEXT, True),
                        ("city", "City", FieldType.TEXT, True),
                        ("state", "State", FieldType.SELECT, True, None),  # States added dynamically
                        ("zip_code", "ZIP Code", FieldType.TEXT, True),
                    ]
                },
                {
                    "title": "Additional Information",
                    "fields": [
                        ("purpose", "Purpose of Application", FieldType.TEXTAREA, True),
                        ("references", "References", FieldType.TEXTAREA, False),
                        ("attachments", "Supporting Documents", FieldType.FILE, False),
                    ]
                }
            ],
            "instructions": "Please complete all required fields marked with an asterisk (*). Ensure all information is accurate and up-to-date.",
            "disclaimer": "I certify that the information provided in this application is true and complete to the best of my knowledge."
        },
        
        FormType.MEDICAL: {
            "title": "Medical Information Form",
            "sections": [
                {
                    "title": "Patient Information",
                    "fields": [
                        ("patient_name", "Full Name", FieldType.TEXT, True),
                        ("dob", "Date of Birth", FieldType.DATE, True),
                        ("mrn", "Medical Record Number", FieldType.TEXT, False),
                        ("insurance_id", "Insurance ID", FieldType.TEXT, True),
                        ("emergency_contact", "Emergency Contact", FieldType.TEXT, True),
                        ("emergency_phone", "Emergency Phone", FieldType.PHONE, True),
                    ]
                },
                {
                    "title": "Medical History",
                    "fields": [
                        ("allergies", "Known Allergies", FieldType.TEXTAREA, True),
                        ("medications", "Current Medications", FieldType.TEXTAREA, True),
                        ("conditions", "Medical Conditions", FieldType.TEXTAREA, True),
                        ("surgeries", "Previous Surgeries", FieldType.TEXTAREA, False),
                        ("family_history", "Family Medical History", FieldType.TEXTAREA, False),
                    ]
                },
                {
                    "title": "Current Visit",
                    "fields": [
                        ("symptoms", "Current Symptoms", FieldType.TEXTAREA, True),
                        ("pain_level", "Pain Level (1-10)", FieldType.SLIDER, False),
                        ("duration", "Duration of Symptoms", FieldType.TEXT, True),
                        ("preferred_pharmacy", "Preferred Pharmacy", FieldType.TEXT, False),
                    ]
                }
            ],
            "instructions": "Please provide complete and accurate medical information. This information is confidential and protected under HIPAA.",
            "disclaimer": "I authorize the release of medical information necessary for treatment and insurance processing."
        },
        
        FormType.FINANCIAL: {
            "title": "Financial Application",
            "sections": [
                {
                    "title": "Applicant Information",
                    "fields": [
                        ("full_name", "Full Legal Name", FieldType.TEXT, True),
                        ("ssn", "Social Security Number", FieldType.TEXT, True),
                        ("dob", "Date of Birth", FieldType.DATE, True),
                        ("driver_license", "Driver's License Number", FieldType.TEXT, True),
                        ("citizenship", "Citizenship Status", FieldType.SELECT, True, ["US Citizen", "Permanent Resident", "Other"]),
                    ]
                },
                {
                    "title": "Employment Information",
                    "fields": [
                        ("employer", "Current Employer", FieldType.TEXT, True),
                        ("job_title", "Job Title", FieldType.TEXT, True),
                        ("employment_length", "Years Employed", FieldType.NUMBER, True),
                        ("annual_income", "Annual Income", FieldType.CURRENCY, True),
                        ("other_income", "Other Income Sources", FieldType.TEXTAREA, False),
                    ]
                },
                {
                    "title": "Financial Information",
                    "fields": [
                        ("account_type", "Account Type", FieldType.SELECT, True, ["Checking", "Savings", "Investment", "Loan"]),
                        ("requested_amount", "Requested Amount", FieldType.CURRENCY, False),
                        ("purpose", "Purpose of Request", FieldType.TEXTAREA, True),
                        ("monthly_expenses", "Monthly Expenses", FieldType.CURRENCY, True),
                        ("assets", "Total Assets", FieldType.CURRENCY, False),
                        ("liabilities", "Total Liabilities", FieldType.CURRENCY, False),
                    ]
                }
            ],
            "instructions": "All financial information must be accurate and verifiable. False information may result in legal action.",
            "disclaimer": "I authorize verification of all information provided and consent to credit checks as necessary."
        },
        
        FormType.SURVEY: {
            "title": "Customer Satisfaction Survey",
            "sections": [
                {
                    "title": "About You",
                    "fields": [
                        ("age_group", "Age Group", FieldType.SELECT, True, ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]),
                        ("location", "Location", FieldType.TEXT, False),
                        ("customer_type", "Customer Type", FieldType.RADIO, True, ["New", "Returning", "Regular"]),
                    ]
                },
                {
                    "title": "Service Experience",
                    "fields": [
                        ("overall_rating", "Overall Satisfaction", FieldType.RATING, True),
                        ("service_quality", "Service Quality", FieldType.RATING, True),
                        ("wait_time", "Wait Time Satisfaction", FieldType.RATING, True),
                        ("staff_friendliness", "Staff Friendliness", FieldType.RATING, True),
                        ("recommendation", "Would you recommend us?", FieldType.RADIO, True, ["Yes", "No", "Maybe"]),
                    ]
                },
                {
                    "title": "Feedback",
                    "fields": [
                        ("liked_most", "What did you like most?", FieldType.TEXTAREA, False),
                        ("improvements", "Areas for improvement", FieldType.TEXTAREA, False),
                        ("additional_comments", "Additional Comments", FieldType.TEXTAREA, False),
                    ]
                }
            ],
            "instructions": "Your feedback helps us improve our services. All responses are anonymous.",
            "disclaimer": None
        }
    }
    
    # US States for dropdowns
    US_STATES = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
        "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
        "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
        "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
        "Wisconsin", "Wyoming"
    ]
    
    def __init__(self, config: Optional[FormConfig] = None):
        """Initialize form generator"""
        self.config = config or FormConfig()
        self.layout_engine = LayoutEngine(LayoutConfig(layout_type=LayoutType.FORM))
        logger.info(f"Initialized FormGenerator for {self.config.form_type.value} forms")
    
    def generate_form(self, form_type: Optional[FormType] = None) -> FormLayout:
        """Generate a complete form layout"""
        form_type = form_type or self.config.form_type
        
        # Get template or create custom
        if form_type in self.FORM_TEMPLATES:
            template = self.FORM_TEMPLATES[form_type]
            form = self._create_form_from_template(form_type, template)
        else:
            form = self._create_custom_form(form_type)
        
        # Add signatures if enabled
        if self.config.include_signatures:
            self._add_signature_section(form)
        
        return form
    
    def _create_form_from_template(self, form_type: FormType, template: Dict[str, Any]) -> FormLayout:
        """Create form from template"""
        sections = []
        
        for section_data in template["sections"]:
            section = FormSection(
                title=section_data["title"],
                description=section_data.get("description")
            )
            
            for field_data in section_data["fields"]:
                name, label, field_type, required = field_data[:4]
                options = field_data[4] if len(field_data) > 4 else None
                
                # Special handling for state dropdown
                if name == "state" and options is None:
                    options = self.US_STATES
                
                # Randomly skip optional fields if configured
                if not required and self.config.randomize_optional_fields and random.random() < 0.3:
                    continue
                
                field = FormField(
                    name=name,
                    label=label,
                    field_type=field_type,
                    required=required,
                    options=options,
                    placeholder=self._generate_placeholder(field_type, label)
                )
                
                section.fields.append(field)
            
            if section.fields:  # Only add non-empty sections
                sections.append(section)
        
        return FormLayout(
            title=template["title"],
            form_type=form_type,
            sections=sections,
            instructions=template.get("instructions") if self.config.include_instructions else None,
            disclaimer=template.get("disclaimer") if self.config.include_disclaimers else None
        )
    
    def _create_custom_form(self, form_type: FormType) -> FormLayout:
        """Create a custom form layout"""
        title = f"{form_type.value.title()} Form"
        sections = []
        
        # Create 2-4 sections
        num_sections = random.randint(2, 4)
        section_names = ["Basic Information", "Details", "Additional Information", "Preferences"]
        
        for i in range(num_sections):
            section = FormSection(
                title=section_names[i] if i < len(section_names) else f"Section {i+1}"
            )
            
            # Add 3-8 fields per section
            num_fields = random.randint(3, min(8, self.config.max_fields_per_section))
            
            for j in range(num_fields):
                field = self._generate_random_field(f"field_{i}_{j}")
                section.fields.append(field)
            
            sections.append(section)
        
        return FormLayout(
            title=title,
            form_type=form_type,
            sections=sections,
            instructions="Please complete all required fields." if self.config.include_instructions else None
        )
    
    def _generate_random_field(self, name: str) -> FormField:
        """Generate a random form field"""
        field_types = [
            (FieldType.TEXT, "Text Field", 0.3),
            (FieldType.EMAIL, "Email Address", 0.1),
            (FieldType.PHONE, "Phone Number", 0.1),
            (FieldType.DATE, "Date", 0.1),
            (FieldType.NUMBER, "Number", 0.1),
            (FieldType.SELECT, "Selection", 0.1),
            (FieldType.TEXTAREA, "Comments", 0.1),
            (FieldType.CHECKBOX, "Checkbox", 0.05),
            (FieldType.RADIO, "Choice", 0.05)
        ]
        
        # Weighted random selection
        field_type, base_label, _ = random.choices(
            field_types,
            weights=[w for _, _, w in field_types]
        )[0]
        
        label = f"{base_label} {name.split('_')[-1]}"
        required = random.random() < 0.6  # 60% chance of being required
        
        options = None
        if field_type in [FieldType.SELECT, FieldType.RADIO]:
            options = [f"Option {i+1}" for i in range(random.randint(2, 5))]
        
        return FormField(
            name=name,
            label=label,
            field_type=field_type,
            required=required,
            options=options,
            placeholder=self._generate_placeholder(field_type, label)
        )
    
    def _generate_placeholder(self, field_type: FieldType, label: str) -> str:
        """Generate appropriate placeholder text"""
        placeholders = {
            FieldType.TEXT: "Enter text...",
            FieldType.EMAIL: "email@example.com",
            FieldType.PHONE: "(555) 123-4567",
            FieldType.DATE: "MM/DD/YYYY",
            FieldType.TIME: "HH:MM",
            FieldType.NUMBER: "0",
            FieldType.CURRENCY: "$0.00",
            FieldType.URL: "https://example.com",
            FieldType.TEXTAREA: f"Enter {label.lower()}..."
        }
        return placeholders.get(field_type, f"Enter {label.lower()}")
    
    def _add_signature_section(self, form: FormLayout) -> None:
        """Add signature section to form"""
        signature_section = FormSection(
            title="Signatures",
            description="By signing below, you acknowledge that all information provided is accurate."
        )
        
        # Add signature fields based on form type
        if form.form_type in [FormType.LEGAL, FormType.FINANCIAL, FormType.MEDICAL]:
            signature_section.fields.extend([
                FormField(
                    name="signature",
                    label="Signature",
                    field_type=FieldType.SIGNATURE,
                    required=True
                ),
                FormField(
                    name="signature_date",
                    label="Date",
                    field_type=FieldType.DATE,
                    required=True
                ),
                FormField(
                    name="printed_name",
                    label="Print Name",
                    field_type=FieldType.TEXT,
                    required=True
                )
            ])
            
            # Add witness signature for legal forms
            if form.form_type == FormType.LEGAL:
                signature_section.fields.extend([
                    FormField(
                        name="witness_signature",
                        label="Witness Signature",
                        field_type=FieldType.SIGNATURE,
                        required=True
                    ),
                    FormField(
                        name="witness_name",
                        label="Witness Name",
                        field_type=FieldType.TEXT,
                        required=True
                    )
                ])
        else:
            # Simple signature for other forms
            signature_section.fields.append(
                FormField(
                    name="signature",
                    label="Signature",
                    field_type=FieldType.SIGNATURE,
                    required=True
                )
            )
        
        form.sections.append(signature_section)
    
    def render_form_to_page(self, form: FormLayout) -> Page:
        """Render form layout to a page using layout engine"""
        page = self.layout_engine.create_page(page_number=1)
        
        # Add form title
        self.layout_engine.add_text_element(
            page,
            form.title,
            ElementType.HEADING,
            height=0.4,
            style={"font_size": 16, "bold": True}
        )
        
        # Add instructions
        if form.instructions:
            self.layout_engine.add_text_element(
                page,
                form.instructions,
                ElementType.TEXT,
                height=0.3,
                style={"font_size": 10, "italic": True}
            )
        
        # Add sections
        for section in form.sections:
            # Section title
            self.layout_engine.add_text_element(
                page,
                section.title,
                ElementType.HEADING,
                height=0.3,
                style={"font_size": 12, "bold": True}
            )
            
            # Section description
            if section.description:
                self.layout_engine.add_text_element(
                    page,
                    section.description,
                    ElementType.TEXT,
                    height=0.2,
                    style={"font_size": 9}
                )
            
            # Fields
            for field in section.fields:
                label = field.label
                if field.required:
                    label += " *"
                
                self.layout_engine.add_form_field(
                    page,
                    label,
                    field.field_type.value,
                    height=0.4 if field.field_type == FieldType.TEXTAREA else 0.3
                )
        
        # Add disclaimer
        if form.disclaimer:
            self.layout_engine.add_text_element(
                page,
                form.disclaimer,
                ElementType.TEXT,
                height=0.3,
                style={"font_size": 9, "italic": True}
            )
        
        # Add signature block if present
        signature_labels = []
        for section in form.sections:
            if section.title == "Signatures":
                for field in section.fields:
                    if field.field_type == FieldType.SIGNATURE:
                        signature_labels.append(field.label)
        
        if signature_labels:
            self.layout_engine.add_signature_block(page, signature_labels)
        
        return page
    
    def generate_form_variations(self, base_form_type: FormType, count: int = 5) -> List[FormLayout]:
        """Generate multiple variations of a form type"""
        variations = []
        
        for i in range(count):
            # Temporarily modify config for variation
            original_randomize = self.config.randomize_optional_fields
            self.config.randomize_optional_fields = True
            
            form = self.generate_form(base_form_type)
            form.metadata["variation_id"] = i + 1
            variations.append(form)
            
            # Restore config
            self.config.randomize_optional_fields = original_randomize
        
        return variations


def create_form_generator(config: Optional[Union[Dict[str, Any], FormConfig]] = None) -> FormGenerator:
    """Factory function to create form generator"""
    if isinstance(config, dict):
        config = FormConfig(**config)
    return FormGenerator(config)


def generate_sample_forms() -> Dict[str, FormLayout]:
    """Generate sample forms for all form types"""
    generator = create_form_generator()
    samples = {}
    
    for form_type in FormType:
        try:
            form = generator.generate_form(form_type)
            samples[form_type.value] = form
            logger.info(f"Generated sample form for {form_type.value}")
        except Exception as e:
            logger.error(f"Failed to generate form for {form_type.value}: {str(e)}")
    
    return samples