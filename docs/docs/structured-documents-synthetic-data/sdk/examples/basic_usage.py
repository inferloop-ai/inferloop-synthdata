"""
Basic usage examples for the Structured Documents Synthetic Data SDK.
Demonstrates common operations and basic document generation workflows.
"""

from sdk.client import StructuredDocsClient
from sdk.models import (
    DocumentType, DocumentFormat, GenerationConfig, 
    LayoutComplexity, ContentQuality, create_generation_request
)


def example_generate_single_document():
    """Example: Generate a single document"""
    print("=== Single Document Generation ===")
    
    # Initialize client
    client = StructuredDocsClient(api_key="your-api-key")
    
    # Generate a simple academic paper
    response = client.generate_document(
        document_type=DocumentType.ACADEMIC_PAPER,
        document_format=DocumentFormat.PDF,
        metadata={
            "title": "Machine Learning in Document Processing",
            "author": "Dr. Jane Smith",
            "subject": "Computer Science"
        }
    )
    
    if response.is_success:
        print(f" Document generated successfully!")
        print(f"   Document ID: {response.document_id}")
        print(f"   File path: {response.file_path}")
        print(f"   File size: {response.file_size} bytes")
        print(f"   Generation time: {response.generation_time_seconds:.2f}s")
        if response.quality_score:
            print(f"   Quality score: {response.quality_score:.3f}")
    else:
        print(f"L Generation failed: {response.message}")
        for error in response.errors:
            print(f"   Error: {error.message}")


def example_generate_with_custom_config():
    """Example: Generate document with custom configuration"""
    print("\n=== Document Generation with Custom Config ===")
    
    client = StructuredDocsClient(api_key="your-api-key")
    
    # Create custom configuration
    config = GenerationConfig.quality_mode()
    config.layout.complexity = LayoutComplexity.COMPLEX
    config.content.quality = ContentQuality.PREMIUM
    config.layout.columns = 2
    config.layout.font_size = 11.0
    config.privacy.anonymize_names = True
    
    # Generate business form with custom config
    response = client.generate_document(
        document_type=DocumentType.BUSINESS_FORM,
        document_format=DocumentFormat.PDF,
        config=config,
        metadata={
            "title": "Job Application Form",
            "author": "HR Department"
        }
    )
    
    if response.is_success:
        print(f" Custom configured document generated!")
        print(f"   Document ID: {response.document_id}")
        print(f"   Quality score: {response.quality_score:.3f}")
    else:
        print(f"L Generation failed: {response.message}")


def example_generate_multiple_documents():
    """Example: Generate multiple documents"""
    print("\n=== Multiple Document Generation ===")
    
    client = StructuredDocsClient(api_key="your-api-key")
    
    # Generate multiple documents of different types
    document_types = [
        (DocumentType.ACADEMIC_PAPER, "Research Paper on AI"),
        (DocumentType.BUSINESS_FORM, "Employee Information Form"),
        (DocumentType.FINANCIAL_REPORT, "Q3 Financial Report"),
        (DocumentType.TECHNICAL_MANUAL, "Software User Guide")
    ]
    
    generated_docs = []
    
    for doc_type, title in document_types:
        print(f"Generating {doc_type.value}...")
        
        response = client.generate_document(
            document_type=doc_type,
            document_format=DocumentFormat.PDF,
            metadata={"title": title}
        )
        
        if response.is_success:
            generated_docs.append(response)
            print(f"    Generated: {response.document_id}")
        else:
            print(f"   L Failed: {response.message}")
    
    print(f"\n=Ê Summary: Generated {len(generated_docs)} out of {len(document_types)} documents")


def example_validate_document():
    """Example: Validate document quality"""
    print("\n=== Document Validation ===")
    
    client = StructuredDocsClient(api_key="your-api-key")
    
    # First generate a document
    gen_response = client.generate_document(
        document_type=DocumentType.LEGAL_DOCUMENT,
        document_format=DocumentFormat.PDF
    )
    
    if gen_response.is_success:
        print(f"Generated document: {gen_response.document_id}")
        
        # Validate the generated document
        validation_response = client.validate_document(
            document_id=gen_response.document_id,
            document_type=DocumentType.LEGAL_DOCUMENT,
            validation_types=["structural", "completeness", "semantic"],
            strict_mode=True
        )
        
        if validation_response.is_success:
            print(f" Validation completed!")
            print(f"   Valid: {validation_response.is_valid}")
            print(f"   Score: {validation_response.validation_score:.3f}")
            print(f"   Issues found: {len(validation_response.issues)}")
            
            if validation_response.issues:
                print("   Issues:")
                for issue in validation_response.issues[:3]:  # Show first 3 issues
                    print(f"     - {issue.get('message', 'Unknown issue')}")
            
            if validation_response.recommendations:
                print("   Recommendations:")
                for rec in validation_response.recommendations[:3]:  # Show first 3
                    print(f"     - {rec}")
        else:
            print(f"L Validation failed: {validation_response.message}")
    else:
        print(f"L Document generation failed: {gen_response.message}")


def example_export_documents():
    """Example: Export documents to different formats"""
    print("\n=== Document Export ===")
    
    client = StructuredDocsClient(api_key="your-api-key")
    
    # Generate a few documents first
    document_ids = []
    
    for i in range(3):
        response = client.generate_document(
            document_type=DocumentType.INVOICE,
            document_format=DocumentFormat.PDF,
            metadata={"title": f"Invoice #{1000 + i}"}
        )
        
        if response.is_success:
            document_ids.append(response.document_id)
            print(f"Generated invoice: {response.document_id}")
    
    if document_ids:
        # Export to COCO format
        print("\nExporting documents to COCO format...")
        export_response = client.export_documents(
            document_ids=document_ids,
            export_format="coco",
            output_path="./exports/invoices_coco",
            include_images=True,
            include_annotations=True,
            compression=True
        )
        
        if export_response.is_success:
            print(f" Export completed!")
            print(f"   Export ID: {export_response.export_id}")
            print(f"   Exported documents: {export_response.exported_documents}")
            print(f"   Output path: {export_response.output_path}")
            print(f"   Export time: {export_response.export_time_seconds:.2f}s")
        else:
            print(f"L Export failed: {export_response.message}")


def example_list_and_manage_documents():
    """Example: List and manage generated documents"""
    print("\n=== Document Management ===")
    
    client = StructuredDocsClient(api_key="your-api-key")
    
    # List all documents
    print("Listing documents...")
    list_response = client.list_documents(
        page=1,
        page_size=10,
        sort_by="created_at",
        sort_order="desc"
    )
    
    if list_response.is_success:
        print(f" Found {list_response.total_count} documents")
        print(f"   Showing page {list_response.page} of {list_response.total_pages}")
        
        for i, doc in enumerate(list_response.items[:3]):  # Show first 3
            print(f"   {i+1}. {doc.document_id} - {doc.document_type}")
        
        # Get details of first document
        if list_response.items:
            doc_id = list_response.items[0].document_id
            print(f"\nGetting details for document: {doc_id}")
            
            doc_response = client.get_document(doc_id)
            if doc_response.is_success:
                print(f"    Document details retrieved")
                print(f"   Type: {doc_response.document_type}")
                print(f"   Format: {doc_response.document_format}")
                print(f"   Size: {doc_response.file_size} bytes")
            else:
                print(f"   L Failed to get document: {doc_response.message}")
    else:
        print(f"L Failed to list documents: {list_response.message}")


def example_check_quota_and_status():
    """Example: Check service status and usage quota"""
    print("\n=== Service Status and Quota ===")
    
    client = StructuredDocsClient(api_key="your-api-key")
    
    # Check service status
    print("Checking service status...")
    status_response = client.get_status()
    
    if status_response.is_success:
        print(f" Service is {status_response.health_status}")
        print(f"   Version: {status_response.version}")
        print(f"   Uptime: {status_response.uptime_seconds:.0f} seconds")
        if status_response.active_jobs is not None:
            print(f"   Active jobs: {status_response.active_jobs}")
    else:
        print(f"L Service status check failed: {status_response.message}")
    
    # Check usage quota
    print("\nChecking usage quota...")
    quota_response = client.get_quota()
    
    if quota_response.is_success:
        print(f" Quota information:")
        print(f"   Plan: {quota_response.plan_type}")
        print(f"   Documents today: {quota_response.documents_generated_today}/{quota_response.daily_document_limit}")
        print(f"   Documents this month: {quota_response.documents_generated_month}/{quota_response.monthly_document_limit}")
        print(f"   Storage used: {quota_response.storage_usage_percentage:.1f}%")
        
        if quota_response.is_quota_exceeded:
            print("      Quota exceeded!")
        else:
            print(f"    Remaining today: {quota_response.daily_documents_remaining} documents")
    else:
        print(f"L Quota check failed: {quota_response.message}")


def example_error_handling():
    """Example: Proper error handling"""
    print("\n=== Error Handling ===")
    
    # Initialize client with invalid API key
    client = StructuredDocsClient(api_key="invalid-key")
    
    # Try to generate a document
    response = client.generate_document(
        document_type=DocumentType.ACADEMIC_PAPER
    )
    
    if response.is_failure:
        print(f"L Expected failure occurred: {response.message}")
        
        # Handle different error types
        for error in response.errors:
            if error.error_type.value == "authentication_error":
                print("   ’ Authentication failed - check your API key")
            elif error.error_type.value == "validation_error":
                print("   ’ Validation failed - check your request parameters")
            elif error.error_type.value == "rate_limit_error":
                print("   ’ Rate limit exceeded - wait before retrying")
            else:
                print(f"   ’ {error.error_type.value}: {error.message}")
    
    # Example of validation errors
    try:
        invalid_request = create_generation_request(
            document_type="invalid_type"  # This will raise an error
        )
    except ValueError as e:
        print(f"L Validation error caught: {str(e)}")


def main():
    """Run all basic usage examples"""
    print("=€ Structured Documents SDK - Basic Usage Examples")
    print("=" * 60)
    
    # Note: These examples assume you have a valid API key
    # Replace "your-api-key" with your actual API key
    
    try:
        example_generate_single_document()
        example_generate_with_custom_config()
        example_generate_multiple_documents()
        example_validate_document()
        example_export_documents()
        example_list_and_manage_documents()
        example_check_quota_and_status()
        example_error_handling()
        
        print("\n All examples completed!")
        
    except Exception as e:
        print(f"\nL An error occurred: {str(e)}")
        print("Note: These examples require a valid API key and running service")


if __name__ == "__main__":
    main()