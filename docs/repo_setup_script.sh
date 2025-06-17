#!/bin/bash

# Simple Repository Setup Script - Directories and Empty Files Only
# Creates the complete structure for inferloop-synthdata with empty placeholder files

set -e

# Configuration
REPO_DIR="${1:-docs}"
PROJECT_NAME="structured-documents-synthetic-data"

echo "Creating repository structure: $REPO_DIR/$PROJECT_NAME"
echo "=============================================="

# Create target directory
mkdir -p "$REPO_DIR/$PROJECT_NAME"
cd "$REPO_DIR/$PROJECT_NAME"

# Function to create empty file
create_empty_file() {
    local filepath="$1"
    mkdir -p "$(dirname "$filepath")"
    touch "$filepath"
    echo "Created: $filepath"
}

# Function to create directory
create_dir() {
    mkdir -p "$1"
    echo "Created dir: $1"
}

echo "Creating root files..."

# Root files
create_empty_file "README.md"
create_empty_file "LICENSE"
create_empty_file "requirements.txt"
create_empty_file "setup.py"
create_empty_file "pyproject.toml"
create_empty_file "Makefile"
create_empty_file ".env.example"
create_empty_file ".gitignore"
create_empty_file "CHANGELOG.md"

echo "Creating GitHub structure..."

# GitHub workflows
create_empty_file ".github/workflows/ci.yml"
create_empty_file ".github/workflows/release.yml"
create_empty_file ".github/workflows/security-scan.yml"
create_empty_file ".github/workflows/ai-security-testing.yml"
create_empty_file ".github/workflows/synthetic-data-validation.yml"
create_empty_file ".github/workflows/llm-safety-testing.yml"
create_empty_file ".github/workflows/agent-security-testing.yml"
create_empty_file ".github/workflows/rag-security-testing.yml"
create_empty_file ".github/workflows/dependency-security.yml"
create_empty_file ".github/ISSUE_TEMPLATE/bug_report.md"
create_empty_file ".github/ISSUE_TEMPLATE/feature_request.md"
create_empty_file ".github/PULL_REQUEST_TEMPLATE.md"
create_empty_file ".github/SECURITY.md"

echo "Creating documentation structure..."

# Documentation
create_empty_file "docs/README.md"
create_empty_file "docs/api/README.md"
create_empty_file "docs/examples/basic_usage.md"
create_empty_file "docs/deployment/README.md"
create_empty_file "docs/compliance/README.md"
create_empty_file "docs/integration/inferloop-integration.md"
create_empty_file "docs/security/README.md"

echo "Creating configuration structure..."

# Schema bank configs
create_empty_file "configs/schema_bank/legal/contract_template.yaml"
create_empty_file "configs/schema_bank/legal/court_filing_template.yaml"
create_empty_file "configs/schema_bank/legal/patent_application_template.yaml"
create_empty_file "configs/schema_bank/healthcare/medical_form_template.yaml"
create_empty_file "configs/schema_bank/healthcare/insurance_claim_template.yaml"
create_empty_file "configs/schema_bank/healthcare/patient_record_template.yaml"
create_empty_file "configs/schema_bank/banking/loan_application_template.yaml"
create_empty_file "configs/schema_bank/banking/financial_statement_template.yaml"
create_empty_file "configs/schema_bank/banking/compliance_report_template.yaml"
create_empty_file "configs/schema_bank/government/tax_form_template.yaml"
create_empty_file "configs/schema_bank/government/permit_application_template.yaml"
create_empty_file "configs/schema_bank/government/regulatory_filing_template.yaml"
create_empty_file "configs/schema_bank/insurance/policy_document_template.yaml"
create_empty_file "configs/schema_bank/insurance/claim_form_template.yaml"
create_empty_file "configs/schema_bank/insurance/actuarial_report_template.yaml"

# Compliance configs
create_empty_file "configs/compliance/gdpr_rules.yaml"
create_empty_file "configs/compliance/hipaa_rules.yaml"
create_empty_file "configs/compliance/pci_dss_rules.yaml"
create_empty_file "configs/compliance/sox_rules.yaml"

# Generation configs
create_empty_file "configs/generation/latex_configs.yaml"
create_empty_file "configs/generation/faker_providers.yaml"
create_empty_file "configs/generation/ocr_noise_profiles.yaml"

# Security configs
create_empty_file "configs/security/ai_security/llm_security_configs/prompt_injection_filters.yaml"
create_empty_file "configs/security/ai_security/llm_security_configs/jailbreak_detection_rules.yaml"
create_empty_file "configs/security/ai_security/llm_security_configs/content_moderation_rules.yaml"
create_empty_file "configs/security/ai_security/llm_security_configs/bias_detection_configs.yaml"
create_empty_file "configs/security/ai_security/llm_security_configs/adversarial_defense_configs.yaml"

create_empty_file "configs/security/ai_security/agent_security_configs/agent_isolation_policies.yaml"
create_empty_file "configs/security/ai_security/agent_security_configs/privilege_escalation_prevention.yaml"
create_empty_file "configs/security/ai_security/agent_security_configs/resource_limitation_policies.yaml"
create_empty_file "configs/security/ai_security/agent_security_configs/tool_access_controls.yaml"
create_empty_file "configs/security/ai_security/agent_security_configs/communication_security_policies.yaml"

create_empty_file "configs/security/ai_security/rag_security_configs/vector_store_security.yaml"
create_empty_file "configs/security/ai_security/rag_security_configs/retrieval_validation_rules.yaml"
create_empty_file "configs/security/ai_security/rag_security_configs/context_sanitization_rules.yaml"
create_empty_file "configs/security/ai_security/rag_security_configs/knowledge_access_controls.yaml"
create_empty_file "configs/security/ai_security/rag_security_configs/deepseek_security_configs.yaml"

create_empty_file "configs/security/ai_security/mcp_security_configs/protocol_validation_rules.yaml"
create_empty_file "configs/security/ai_security/mcp_security_configs/context_isolation_policies.yaml"
create_empty_file "configs/security/ai_security/mcp_security_configs/capability_restrictions.yaml"
create_empty_file "configs/security/ai_security/mcp_security_configs/resource_access_policies.yaml"

create_empty_file "configs/security/ai_security/synthetic_data_security/privacy_preservation_configs.yaml"
create_empty_file "configs/security/ai_security/synthetic_data_security/anonymization_policies.yaml"
create_empty_file "configs/security/ai_security/synthetic_data_security/membership_inference_protection.yaml"
create_empty_file "configs/security/ai_security/synthetic_data_security/reconstruction_attack_prevention.yaml"

# Infrastructure security configs
create_empty_file "configs/security/infrastructure_security/container_security_policies.yaml"
create_empty_file "configs/security/infrastructure_security/kubernetes_security_policies.yaml"
create_empty_file "configs/security/infrastructure_security/network_security_policies.yaml"
create_empty_file "configs/security/infrastructure_security/api_security_configs.yaml"
create_empty_file "configs/security/infrastructure_security/encryption_configs.yaml"

# CI/CD configs
create_empty_file "configs/ci_cd/pipeline_configs/build_pipeline_config.yaml"
create_empty_file "configs/ci_cd/pipeline_configs/test_pipeline_config.yaml"
create_empty_file "configs/ci_cd/pipeline_configs/security_pipeline_config.yaml"
create_empty_file "configs/ci_cd/pipeline_configs/ai_safety_pipeline_config.yaml"
create_empty_file "configs/ci_cd/pipeline_configs/deployment_pipeline_config.yaml"

create_empty_file "configs/ci_cd/quality_gates/security_quality_gates.yaml"
create_empty_file "configs/ci_cd/quality_gates/ai_safety_quality_gates.yaml"
create_empty_file "configs/ci_cd/quality_gates/performance_quality_gates.yaml"
create_empty_file "configs/ci_cd/quality_gates/compliance_quality_gates.yaml"

# Deployment configs
create_empty_file "configs/deployment/docker-compose.yml"
create_empty_file "configs/deployment/kubernetes/namespace.yaml"
create_empty_file "configs/deployment/kubernetes/deployment.yaml"
create_empty_file "configs/deployment/terraform/main.tf"

echo "Creating source code structure..."

# Core modules
create_empty_file "src/structured_docs_synth/__init__.py"
create_empty_file "src/structured_docs_synth/core/__init__.py"
create_empty_file "src/structured_docs_synth/core/config.py"
create_empty_file "src/structured_docs_synth/core/exceptions.py"
create_empty_file "src/structured_docs_synth/core/logging.py"

# Ingestion modules
create_empty_file "src/structured_docs_synth/ingestion/__init__.py"
create_empty_file "src/structured_docs_synth/ingestion/external_datasets/__init__.py"
create_empty_file "src/structured_docs_synth/ingestion/external_datasets/legal_data_adapter.py"
create_empty_file "src/structured_docs_synth/ingestion/external_datasets/healthcare_data_adapter.py"
create_empty_file "src/structured_docs_synth/ingestion/external_datasets/banking_data_adapter.py"
create_empty_file "src/structured_docs_synth/ingestion/external_datasets/government_data_adapter.py"
create_empty_file "src/structured_docs_synth/ingestion/external_datasets/document_datasets_adapter.py"

create_empty_file "src/structured_docs_synth/ingestion/streaming/__init__.py"
create_empty_file "src/structured_docs_synth/ingestion/streaming/kafka_consumer.py"
create_empty_file "src/structured_docs_synth/ingestion/streaming/webhook_handler.py"
create_empty_file "src/structured_docs_synth/ingestion/streaming/api_poller.py"

create_empty_file "src/structured_docs_synth/ingestion/batch/__init__.py"
create_empty_file "src/structured_docs_synth/ingestion/batch/file_processor.py"
create_empty_file "src/structured_docs_synth/ingestion/batch/dataset_loader.py"

# Generation modules
create_empty_file "src/structured_docs_synth/generation/__init__.py"
create_empty_file "src/structured_docs_synth/generation/engines/__init__.py"
create_empty_file "src/structured_docs_synth/generation/engines/latex_generator.py"
create_empty_file "src/structured_docs_synth/generation/engines/docx_generator.py"
create_empty_file "src/structured_docs_synth/generation/engines/pdf_generator.py"
create_empty_file "src/structured_docs_synth/generation/engines/template_engine.py"

create_empty_file "src/structured_docs_synth/generation/content/__init__.py"
create_empty_file "src/structured_docs_synth/generation/content/entity_generator.py"
create_empty_file "src/structured_docs_synth/generation/content/faker_providers.py"
create_empty_file "src/structured_docs_synth/generation/content/domain_data_generator.py"

create_empty_file "src/structured_docs_synth/generation/layout/__init__.py"
create_empty_file "src/structured_docs_synth/generation/layout/layout_engine.py"
create_empty_file "src/structured_docs_synth/generation/layout/table_generator.py"
create_empty_file "src/structured_docs_synth/generation/layout/form_generator.py"

create_empty_file "src/structured_docs_synth/generation/rendering/__init__.py"
create_empty_file "src/structured_docs_synth/generation/rendering/pdf_renderer.py"
create_empty_file "src/structured_docs_synth/generation/rendering/image_renderer.py"
create_empty_file "src/structured_docs_synth/generation/rendering/ocr_noise_injector.py"

# Processing modules
create_empty_file "src/structured_docs_synth/processing/__init__.py"
create_empty_file "src/structured_docs_synth/processing/ocr/__init__.py"
create_empty_file "src/structured_docs_synth/processing/ocr/tesseract_engine.py"
create_empty_file "src/structured_docs_synth/processing/ocr/trocr_engine.py"
create_empty_file "src/structured_docs_synth/processing/ocr/custom_ocr_models.py"
create_empty_file "src/structured_docs_synth/processing/ocr/ocr_pipeline.py"

create_empty_file "src/structured_docs_synth/processing/nlp/__init__.py"
create_empty_file "src/structured_docs_synth/processing/nlp/ner_processor.py"
create_empty_file "src/structured_docs_synth/processing/nlp/layout_tokenizer.py"
create_empty_file "src/structured_docs_synth/processing/nlp/relationship_extractor.py"
create_empty_file "src/structured_docs_synth/processing/nlp/entity_linker.py"

create_empty_file "src/structured_docs_synth/processing/annotation/__init__.py"
create_empty_file "src/structured_docs_synth/processing/annotation/bounding_box_annotator.py"
create_empty_file "src/structured_docs_synth/processing/annotation/structure_annotator.py"
create_empty_file "src/structured_docs_synth/processing/annotation/entity_annotator.py"
create_empty_file "src/structured_docs_synth/processing/annotation/ground_truth_generator.py"

# Privacy modules
create_empty_file "src/structured_docs_synth/privacy/__init__.py"
create_empty_file "src/structured_docs_synth/privacy/differential_privacy/__init__.py"
create_empty_file "src/structured_docs_synth/privacy/differential_privacy/laplace_mechanism.py"
create_empty_file "src/structured_docs_synth/privacy/differential_privacy/exponential_mechanism.py"
create_empty_file "src/structured_docs_synth/privacy/differential_privacy/composition_analyzer.py"
create_empty_file "src/structured_docs_synth/privacy/differential_privacy/privacy_budget_tracker.py"

create_empty_file "src/structured_docs_synth/privacy/pii_protection/__init__.py"
create_empty_file "src/structured_docs_synth/privacy/pii_protection/pii_detector.py"
create_empty_file "src/structured_docs_synth/privacy/pii_protection/masking_strategies.py"
create_empty_file "src/structured_docs_synth/privacy/pii_protection/anonymization_verifier.py"
create_empty_file "src/structured_docs_synth/privacy/pii_protection/tokenizer.py"

create_empty_file "src/structured_docs_synth/privacy/compliance/__init__.py"
create_empty_file "src/structured_docs_synth/privacy/compliance/gdpr_enforcer.py"
create_empty_file "src/structured_docs_synth/privacy/compliance/hipaa_enforcer.py"
create_empty_file "src/structured_docs_synth/privacy/compliance/pci_dss_enforcer.py"
create_empty_file "src/structured_docs_synth/privacy/compliance/sox_enforcer.py"
create_empty_file "src/structured_docs_synth/privacy/compliance/audit_logger.py"

# Quality modules
create_empty_file "src/structured_docs_synth/quality/__init__.py"
create_empty_file "src/structured_docs_synth/quality/metrics/__init__.py"
create_empty_file "src/structured_docs_synth/quality/metrics/ocr_metrics.py"
create_empty_file "src/structured_docs_synth/quality/metrics/layout_metrics.py"
create_empty_file "src/structured_docs_synth/quality/metrics/content_metrics.py"
create_empty_file "src/structured_docs_synth/quality/metrics/teds_calculator.py"
create_empty_file "src/structured_docs_synth/quality/metrics/benchmark_runner.py"

create_empty_file "src/structured_docs_synth/quality/validation/__init__.py"
create_empty_file "src/structured_docs_synth/quality/validation/structural_validator.py"
create_empty_file "src/structured_docs_synth/quality/validation/semantic_validator.py"
create_empty_file "src/structured_docs_synth/quality/validation/completeness_checker.py"
create_empty_file "src/structured_docs_synth/quality/validation/drift_detector.py"

# Delivery modules
create_empty_file "src/structured_docs_synth/delivery/__init__.py"
create_empty_file "src/structured_docs_synth/delivery/api/__init__.py"
create_empty_file "src/structured_docs_synth/delivery/api/rest_api.py"
create_empty_file "src/structured_docs_synth/delivery/api/graphql_api.py"
create_empty_file "src/structured_docs_synth/delivery/api/websocket_api.py"

create_empty_file "src/structured_docs_synth/delivery/storage/__init__.py"
create_empty_file "src/structured_docs_synth/delivery/storage/cloud_storage.py"
create_empty_file "src/structured_docs_synth/delivery/storage/database_storage.py"
create_empty_file "src/structured_docs_synth/delivery/storage/vector_store.py"
create_empty_file "src/structured_docs_synth/delivery/storage/cache_manager.py"

create_empty_file "src/structured_docs_synth/delivery/export/__init__.py"
create_empty_file "src/structured_docs_synth/delivery/export/format_exporters.py"
create_empty_file "src/structured_docs_synth/delivery/export/batch_exporter.py"
create_empty_file "src/structured_docs_synth/delivery/export/streaming_exporter.py"
create_empty_file "src/structured_docs_synth/delivery/export/rag_integrator.py"

# Orchestration modules
create_empty_file "src/structured_docs_synth/orchestration/__init__.py"
create_empty_file "src/structured_docs_synth/orchestration/workflow/__init__.py"
create_empty_file "src/structured_docs_synth/orchestration/workflow/custom_orchestrator.py"

create_empty_file "src/structured_docs_synth/orchestration/scheduling/__init__.py"
create_empty_file "src/structured_docs_synth/orchestration/scheduling/job_scheduler.py"
create_empty_file "src/structured_docs_synth/orchestration/scheduling/cron_manager.py"
create_empty_file "src/structured_docs_synth/orchestration/scheduling/event_scheduler.py"

create_empty_file "src/structured_docs_synth/orchestration/monitoring/__init__.py"
create_empty_file "src/structured_docs_synth/orchestration/monitoring/health_checker.py"
create_empty_file "src/structured_docs_synth/orchestration/monitoring/performance_monitor.py"
create_empty_file "src/structured_docs_synth/orchestration/monitoring/alert_manager.py"
create_empty_file "src/structured_docs_synth/orchestration/monitoring/metrics_collector.py"

# Utils
create_empty_file "src/structured_docs_synth/utils/__init__.py"
create_empty_file "src/structured_docs_synth/utils/file_utils.py"
create_empty_file "src/structured_docs_synth/utils/crypto_utils.py"
create_empty_file "src/structured_docs_synth/utils/format_utils.py"
create_empty_file "src/structured_docs_synth/utils/validation_utils.py"
create_empty_file "src/structured_docs_synth/utils/test_utils.py"

echo "Creating comprehensive test structure..."

# Unit tests
create_empty_file "tests/__init__.py"
create_empty_file "tests/conftest.py"
create_empty_file "tests/unit/__init__.py"
create_empty_file "tests/unit/test_generation/__init__.py"
create_empty_file "tests/unit/test_processing/__init__.py"
create_empty_file "tests/unit/test_privacy/__init__.py"
create_empty_file "tests/unit/test_quality/__init__.py"
create_empty_file "tests/unit/test_delivery/__init__.py"

# Integration tests
create_empty_file "tests/integration/__init__.py"
create_empty_file "tests/integration/test_pipelines/__init__.py"
create_empty_file "tests/integration/test_api/__init__.py"
create_empty_file "tests/integration/test_workflows/__init__.py"

# E2E tests
create_empty_file "tests/e2e/__init__.py"
create_empty_file "tests/e2e/test_full_pipeline/__init__.py"
create_empty_file "tests/e2e/test_compliance/__init__.py"

# Performance tests
create_empty_file "tests/performance/__init__.py"
create_empty_file "tests/performance/load_tests/__init__.py"
create_empty_file "tests/performance/benchmark_tests/__init__.py"

# Security tests
create_empty_file "tests/security/__init__.py"
create_empty_file "tests/security/ai_security/__init__.py"

# LLM Security Tests
create_empty_file "tests/security/ai_security/llm_security_tests/__init__.py"
create_empty_file "tests/security/ai_security/llm_security_tests/prompt_injection_tests.py"
create_empty_file "tests/security/ai_security/llm_security_tests/jailbreak_resistance_tests.py"
create_empty_file "tests/security/ai_security/llm_security_tests/data_leakage_tests.py"
create_empty_file "tests/security/ai_security/llm_security_tests/bias_fairness_tests.py"
create_empty_file "tests/security/ai_security/llm_security_tests/adversarial_input_tests.py"
create_empty_file "tests/security/ai_security/llm_security_tests/model_inversion_tests.py"

# Agent Security Tests
create_empty_file "tests/security/ai_security/agent_security_tests/__init__.py"
create_empty_file "tests/security/ai_security/agent_security_tests/agent_isolation_tests.py"
create_empty_file "tests/security/ai_security/agent_security_tests/privilege_escalation_tests.py"
create_empty_file "tests/security/ai_security/agent_security_tests/agent_communication_security.py"
create_empty_file "tests/security/ai_security/agent_security_tests/resource_abuse_tests.py"
create_empty_file "tests/security/ai_security/agent_security_tests/malicious_tool_usage_tests.py"
create_empty_file "tests/security/ai_security/agent_security_tests/agent_orchestration_security.py"

# RAG Security Tests
create_empty_file "tests/security/ai_security/rag_security_tests/__init__.py"
create_empty_file "tests/security/ai_security/rag_security_tests/vector_store_poisoning_tests.py"
create_empty_file "tests/security/ai_security/rag_security_tests/retrieval_manipulation_tests.py"
create_empty_file "tests/security/ai_security/rag_security_tests/context_injection_tests.py"
create_empty_file "tests/security/ai_security/rag_security_tests/knowledge_extraction_tests.py"
create_empty_file "tests/security/ai_security/rag_security_tests/deepseek_rag_security_tests.py"
create_empty_file "tests/security/ai_security/rag_security_tests/embedding_security_tests.py"

# Synthetic Data Security Tests
create_empty_file "tests/security/ai_security/synthetic_data_security/__init__.py"
create_empty_file "tests/security/ai_security/synthetic_data_security/data_reconstruction_tests.py"
create_empty_file "tests/security/ai_security/synthetic_data_security/membership_inference_tests.py"
create_empty_file "tests/security/ai_security/synthetic_data_security/model_inversion_attacks.py"
create_empty_file "tests/security/ai_security/synthetic_data_security/property_inference_tests.py"
create_empty_file "tests/security/ai_security/synthetic_data_security/differential_privacy_tests.py"
create_empty_file "tests/security/ai_security/synthetic_data_security/anonymization_robustness_tests.py"

# MCP Security Tests
create_empty_file "tests/security/ai_security/mcp_security_tests/__init__.py"
create_empty_file "tests/security/ai_security/mcp_security_tests/protocol_validation_tests.py"
create_empty_file "tests/security/ai_security/mcp_security_tests/context_isolation_tests.py"
create_empty_file "tests/security/ai_security/mcp_security_tests/capability_boundary_tests.py"
create_empty_file "tests/security/ai_security/mcp_security_tests/resource_access_control_tests.py"
create_empty_file "tests/security/ai_security/mcp_security_tests/mcp_communication_security.py"

# Red Team Tests
create_empty_file "tests/security/ai_security/red_team_tests/__init__.py"
create_empty_file "tests/security/ai_security/red_team_tests/adversarial_scenarios/__init__.py"
create_empty_file "tests/security/ai_security/red_team_tests/attack_simulations/__init__.py"
create_empty_file "tests/security/ai_security/red_team_tests/social_engineering_tests/__init__.py"
create_empty_file "tests/security/ai_security/red_team_tests/multi_vector_attacks/__init__.py"

# Infrastructure Security Tests
create_empty_file "tests/security/infrastructure_security/__init__.py"
create_empty_file "tests/security/infrastructure_security/container_security_tests.py"
create_empty_file "tests/security/infrastructure_security/kubernetes_security_tests.py"
create_empty_file "tests/security/infrastructure_security/network_security_tests.py"
create_empty_file "tests/security/infrastructure_security/api_security_tests.py"
create_empty_file "tests/security/infrastructure_security/data_encryption_tests.py"
create_empty_file "tests/security/infrastructure_security/access_control_tests.py"

# Compliance Security Tests
create_empty_file "tests/security/compliance_security/__init__.py"
create_empty_file "tests/security/compliance_security/gdpr_security_tests.py"
create_empty_file "tests/security/compliance_security/hipaa_security_tests.py"
create_empty_file "tests/security/compliance_security/pci_dss_security_tests.py"
create_empty_file "tests/security/compliance_security/sox_security_tests.py"
create_empty_file "tests/security/compliance_security/audit_trail_security_tests.py"

# Penetration Tests
create_empty_file "tests/security/penetration_tests/__init__.py"
create_empty_file "tests/security/penetration_tests/api_penetration_tests.py"
create_empty_file "tests/security/penetration_tests/ai_model_penetration_tests.py"
create_empty_file "tests/security/penetration_tests/data_pipeline_penetration_tests.py"
create_empty_file "tests/security/penetration_tests/system_penetration_tests.py"

# Test fixtures
create_empty_file "tests/fixtures/__init__.py"
create_empty_file "tests/fixtures/sample_documents/.gitkeep"
create_empty_file "tests/fixtures/test_templates/.gitkeep"
create_empty_file "tests/fixtures/mock_data/.gitkeep"
create_empty_file "tests/fixtures/attack_vectors/.gitkeep"
create_empty_file "tests/fixtures/malicious_prompts/.gitkeep"
create_empty_file "tests/fixtures/security_test_data/.gitkeep"

echo "Creating data directory structure..."

# Data directories
create_empty_file "data/external/legal/.gitkeep"
create_empty_file "data/external/healthcare/.gitkeep"
create_empty_file "data/external/banking/.gitkeep"
create_empty_file "data/external/government/.gitkeep"
create_empty_file "data/external/reference_datasets/funsd/.gitkeep"
create_empty_file "data/external/reference_datasets/docbank/.gitkeep"
create_empty_file "data/external/reference_datasets/sroie/.gitkeep"

create_empty_file "data/templates/legal/.gitkeep"
create_empty_file "data/templates/healthcare/.gitkeep"
create_empty_file "data/templates/banking/.gitkeep"
create_empty_file "data/templates/government/.gitkeep"
create_empty_file "data/templates/insurance/.gitkeep"

create_empty_file "data/synthetic/generated/pdf/.gitkeep"
create_empty_file "data/synthetic/generated/docx/.gitkeep"
create_empty_file "data/synthetic/generated/json/.gitkeep"
create_empty_file "data/synthetic/generated/images/.gitkeep"

create_empty_file "data/synthetic/annotated/ocr_results/.gitkeep"
create_empty_file "data/synthetic/annotated/ner_labels/.gitkeep"
create_empty_file "data/synthetic/annotated/layout_tokens/.gitkeep"
create_empty_file "data/synthetic/annotated/ground_truth/.gitkeep"

create_empty_file "data/synthetic/processed/validated/.gitkeep"
create_empty_file "data/synthetic/processed/privacy_filtered/.gitkeep"
create_empty_file "data/synthetic/processed/compliance_approved/.gitkeep"

create_empty_file "data/output/exports/.gitkeep"
create_empty_file "data/output/benchmarks/.gitkeep"
create_empty_file "data/output/reports/.gitkeep"

echo "Creating scripts structure..."

# Setup scripts
create_empty_file "scripts/setup/install_dependencies.sh"
create_empty_file "scripts/setup/setup_database.py"
create_empty_file "scripts/setup/download_models.py"
create_empty_file "scripts/setup/setup_security_tools.sh"

# Deployment scripts
create_empty_file "scripts/deployment/deploy_aws.sh"
create_empty_file "scripts/deployment/deploy_gcp.sh"
create_empty_file "scripts/deployment/deploy_azure.sh"
create_empty_file "scripts/deployment/deploy_kubernetes.sh"

# Data management scripts
create_empty_file "scripts/data_management/download_datasets.py"
create_empty_file "scripts/data_management/setup_templates.py"
create_empty_file "scripts/data_management/migrate_data.py"

# Security scripts
create_empty_file "scripts/security/run_security_scans.sh"
create_empty_file "scripts/security/ai_red_team_testing.py"
create_empty_file "scripts/security/vulnerability_assessment.py"
create_empty_file "scripts/security/compliance_checker.py"
create_empty_file "scripts/security/penetration_testing.sh"
create_empty_file "scripts/security/security_report_generator.py"

# CI/CD scripts
create_empty_file "scripts/ci_cd/build_pipeline.sh"
create_empty_file "scripts/ci_cd/test_pipeline.sh"
create_empty_file "scripts/ci_cd/security_pipeline.sh"
create_empty_file "scripts/ci_cd/ai_safety_pipeline.sh"
create_empty_file "scripts/ci_cd/deployment_pipeline.sh"
create_empty_file "scripts/ci_cd/rollback_pipeline.sh"
create_empty_file "scripts/ci_cd/validate_inferloop_integration.sh"

# Maintenance scripts
create_empty_file "scripts/maintenance/cleanup_storage.py"
create_empty_file "scripts/maintenance/update_models.py"
create_empty_file "scripts/maintenance/security_patch_manager.py"
create_empty_file "scripts/maintenance/backup_data.py"

echo "Creating CLI and SDK structure..."

# CLI
create_empty_file "cli/__init__.py"
create_empty_file "cli/main.py"
create_empty_file "cli/commands/__init__.py"
create_empty_file "cli/commands/generate.py"
create_empty_file "cli/commands/validate.py"
create_empty_file "cli/commands/export.py"
create_empty_file "cli/commands/benchmark.py"
create_empty_file "cli/commands/deploy.py"

create_empty_file "cli/utils/__init__.py"
create_empty_file "cli/utils/progress_tracker.py"
create_empty_file "cli/utils/output_formatter.py"

# SDK
create_empty_file "sdk/__init__.py"
create_empty_file "sdk/client.py"
create_empty_file "sdk/async_client.py"
create_empty_file "sdk/models/__init__.py"
create_empty_file "sdk/models/document_types.py"
create_empty_file "sdk/models/generation_config.py"
create_empty_file "sdk/models/response_models.py"

create_empty_file "sdk/examples/basic_usage.py"
create_empty_file "sdk/examples/advanced_generation.py"
create_empty_file "sdk/examples/batch_processing.py"

echo "Creating deployment structure..."

# Docker
create_empty_file "deployment/docker/Dockerfile"
create_empty_file "deployment/docker/docker-compose.yml"
create_empty_file "deployment/docker/docker-compose.prod.yml"
create_empty_file "deployment/docker/.dockerignore"

# Kubernetes
create_empty_file "deployment/kubernetes/namespace.yaml"
create_empty_file "deployment/kubernetes/deployment.yaml"
create_empty_file "deployment/kubernetes/service.yaml"
create_empty_file "deployment/kubernetes/ingress.yaml"
create_empty_file "deployment/kubernetes/configmap.yaml"
create_empty_file "deployment/kubernetes/secret.yaml"
create_empty_file "deployment/kubernetes/hpa.yaml"

# Terraform
create_empty_file "deployment/terraform/main.tf"
create_empty_file "deployment/terraform/variables.tf"
create_empty_file "deployment/terraform/outputs.tf"
create_empty_file "deployment/terraform/aws/main.tf"
create_empty_file "deployment/terraform/gcp/main.tf"
create_empty_file "deployment/terraform/azure/main.tf"

# Helm
create_empty_file "deployment/helm/Chart.yaml"
create_empty_file "deployment/helm/values.yaml"
create_empty_file "deployment/helm/templates/.gitkeep"

echo "Creating monitoring structure..."

# Prometheus
create_empty_file "monitoring/prometheus/prometheus.yml"
create_empty_file "monitoring/prometheus/rules/ai_model_alerts.yml"
create_empty_file "monitoring/prometheus/rules/security_alerts.yml"
create_empty_file "monitoring/prometheus/rules/agent_monitoring_rules.yml"
create_empty_file "monitoring/prometheus/rules/rag_performance_rules.yml"
create_empty_file "monitoring/prometheus/rules/synthetic_data_quality_rules.yml"

create_empty_file "monitoring/prometheus/exporters/ai_metrics_exporter.py"
create_empty_file "monitoring/prometheus/exporters/security_metrics_exporter.py"
create_empty_file "monitoring/prometheus/exporters/compliance_metrics_exporter.py"

# Grafana
create_empty_file "monitoring/grafana/dashboards/ai_security_dashboard.json"
create_empty_file "monitoring/grafana/dashboards/agent_orchestration_dashboard.json"
create_empty_file "monitoring/grafana/dashboards/rag_security_dashboard.json"
create_empty_file "monitoring/grafana/dashboards/llm_safety_dashboard.json"
create_empty_file "monitoring/grafana/dashboards/synthetic_data_quality_dashboard.json"
create_empty_file "monitoring/grafana/dashboards/compliance_dashboard.json"

create_empty_file "monitoring/grafana/datasources/prometheus.yml"
create_empty_file "monitoring/grafana/datasources/elasticsearch.yml"
create_empty_file "monitoring/grafana/datasources/security_db.yml"

# Alerts
create_empty_file "monitoring/alerts/alertmanager.yml"
create_empty_file "monitoring/alerts/notification_templates/security_incident_template.yml"
create_empty_file "monitoring/alerts/notification_templates/ai_safety_alert_template.yml"
create_empty_file "monitoring/alerts/notification_templates/compliance_violation_template.yml"
create_empty_file "monitoring/alerts/notification_templates/agent_security_alert_template.yml"

create_empty_file "monitoring/alerts/escalation_policies/security_escalation.yml"
create_empty_file "monitoring/alerts/escalation_policies/ai_safety_escalation.yml"
create_empty_file "monitoring/alerts/escalation_policies/compliance_escalation.yml"

# Logging
create_empty_file "monitoring/logging/fluentd/fluentd.conf"
create_empty_file "monitoring/logging/fluentd/ai_logs_parser.conf"
create_empty_file "monitoring/logging/fluentd/security_logs_parser.conf"
create_empty_file "monitoring/logging/fluentd/agent_logs_parser.conf"

create_empty_file "monitoring/logging/elasticsearch/elasticsearch.yml"
create_empty_file "monitoring/logging/elasticsearch/index_templates/.gitkeep"
create_empty_file "monitoring/logging/elasticsearch/search_templates/.gitkeep"

create_empty_file "monitoring/logging/kibana/kibana.yml"
create_empty_file "monitoring/logging/kibana/visualizations/.gitkeep"
create_empty_file "monitoring/logging/kibana/dashboards/.gitkeep"

echo "Creating notebooks structure..."

# Notebooks
create_empty_file "notebooks/01_data_exploration/explore_legal_documents.ipynb"
create_empty_file "notebooks/01_data_exploration/analyze_healthcare_forms.ipynb"
create_empty_file "notebooks/01_data_exploration/examine_banking_docs.ipynb"

create_empty_file "notebooks/02_generation_examples/generate_contracts.ipynb"
create_empty_file "notebooks/02_generation_examples/create_medical_forms.ipynb"
create_empty_file "notebooks/02_generation_examples/build_financial_reports.ipynb"

create_empty_file "notebooks/03_quality_analysis/ocr_quality_analysis.ipynb"
create_empty_file "notebooks/03_quality_analysis/layout_accuracy_assessment.ipynb"
create_empty_file "notebooks/03_quality_analysis/content_validation.ipynb"

create_empty_file "notebooks/04_privacy_compliance/privacy_analysis.ipynb"
create_empty_file "notebooks/04_privacy_compliance/compliance_validation.ipynb"
create_empty_file "notebooks/04_privacy_compliance/anonymization_testing.ipynb"

create_empty_file "notebooks/05_integration_demos/rag_integration_demo.ipynb"
create_empty_file "notebooks/05_integration_demos/llm_training_prep.ipynb"
create_empty_file "notebooks/05_integration_demos/api_usage_examples.ipynb"

echo ""
echo "SUCCESS! Repository structure created"
echo "====================================="
echo ""
echo "Location: $REPO_DIR/$PROJECT_NAME"
echo "Directories created: $(find . -type d | wc -l)"
echo "Empty files created: $(find . -type f | wc -l)"
echo ""
echo "Structure includes:"
echo "- Complete source code modules"
echo "- Comprehensive AI security testing framework"
echo "- CI/CD pipelines with security automation"
echo "- Privacy and compliance frameworks"
echo "- Monitoring and observability infrastructure"
echo "- Deployment configurations"
echo "- Documentation structure"
echo ""
echo "Next steps:"
echo "1. cd $REPO_DIR/$PROJECT_NAME"
echo "2. Start adding content to the empty files"
echo "3. Set up virtual environment and dependencies"
echo ""
echo "All files are empty placeholders ready for development!"