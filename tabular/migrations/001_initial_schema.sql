-- Initial schema migration for tabular service
-- Creates tables for unified infrastructure integration

-- Users table (shared across services)
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    tier VARCHAR(50) NOT NULL DEFAULT 'starter',
    organization_id VARCHAR(36),
    permissions JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Organizations table (shared across services)
CREATE TABLE IF NOT EXISTS organizations (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(50) NOT NULL DEFAULT 'starter',
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- API Keys table (shared across services)
CREATE TABLE IF NOT EXISTS api_keys (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    last_used_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Service Usage table (shared across services)
CREATE TABLE IF NOT EXISTS service_usage (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    amount INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Service Jobs table (shared across services)
CREATE TABLE IF NOT EXISTS service_jobs (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Billing Records table (shared across services)
CREATE TABLE IF NOT EXISTS billing_records (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,4) NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    billing_period VARCHAR(20) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Tabular-specific tables

-- Generation Jobs table for tabular service
CREATE TABLE IF NOT EXISTS tabular_generation_jobs (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    dataset_rows INTEGER NOT NULL,
    dataset_columns INTEGER NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress INTEGER NOT NULL DEFAULT 0,
    output_path VARCHAR(500),
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Validation Jobs table for tabular service
CREATE TABLE IF NOT EXISTS tabular_validation_jobs (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    generation_job_id VARCHAR(36),
    validator_type VARCHAR(100) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}',
    results JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (generation_job_id) REFERENCES tabular_generation_jobs(id) ON DELETE SET NULL
);

-- Dataset Profiles table for tabular service
CREATE TABLE IF NOT EXISTS tabular_dataset_profiles (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    dataset_name VARCHAR(255) NOT NULL,
    dataset_hash VARCHAR(64) NOT NULL,
    column_count INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_tier ON users(tier);
CREATE INDEX IF NOT EXISTS idx_users_organization_id ON users(organization_id);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);

CREATE INDEX IF NOT EXISTS idx_service_usage_user_id ON service_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_service_usage_service_name ON service_usage(service_name);
CREATE INDEX IF NOT EXISTS idx_service_usage_resource_type ON service_usage(resource_type);
CREATE INDEX IF NOT EXISTS idx_service_usage_created_at ON service_usage(created_at);

CREATE INDEX IF NOT EXISTS idx_service_jobs_user_id ON service_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_service_jobs_service_name ON service_jobs(service_name);
CREATE INDEX IF NOT EXISTS idx_service_jobs_status ON service_jobs(status);
CREATE INDEX IF NOT EXISTS idx_service_jobs_created_at ON service_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_billing_records_user_id ON billing_records(user_id);
CREATE INDEX IF NOT EXISTS idx_billing_records_service_name ON billing_records(service_name);
CREATE INDEX IF NOT EXISTS idx_billing_records_created_at ON billing_records(created_at);

CREATE INDEX IF NOT EXISTS idx_tabular_generation_jobs_user_id ON tabular_generation_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_tabular_generation_jobs_status ON tabular_generation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_tabular_generation_jobs_algorithm ON tabular_generation_jobs(algorithm);
CREATE INDEX IF NOT EXISTS idx_tabular_generation_jobs_created_at ON tabular_generation_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_tabular_validation_jobs_user_id ON tabular_validation_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_tabular_validation_jobs_generation_job_id ON tabular_validation_jobs(generation_job_id);
CREATE INDEX IF NOT EXISTS idx_tabular_validation_jobs_status ON tabular_validation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_tabular_validation_jobs_created_at ON tabular_validation_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_tabular_dataset_profiles_user_id ON tabular_dataset_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_tabular_dataset_profiles_dataset_hash ON tabular_dataset_profiles(dataset_hash);
CREATE INDEX IF NOT EXISTS idx_tabular_dataset_profiles_created_at ON tabular_dataset_profiles(created_at);

-- Constraints and triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE users IS 'User accounts across all Inferloop services';
COMMENT ON TABLE organizations IS 'Organization accounts for multi-user setups';
COMMENT ON TABLE api_keys IS 'API keys for programmatic access';
COMMENT ON TABLE service_usage IS 'Usage tracking for billing and rate limiting';
COMMENT ON TABLE service_jobs IS 'Generic job tracking across all services';
COMMENT ON TABLE billing_records IS 'Billing records for usage-based pricing';

COMMENT ON TABLE tabular_generation_jobs IS 'Synthetic tabular data generation jobs';
COMMENT ON TABLE tabular_validation_jobs IS 'Validation jobs for generated tabular data';
COMMENT ON TABLE tabular_dataset_profiles IS 'Profiling information for datasets';

COMMENT ON COLUMN tabular_generation_jobs.algorithm IS 'Algorithm used: sdv, ctgan, ydata';
COMMENT ON COLUMN tabular_generation_jobs.config IS 'Algorithm-specific configuration parameters';
COMMENT ON COLUMN tabular_generation_jobs.progress IS 'Job progress percentage (0-100)';
COMMENT ON COLUMN tabular_generation_jobs.output_path IS 'Storage path for generated data';

COMMENT ON COLUMN tabular_validation_jobs.validator_type IS 'Validator used: statistical, ml_efficacy, privacy';
COMMENT ON COLUMN tabular_validation_jobs.metrics IS 'Validation metrics configuration';
COMMENT ON COLUMN tabular_validation_jobs.results IS 'Validation results and scores';

COMMENT ON COLUMN tabular_dataset_profiles.dataset_hash IS 'SHA-256 hash of dataset for caching';
COMMENT ON COLUMN tabular_dataset_profiles.profile_data IS 'Statistical profile of the dataset';