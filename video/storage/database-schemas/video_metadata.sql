-- Video Metadata Database Schema for inferloop-synthdata/video

-- Video Projects Table
CREATE TABLE video_projects (
    project_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    vertical VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    tags JSONB
);

-- Video Generation Jobs Table
CREATE TABLE generation_jobs (
    job_id UUID PRIMARY KEY,
    project_id UUID REFERENCES video_projects(project_id),
    engine VARCHAR(50) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    resolution VARCHAR(20) NOT NULL,
    fps INTEGER NOT NULL,
    scene_type VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL,
    progress FLOAT DEFAULT 0.0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    output_path VARCHAR(255),
    metadata JSONB
);

-- Video Assets Table
CREATE TABLE video_assets (
    asset_id UUID PRIMARY KEY,
    project_id UUID REFERENCES video_projects(project_id),
    job_id UUID REFERENCES generation_jobs(job_id),
    name VARCHAR(255) NOT NULL,
    asset_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    file_size_bytes BIGINT,
    format VARCHAR(20) NOT NULL,
    duration_seconds FLOAT,
    resolution VARCHAR(20),
    fps FLOAT,
    bitrate_kbps INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Video Quality Metrics Table
CREATE TABLE quality_metrics (
    metric_id UUID PRIMARY KEY,
    asset_id UUID REFERENCES video_assets(asset_id),
    job_id UUID REFERENCES generation_jobs(job_id),
    metric_type VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    threshold_value FLOAT,
    passed BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Video Validation Results Table
CREATE TABLE validation_results (
    validation_id UUID PRIMARY KEY,
    asset_id UUID REFERENCES video_assets(asset_id),
    job_id UUID REFERENCES generation_jobs(job_id),
    overall_passed BOOLEAN NOT NULL,
    validation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    validator VARCHAR(100),
    report_path VARCHAR(255),
    metrics JSONB,
    issues JSONB
);

-- Video Delivery Table
CREATE TABLE deliveries (
    delivery_id UUID PRIMARY KEY,
    asset_id UUID REFERENCES video_assets(asset_id),
    destination_type VARCHAR(50) NOT NULL,
    destination_path VARCHAR(255) NOT NULL,
    format VARCHAR(20) NOT NULL,
    file_size_bytes BIGINT,
    delivery_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    metadata JSONB
);

-- Create indexes for performance
CREATE INDEX idx_generation_jobs_project_id ON generation_jobs(project_id);
CREATE INDEX idx_generation_jobs_status ON generation_jobs(status);
CREATE INDEX idx_video_assets_project_id ON video_assets(project_id);
CREATE INDEX idx_video_assets_job_id ON video_assets(job_id);
CREATE INDEX idx_quality_metrics_asset_id ON quality_metrics(asset_id);
CREATE INDEX idx_validation_results_asset_id ON validation_results(asset_id);
CREATE INDEX idx_deliveries_asset_id ON deliveries(asset_id);

-- Create timestamp update trigger function
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply timestamp update trigger to video_projects
CREATE TRIGGER update_video_projects_timestamp
BEFORE UPDATE ON video_projects
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();
