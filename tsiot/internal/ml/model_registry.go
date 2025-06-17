package ml

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// ModelRegistry manages machine learning models with versioning, A/B testing, and deployment
type ModelRegistry struct {
	logger        *logrus.Logger
	config        *RegistryConfig
	storage       ModelStorage
	models        map[string]*RegisteredModel
	deployments   map[string]*ModelDeployment
	experiments   map[string]*ABTestExperiment
	metrics       *RegistryMetrics
	mu            sync.RWMutex
	stopCh        chan struct{}
}

// RegistryConfig configures the model registry
type RegistryConfig struct {
	Enabled               bool              `json:"enabled"`
	StorageBackend        string            `json:"storage_backend"` // local, s3, gcs, azure
	StoragePath           string            `json:"storage_path"`
	EnableVersioning      bool              `json:"enable_versioning"`
	EnableABTesting       bool              `json:"enable_ab_testing"`
	MaxVersionsPerModel   int               `json:"max_versions_per_model"`
	ModelRetentionDays    int               `json:"model_retention_days"`
	EnableModelValidation bool              `json:"enable_model_validation"`
	ValidationThreshold   float64           `json:"validation_threshold"`
	AutoDeployment        bool              `json:"auto_deployment"`
	StagingEnvironment    string            `json:"staging_environment"`
	ProductionEnvironment string            `json:"production_environment"`
	MetricsEnabled        bool              `json:"metrics_enabled"`
	SecurityEnabled       bool              `json:"security_enabled"`
	EncryptionKey         string            `json:"encryption_key"`
	AccessControl         AccessControlConfig `json:"access_control"`
}

// AccessControlConfig configures access control for models
type AccessControlConfig struct {
	Enabled     bool              `json:"enabled"`
	Roles       map[string][]Permission `json:"roles"`
	Users       map[string]string `json:"users"` // user -> role
	TokenExpiry time.Duration     `json:"token_expiry"`
}

// Permission defines access permissions
type Permission string

const (
	PermissionRead   Permission = "read"
	PermissionWrite  Permission = "write"
	PermissionDeploy Permission = "deploy"
	PermissionDelete Permission = "delete"
)

// RegisteredModel represents a registered ML model
type RegisteredModel struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Type            ModelType              `json:"type"`
	Framework       ModelFramework         `json:"framework"`
	Versions        []*ModelVersion        `json:"versions"`
	LatestVersion   string                 `json:"latest_version"`
	Tags            []string               `json:"tags"`
	Metadata        map[string]interface{} `json:"metadata"`
	Owner           string                 `json:"owner"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	Status          ModelStatus            `json:"status"`
	ValidationRules []ValidationRule       `json:"validation_rules"`
}

// ModelType defines the type of ML model
type ModelType string

const (
	ModelTypeGenerator    ModelType = "generator"
	ModelTypeValidator    ModelType = "validator"
	ModelTypePreprocessor ModelType = "preprocessor"
	ModelTypeClassifier   ModelType = "classifier"
	ModelTypeRegressor    ModelType = "regressor"
	ModelTypeAnomalyDetector ModelType = "anomaly_detector"
	ModelTypeForecaster   ModelType = "forecaster"
	ModelTypeGAN          ModelType = "gan"
	ModelTypeLSTM         ModelType = "lstm"
	ModelTypeARIMA        ModelType = "arima"
)

// ModelFramework defines the ML framework used
type ModelFramework string

const (
	FrameworkTensorFlow ModelFramework = "tensorflow"
	FrameworkPyTorch    ModelFramework = "pytorch"
	FrameworkSciKitLearn ModelFramework = "scikit-learn"
	FrameworkR          ModelFramework = "r"
	FrameworkONNX       ModelFramework = "onnx"
	FrameworkCustom     ModelFramework = "custom"
)

// ModelStatus defines the status of a model
type ModelStatus string

const (
	StatusDraft      ModelStatus = "draft"
	StatusValidating ModelStatus = "validating"
	StatusReady      ModelStatus = "ready"
	StatusDeprecated ModelStatus = "deprecated"
	StatusArchived   ModelStatus = "archived"
)

// ModelVersion represents a specific version of a model
type ModelVersion struct {
	ID              string                 `json:"id"`
	Version         string                 `json:"version"`
	ModelID         string                 `json:"model_id"`
	Description     string                 `json:"description"`
	ArtifactPath    string                 `json:"artifact_path"`
	Schema          *ModelSchema           `json:"schema"`
	Metrics         *ModelMetrics          `json:"metrics"`
	Signature       string                 `json:"signature"`
	Size            int64                  `json:"size"`
	Checksum        string                 `json:"checksum"`
	Environment     *ModelEnvironment      `json:"environment"`
	Parameters      map[string]interface{} `json:"parameters"`
	TrainingData    *TrainingDataInfo      `json:"training_data"`
	ValidationData  *ValidationDataInfo    `json:"validation_data"`
	CreatedBy       string                 `json:"created_by"`
	CreatedAt       time.Time              `json:"created_at"`
	Status          VersionStatus          `json:"status"`
	IsProduction    bool                   `json:"is_production"`
	IsStaging       bool                   `json:"is_staging"`
}

// VersionStatus defines the status of a model version
type VersionStatus string

const (
	VersionStatusPending   VersionStatus = "pending"
	VersionStatusValidated VersionStatus = "validated"
	VersionStatusFailed    VersionStatus = "failed"
	VersionStatusDeployed  VersionStatus = "deployed"
	VersionStatusRetired   VersionStatus = "retired"
)

// ModelSchema defines the input/output schema of a model
type ModelSchema struct {
	InputSchema  SchemaDefinition `json:"input_schema"`
	OutputSchema SchemaDefinition `json:"output_schema"`
}

// SchemaDefinition defines data schema
type SchemaDefinition struct {
	Type        string                 `json:"type"`
	Shape       []int                  `json:"shape,omitempty"`
	Fields      map[string]FieldType   `json:"fields,omitempty"`
	Required    []string               `json:"required,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// FieldType defines field types in schema
type FieldType struct {
	Type        string      `json:"type"`
	Format      string      `json:"format,omitempty"`
	Constraints interface{} `json:"constraints,omitempty"`
}

// ModelMetrics contains performance metrics for a model
type ModelMetrics struct {
	Accuracy    float64   `json:"accuracy,omitempty"`
	Precision   float64   `json:"precision,omitempty"`
	Recall      float64   `json:"recall,omitempty"`
	F1Score     float64   `json:"f1_score,omitempty"`
	MSE         float64   `json:"mse,omitempty"`
	RMSE        float64   `json:"rmse,omitempty"`
	MAE         float64   `json:"mae,omitempty"`
	R2Score     float64   `json:"r2_score,omitempty"`
	Loss        float64   `json:"loss,omitempty"`
	ValLoss     float64   `json:"val_loss,omitempty"`
	AUC         float64   `json:"auc,omitempty"`
	LogLoss     float64   `json:"log_loss,omitempty"`
	CustomMetrics map[string]float64 `json:"custom_metrics,omitempty"`
	EvaluatedAt time.Time `json:"evaluated_at"`
}

// ModelEnvironment describes the runtime environment
type ModelEnvironment struct {
	PythonVersion   string            `json:"python_version,omitempty"`
	Dependencies    []Dependency      `json:"dependencies"`
	SystemInfo      map[string]string `json:"system_info"`
	GPURequired     bool              `json:"gpu_required"`
	MinMemoryMB     int               `json:"min_memory_mb"`
	MinCPUCores     int               `json:"min_cpu_cores"`
}

// Dependency represents a software dependency
type Dependency struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Source  string `json:"source,omitempty"`
}

// TrainingDataInfo contains information about training data
type TrainingDataInfo struct {
	Location     string                 `json:"location"`
	Size         int64                  `json:"size"`
	RecordCount  int64                  `json:"record_count"`
	Features     []string               `json:"features"`
	Target       string                 `json:"target,omitempty"`
	Checksum     string                 `json:"checksum"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// ValidationDataInfo contains information about validation data
type ValidationDataInfo struct {
	Location    string                 `json:"location"`
	Size        int64                  `json:"size"`
	RecordCount int64                  `json:"record_count"`
	Checksum    string                 `json:"checksum"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ValidationRule defines validation rules for models
type ValidationRule struct {
	Name        string                 `json:"name"`
	Type        ValidationType         `json:"type"`
	Threshold   float64                `json:"threshold"`
	Operator    ComparisonOperator     `json:"operator"`
	Required    bool                   `json:"required"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ValidationType defines types of validation
type ValidationType string

const (
	ValidationAccuracy   ValidationType = "accuracy"
	ValidationPrecision  ValidationType = "precision"
	ValidationRecall     ValidationType = "recall"
	ValidationF1Score    ValidationType = "f1_score"
	ValidationMSE        ValidationType = "mse"
	ValidationLatency    ValidationType = "latency"
	ValidationMemory     ValidationType = "memory"
	ValidationCustom     ValidationType = "custom"
)

// ComparisonOperator defines comparison operators
type ComparisonOperator string

const (
	OperatorGreaterThan    ComparisonOperator = "gt"
	OperatorGreaterOrEqual ComparisonOperator = "gte"
	OperatorLessThan       ComparisonOperator = "lt"
	OperatorLessOrEqual    ComparisonOperator = "lte"
	OperatorEqual          ComparisonOperator = "eq"
	OperatorNotEqual       ComparisonOperator = "ne"
)

// ModelDeployment represents a deployed model
type ModelDeployment struct {
	ID                string                 `json:"id"`
	ModelID           string                 `json:"model_id"`
	VersionID         string                 `json:"version_id"`
	Environment       string                 `json:"environment"`
	Status            DeploymentStatus       `json:"status"`
	Endpoint          string                 `json:"endpoint,omitempty"`
	Resources         *DeploymentResources   `json:"resources"`
	HealthCheck       *HealthCheckConfig     `json:"health_check"`
	Scaling           *ScalingConfig         `json:"scaling"`
	Traffic           *TrafficConfig         `json:"traffic"`
	Monitoring        *MonitoringConfig      `json:"monitoring"`
	CreatedBy         string                 `json:"created_by"`
	CreatedAt         time.Time              `json:"created_at"`
	DeployedAt        *time.Time             `json:"deployed_at,omitempty"`
	RetiredAt         *time.Time             `json:"retired_at,omitempty"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// DeploymentStatus defines deployment status
type DeploymentStatus string

const (
	DeploymentStatusPending   DeploymentStatus = "pending"
	DeploymentStatusDeploying DeploymentStatus = "deploying"
	DeploymentStatusRunning   DeploymentStatus = "running"
	DeploymentStatusStopped   DeploymentStatus = "stopped"
	DeploymentStatusFailed    DeploymentStatus = "failed"
	DeploymentStatusRetired   DeploymentStatus = "retired"
)

// DeploymentResources defines resource requirements for deployment
type DeploymentResources struct {
	CPU       string `json:"cpu"`
	Memory    string `json:"memory"`
	GPU       int    `json:"gpu,omitempty"`
	Storage   string `json:"storage"`
	Replicas  int    `json:"replicas"`
}

// HealthCheckConfig configures health checks
type HealthCheckConfig struct {
	Enabled         bool          `json:"enabled"`
	Path            string        `json:"path"`
	Port            int           `json:"port"`
	Interval        time.Duration `json:"interval"`
	Timeout         time.Duration `json:"timeout"`
	FailureThreshold int          `json:"failure_threshold"`
	SuccessThreshold int          `json:"success_threshold"`
}

// ScalingConfig configures auto-scaling
type ScalingConfig struct {
	Enabled     bool    `json:"enabled"`
	MinReplicas int     `json:"min_replicas"`
	MaxReplicas int     `json:"max_replicas"`
	CPUThreshold float64 `json:"cpu_threshold"`
	MemoryThreshold float64 `json:"memory_threshold"`
	CustomMetrics []CustomMetric `json:"custom_metrics,omitempty"`
}

// CustomMetric defines custom scaling metrics
type CustomMetric struct {
	Name      string  `json:"name"`
	Threshold float64 `json:"threshold"`
	Type      string  `json:"type"`
}

// TrafficConfig configures traffic routing
type TrafficConfig struct {
	Strategy      TrafficStrategy `json:"strategy"`
	Weight        int             `json:"weight,omitempty"`
	CanaryPercent int             `json:"canary_percent,omitempty"`
	StickySession bool            `json:"sticky_session"`
}

// TrafficStrategy defines traffic routing strategies
type TrafficStrategy string

const (
	StrategyBlueGreen TrafficStrategy = "blue_green"
	StrategyCanary    TrafficStrategy = "canary"
	StrategyRolling   TrafficStrategy = "rolling"
	StrategyWeighted  TrafficStrategy = "weighted"
)

// MonitoringConfig configures deployment monitoring
type MonitoringConfig struct {
	Enabled       bool              `json:"enabled"`
	MetricsPort   int               `json:"metrics_port"`
	LogLevel      string            `json:"log_level"`
	Alerts        []AlertRule       `json:"alerts"`
	Dashboards    []string          `json:"dashboards"`
	Tracing       bool              `json:"tracing"`
	CustomMetrics map[string]string `json:"custom_metrics"`
}

// AlertRule defines monitoring alerts
type AlertRule struct {
	Name        string                 `json:"name"`
	Metric      string                 `json:"metric"`
	Threshold   float64                `json:"threshold"`
	Operator    ComparisonOperator     `json:"operator"`
	Duration    time.Duration          `json:"duration"`
	Severity    AlertSeverity          `json:"severity"`
	Actions     []AlertAction          `json:"actions"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// AlertSeverity defines alert severity levels
type AlertSeverity string

const (
	SeverityInfo     AlertSeverity = "info"
	SeverityWarning  AlertSeverity = "warning"
	SeverityError    AlertSeverity = "error"
	SeverityCritical AlertSeverity = "critical"
)

// AlertAction defines actions to take on alerts
type AlertAction struct {
	Type   ActionType             `json:"type"`
	Config map[string]interface{} `json:"config"`
}

// ActionType defines types of alert actions
type ActionType string

const (
	ActionEmail      ActionType = "email"
	ActionSlack      ActionType = "slack"
	ActionWebhook    ActionType = "webhook"
	ActionPagerDuty  ActionType = "pagerduty"
	ActionAutoScale  ActionType = "auto_scale"
	ActionRollback   ActionType = "rollback"
)

// ABTestExperiment represents an A/B test experiment
type ABTestExperiment struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Status          ExperimentStatus       `json:"status"`
	ModelA          *ExperimentModel       `json:"model_a"`
	ModelB          *ExperimentModel       `json:"model_b"`
	TrafficSplit    TrafficSplit           `json:"traffic_split"`
	SuccessMetrics  []SuccessMetric        `json:"success_metrics"`
	Duration        time.Duration          `json:"duration"`
	MinSampleSize   int                    `json:"min_sample_size"`
	Confidence      float64                `json:"confidence"`
	Results         *ExperimentResults     `json:"results,omitempty"`
	CreatedBy       string                 `json:"created_by"`
	StartedAt       *time.Time             `json:"started_at,omitempty"`
	EndedAt         *time.Time             `json:"ended_at,omitempty"`
	CreatedAt       time.Time              `json:"created_at"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// ExperimentStatus defines experiment status
type ExperimentStatus string

const (
	ExperimentStatusDraft     ExperimentStatus = "draft"
	ExperimentStatusRunning   ExperimentStatus = "running"
	ExperimentStatusCompleted ExperimentStatus = "completed"
	ExperimentStatusStopped   ExperimentStatus = "stopped"
	ExperimentStatusFailed    ExperimentStatus = "failed"
)

// ExperimentModel represents a model in an experiment
type ExperimentModel struct {
	ModelID     string                 `json:"model_id"`
	VersionID   string                 `json:"version_id"`
	DeploymentID string                `json:"deployment_id"`
	Name        string                 `json:"name"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// TrafficSplit defines how traffic is split between models
type TrafficSplit struct {
	PercentA int    `json:"percent_a"`
	PercentB int    `json:"percent_b"`
	Strategy string `json:"strategy"`
}

// SuccessMetric defines metrics for experiment success
type SuccessMetric struct {
	Name      string             `json:"name"`
	Type      MetricType         `json:"type"`
	Threshold float64            `json:"threshold"`
	Operator  ComparisonOperator `json:"operator"`
	Weight    float64            `json:"weight"`
}

// MetricType defines types of success metrics
type MetricType string

const (
	MetricAccuracy     MetricType = "accuracy"
	MetricLatency      MetricType = "latency"
	MetricThroughput   MetricType = "throughput"
	MetricErrorRate    MetricType = "error_rate"
	MetricUserSatisfaction MetricType = "user_satisfaction"
	MetricBusinessMetric MetricType = "business_metric"
)

// ExperimentResults contains the results of an A/B test
type ExperimentResults struct {
	ModelAMetrics   map[string]float64 `json:"model_a_metrics"`
	ModelBMetrics   map[string]float64 `json:"model_b_metrics"`
	StatisticalTest *StatisticalTest   `json:"statistical_test"`
	Winner          string             `json:"winner,omitempty"` // "A", "B", or "inconclusive"
	Confidence      float64            `json:"confidence"`
	PValue          float64            `json:"p_value"`
	Effect          float64            `json:"effect"`
	SampleSizeA     int                `json:"sample_size_a"`
	SampleSizeB     int                `json:"sample_size_b"`
	Duration        time.Duration      `json:"duration"`
	Recommendation  string             `json:"recommendation"`
	Summary         string             `json:"summary"`
}

// StatisticalTest represents statistical test results
type StatisticalTest struct {
	TestType    string  `json:"test_type"`
	Statistic   float64 `json:"statistic"`
	PValue      float64 `json:"p_value"`
	CriticalValue float64 `json:"critical_value"`
	IsSignificant bool   `json:"is_significant"`
}

// RegistryMetrics contains metrics for the model registry
type RegistryMetrics struct {
	TotalModels       int                     `json:"total_models"`
	TotalVersions     int                     `json:"total_versions"`
	TotalDeployments  int                     `json:"total_deployments"`
	ActiveDeployments int                     `json:"active_deployments"`
	RunningExperiments int                    `json:"running_experiments"`
	ModelsByType      map[ModelType]int       `json:"models_by_type"`
	ModelsByFramework map[ModelFramework]int  `json:"models_by_framework"`
	ModelsByStatus    map[ModelStatus]int     `json:"models_by_status"`
	DeploymentsByEnv  map[string]int          `json:"deployments_by_env"`
	LastUpdated       time.Time               `json:"last_updated"`
}

// ModelStorage interface for storing model artifacts
type ModelStorage interface {
	Store(ctx context.Context, modelID, versionID string, artifact io.Reader) (string, error)
	Retrieve(ctx context.Context, path string) (io.ReadCloser, error)
	Delete(ctx context.Context, path string) error
	Exists(ctx context.Context, path string) (bool, error)
	ListVersions(ctx context.Context, modelID string) ([]string, error)
	GetMetadata(ctx context.Context, path string) (*StorageMetadata, error)
}

// StorageMetadata contains metadata about stored artifacts
type StorageMetadata struct {
	Size         int64     `json:"size"`
	Checksum     string    `json:"checksum"`
	ContentType  string    `json:"content_type"`
	LastModified time.Time `json:"last_modified"`
	Metadata     map[string]string `json:"metadata"`
}

// NewModelRegistry creates a new model registry
func NewModelRegistry(config *RegistryConfig, logger *logrus.Logger) (*ModelRegistry, error) {
	if config == nil {
		config = getDefaultRegistryConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	registry := &ModelRegistry{
		logger:      logger,
		config:      config,
		models:      make(map[string]*RegisteredModel),
		deployments: make(map[string]*ModelDeployment),
		experiments: make(map[string]*ABTestExperiment),
		metrics:     &RegistryMetrics{},
		stopCh:      make(chan struct{}),
	}

	// Initialize storage backend
	storage, err := NewLocalModelStorage(config.StoragePath, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize storage: %w", err)
	}
	registry.storage = storage

	return registry, nil
}

// Start starts the model registry
func (mr *ModelRegistry) Start(ctx context.Context) error {
	if !mr.config.Enabled {
		mr.logger.Info("Model registry disabled")
		return nil
	}

	mr.logger.Info("Starting model registry")

	// Start metrics collection
	go mr.metricsCollectionLoop(ctx)

	// Start cleanup routine
	go mr.cleanupLoop(ctx)

	return nil
}

// Stop stops the model registry
func (mr *ModelRegistry) Stop(ctx context.Context) error {
	mr.logger.Info("Stopping model registry")
	close(mr.stopCh)
	return nil
}

// RegisterModel registers a new model
func (mr *ModelRegistry) RegisterModel(ctx context.Context, model *RegisteredModel) error {
	if err := mr.validateModel(model); err != nil {
		return fmt.Errorf("invalid model: %w", err)
	}

	mr.mu.Lock()
	defer mr.mu.Unlock()

	// Check if model already exists
	if _, exists := mr.models[model.ID]; exists {
		return fmt.Errorf("model already exists: %s", model.ID)
	}

	model.CreatedAt = time.Now()
	model.UpdatedAt = time.Now()
	model.Status = StatusDraft

	mr.models[model.ID] = model

	mr.logger.WithFields(logrus.Fields{
		"model_id": model.ID,
		"name":     model.Name,
		"type":     model.Type,
	}).Info("Registered new model")

	return nil
}

// CreateModelVersion creates a new version of a model
func (mr *ModelRegistry) CreateModelVersion(ctx context.Context, version *ModelVersion, artifact io.Reader) error {
	if err := mr.validateModelVersion(version); err != nil {
		return fmt.Errorf("invalid model version: %w", err)
	}

	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[version.ModelID]
	if !exists {
		return fmt.Errorf("model not found: %s", version.ModelID)
	}

	// Store model artifact
	artifactPath, err := mr.storage.Store(ctx, version.ModelID, version.ID, artifact)
	if err != nil {
		return fmt.Errorf("failed to store model artifact: %w", err)
	}

	version.ArtifactPath = artifactPath
	version.CreatedAt = time.Now()
	version.Status = VersionStatusPending

	// Calculate checksum
	if artifact != nil {
		version.Checksum = mr.calculateChecksum(artifact)
	}

	// Add version to model
	model.Versions = append(model.Versions, version)
	model.LatestVersion = version.Version
	model.UpdatedAt = time.Now()

	// Validate version if enabled
	if mr.config.EnableModelValidation {
		go mr.validateVersion(ctx, version)
	}

	mr.logger.WithFields(logrus.Fields{
		"model_id":   version.ModelID,
		"version_id": version.ID,
		"version":    version.Version,
	}).Info("Created new model version")

	return nil
}

// DeployModel deploys a model version
func (mr *ModelRegistry) DeployModel(ctx context.Context, deployment *ModelDeployment) error {
	if err := mr.validateDeployment(deployment); err != nil {
		return fmt.Errorf("invalid deployment: %w", err)
	}

	mr.mu.Lock()
	defer mr.mu.Unlock()

	// Verify model and version exist
	model, exists := mr.models[deployment.ModelID]
	if !exists {
		return fmt.Errorf("model not found: %s", deployment.ModelID)
	}

	var version *ModelVersion
	for _, v := range model.Versions {
		if v.ID == deployment.VersionID {
			version = v
			break
		}
	}

	if version == nil {
		return fmt.Errorf("version not found: %s", deployment.VersionID)
	}

	if version.Status != VersionStatusValidated {
		return fmt.Errorf("version not validated: %s", deployment.VersionID)
	}

	deployment.CreatedAt = time.Now()
	deployment.Status = DeploymentStatusPending

	mr.deployments[deployment.ID] = deployment

	// Perform deployment
	go mr.performDeployment(ctx, deployment)

	mr.logger.WithFields(logrus.Fields{
		"deployment_id": deployment.ID,
		"model_id":      deployment.ModelID,
		"version_id":    deployment.VersionID,
		"environment":   deployment.Environment,
	}).Info("Started model deployment")

	return nil
}

// CreateABTest creates a new A/B test experiment
func (mr *ModelRegistry) CreateABTest(ctx context.Context, experiment *ABTestExperiment) error {
	if err := mr.validateExperiment(experiment); err != nil {
		return fmt.Errorf("invalid experiment: %w", err)
	}

	mr.mu.Lock()
	defer mr.mu.Unlock()

	experiment.CreatedAt = time.Now()
	experiment.Status = ExperimentStatusDraft

	mr.experiments[experiment.ID] = experiment

	mr.logger.WithFields(logrus.Fields{
		"experiment_id": experiment.ID,
		"name":          experiment.Name,
		"model_a":       experiment.ModelA.ModelID,
		"model_b":       experiment.ModelB.ModelID,
	}).Info("Created A/B test experiment")

	return nil
}

// Helper methods

func (mr *ModelRegistry) validateModel(model *RegisteredModel) error {
	if model.ID == "" {
		return fmt.Errorf("model ID is required")
	}
	if model.Name == "" {
		return fmt.Errorf("model name is required")
	}
	if model.Type == "" {
		return fmt.Errorf("model type is required")
	}
	return nil
}

func (mr *ModelRegistry) validateModelVersion(version *ModelVersion) error {
	if version.ID == "" {
		return fmt.Errorf("version ID is required")
	}
	if version.ModelID == "" {
		return fmt.Errorf("model ID is required")
	}
	if version.Version == "" {
		return fmt.Errorf("version string is required")
	}
	return nil
}

func (mr *ModelRegistry) validateDeployment(deployment *ModelDeployment) error {
	if deployment.ID == "" {
		return fmt.Errorf("deployment ID is required")
	}
	if deployment.ModelID == "" {
		return fmt.Errorf("model ID is required")
	}
	if deployment.VersionID == "" {
		return fmt.Errorf("version ID is required")
	}
	if deployment.Environment == "" {
		return fmt.Errorf("environment is required")
	}
	return nil
}

func (mr *ModelRegistry) validateExperiment(experiment *ABTestExperiment) error {
	if experiment.ID == "" {
		return fmt.Errorf("experiment ID is required")
	}
	if experiment.ModelA == nil || experiment.ModelB == nil {
		return fmt.Errorf("both models A and B are required")
	}
	if experiment.TrafficSplit.PercentA+experiment.TrafficSplit.PercentB != 100 {
		return fmt.Errorf("traffic split must sum to 100")
	}
	return nil
}

func (mr *ModelRegistry) calculateChecksum(reader io.Reader) string {
	hash := sha256.New()
	io.Copy(hash, reader)
	return hex.EncodeToString(hash.Sum(nil))
}

func (mr *ModelRegistry) validateVersion(ctx context.Context, version *ModelVersion) {
	// Mock validation logic
	time.Sleep(2 * time.Second)
	
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	version.Status = VersionStatusValidated
	
	mr.logger.WithField("version_id", version.ID).Info("Version validated successfully")
}

func (mr *ModelRegistry) performDeployment(ctx context.Context, deployment *ModelDeployment) {
	// Mock deployment logic
	time.Sleep(5 * time.Second)
	
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	deployment.Status = DeploymentStatusRunning
	deployedAt := time.Now()
	deployment.DeployedAt = &deployedAt
	
	mr.logger.WithField("deployment_id", deployment.ID).Info("Deployment completed successfully")
}

func (mr *ModelRegistry) metricsCollectionLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mr.stopCh:
			return
		case <-ticker.C:
			mr.updateMetrics()
		}
	}
}

func (mr *ModelRegistry) updateMetrics() {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	mr.metrics.TotalModels = len(mr.models)
	mr.metrics.TotalDeployments = len(mr.deployments)
	mr.metrics.LastUpdated = time.Now()

	// Count active deployments
	activeDeployments := 0
	for _, deployment := range mr.deployments {
		if deployment.Status == DeploymentStatusRunning {
			activeDeployments++
		}
	}
	mr.metrics.ActiveDeployments = activeDeployments

	// Count running experiments
	runningExperiments := 0
	for _, experiment := range mr.experiments {
		if experiment.Status == ExperimentStatusRunning {
			runningExperiments++
		}
	}
	mr.metrics.RunningExperiments = runningExperiments
}

func (mr *ModelRegistry) cleanupLoop(ctx context.Context) {
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mr.stopCh:
			return
		case <-ticker.C:
			mr.performCleanup(ctx)
		}
	}
}

func (mr *ModelRegistry) performCleanup(ctx context.Context) {
	mr.logger.Info("Starting model registry cleanup")
	
	mr.mu.Lock()
	defer mr.mu.Unlock()

	// Cleanup old model versions based on retention policy
	retentionDate := time.Now().AddDate(0, 0, -mr.config.ModelRetentionDays)
	
	for modelID, model := range mr.models {
		var keepVersions []*ModelVersion
		
		for _, version := range model.Versions {
			if version.CreatedAt.After(retentionDate) || version.IsProduction {
				keepVersions = append(keepVersions, version)
			} else {
				// Delete old version
				if err := mr.storage.Delete(ctx, version.ArtifactPath); err != nil {
					mr.logger.WithError(err).Error("Failed to delete old version artifact")
				}
				mr.logger.WithFields(logrus.Fields{
					"model_id":   modelID,
					"version_id": version.ID,
				}).Info("Cleaned up old model version")
			}
		}
		
		model.Versions = keepVersions
	}
}

func getDefaultRegistryConfig() *RegistryConfig {
	return &RegistryConfig{
		Enabled:               true,
		StorageBackend:        "local",
		StoragePath:           "./models",
		EnableVersioning:      true,
		EnableABTesting:       true,
		MaxVersionsPerModel:   10,
		ModelRetentionDays:    90,
		EnableModelValidation: true,
		ValidationThreshold:   0.8,
		AutoDeployment:        false,
		StagingEnvironment:    "staging",
		ProductionEnvironment: "production",
		MetricsEnabled:        true,
		SecurityEnabled:       false,
		AccessControl: AccessControlConfig{
			Enabled:     false,
			TokenExpiry: 24 * time.Hour,
		},
	}
}