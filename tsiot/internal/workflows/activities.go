package workflows

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// InMemoryActivityRepository implements ActivityRepository using in-memory storage
type InMemoryActivityRepository struct {
	mu         sync.RWMutex
	activities map[string]Activity
}

// NewInMemoryActivityRepository creates a new in-memory activity repository
func NewInMemoryActivityRepository() *InMemoryActivityRepository {
	return &InMemoryActivityRepository{
		activities: make(map[string]Activity),
	}
}

// Register registers an activity
func (repo *InMemoryActivityRepository) Register(activity Activity) error {
	repo.mu.Lock()
	defer repo.mu.Unlock()

	if activity.Name() == "" {
		return fmt.Errorf("activity name cannot be empty")
	}

	repo.activities[activity.Name()] = activity
	return nil
}

// Get retrieves an activity by name
func (repo *InMemoryActivityRepository) Get(name string) (Activity, error) {
	repo.mu.RLock()
	defer repo.mu.RUnlock()

	activity, exists := repo.activities[name]
	if !exists {
		return nil, fmt.Errorf("activity not found: %s", name)
	}

	return activity, nil
}

// List returns all registered activities
func (repo *InMemoryActivityRepository) List() []Activity {
	repo.mu.RLock()
	defer repo.mu.RUnlock()

	activities := make([]Activity, 0, len(repo.activities))
	for _, activity := range repo.activities {
		activities = append(activities, activity)
	}

	return activities
}

// DataIngestionActivity handles data ingestion tasks
type DataIngestionActivity struct {
	logger *logrus.Logger
}

// Name returns the activity name
func (dia *DataIngestionActivity) Name() string {
	return "data_ingestion"
}

// Execute executes the data ingestion activity
func (dia *DataIngestionActivity) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	// Extract parameters
	sourceType, ok := input["source_type"].(string)
	if !ok {
		return nil, fmt.Errorf("source_type is required")
	}

	sourceConfig, ok := input["source_config"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("source_config is required")
	}

	// Simulate data ingestion based on source type
	var recordsIngested int64
	var dataSize int64

	switch sourceType {
	case "file":
		recordsIngested, dataSize = dia.ingestFromFile(ctx, sourceConfig)
	case "database":
		recordsIngested, dataSize = dia.ingestFromDatabase(ctx, sourceConfig)
	case "api":
		recordsIngested, dataSize = dia.ingestFromAPI(ctx, sourceConfig)
	case "stream":
		recordsIngested, dataSize = dia.ingestFromStream(ctx, sourceConfig)
	default:
		return nil, fmt.Errorf("unsupported source type: %s", sourceType)
	}

	output := map[string]interface{}{
		"records_ingested": recordsIngested,
		"data_size":        dataSize,
		"ingestion_time":   time.Now(),
		"status":           "completed",
	}

	return output, nil
}

// Validate validates the activity input
func (dia *DataIngestionActivity) Validate(input map[string]interface{}) error {
	if _, ok := input["source_type"]; !ok {
		return fmt.Errorf("source_type is required")
	}

	if _, ok := input["source_config"]; !ok {
		return fmt.Errorf("source_config is required")
	}

	return nil
}

// GetSchema returns the activity schema
func (dia *DataIngestionActivity) GetSchema() ActivitySchema {
	return ActivitySchema{
		Input: map[string]ParameterDef{
			"source_type": {
				Type:        "string",
				Required:    true,
				Description: "Type of data source (file, database, api, stream)",
			},
			"source_config": {
				Type:        "object",
				Required:    true,
				Description: "Configuration for the data source",
			},
			"batch_size": {
				Type:        "int",
				Required:    false,
				Default:     1000,
				Description: "Batch size for ingestion",
			},
		},
		Output: map[string]OutputDef{
			"records_ingested": {
				Type:        "int",
				Description: "Number of records ingested",
			},
			"data_size": {
				Type:        "int",
				Description: "Size of ingested data in bytes",
			},
			"ingestion_time": {
				Type:        "timestamp",
				Description: "Time when ingestion completed",
			},
		},
	}
}

// Helper methods for DataIngestionActivity
func (dia *DataIngestionActivity) ingestFromFile(ctx context.Context, config map[string]interface{}) (int64, int64) {
	// Simulate file ingestion
	return 5000, 1024 * 1024 // 5K records, 1MB
}

func (dia *DataIngestionActivity) ingestFromDatabase(ctx context.Context, config map[string]interface{}) (int64, int64) {
	// Simulate database ingestion
	return 10000, 2 * 1024 * 1024 // 10K records, 2MB
}

func (dia *DataIngestionActivity) ingestFromAPI(ctx context.Context, config map[string]interface{}) (int64, int64) {
	// Simulate API ingestion
	return 2000, 512 * 1024 // 2K records, 512KB
}

func (dia *DataIngestionActivity) ingestFromStream(ctx context.Context, config map[string]interface{}) (int64, int64) {
	// Simulate stream ingestion
	return 15000, 3 * 1024 * 1024 // 15K records, 3MB
}

// GenerationActivity handles synthetic data generation
type GenerationActivity struct {
	logger *logrus.Logger
}

// Name returns the activity name
func (ga *GenerationActivity) Name() string {
	return "generation"
}

// Execute executes the generation activity
func (ga *GenerationActivity) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	// Extract parameters
	generatorType, ok := input["generator_type"].(string)
	if !ok {
		return nil, fmt.Errorf("generator_type is required")
	}

	length, ok := input["length"].(int)
	if !ok {
		length = 1000 // default
	}

	frequency, ok := input["frequency"].(string)
	if !ok {
		frequency = "1m" // default
	}

	// Simulate generation based on generator type
	var generatedPoints int
	var quality float64

	switch generatorType {
	case "timegan":
		generatedPoints, quality = ga.generateWithTimeGAN(ctx, length)
	case "arima":
		generatedPoints, quality = ga.generateWithARIMA(ctx, length)
	case "statistical":
		generatedPoints, quality = ga.generateWithStatistical(ctx, length)
	case "lstm":
		generatedPoints, quality = ga.generateWithLSTM(ctx, length)
	default:
		return nil, fmt.Errorf("unsupported generator type: %s", generatorType)
	}

	// Create synthetic time series metadata
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("synthetic_%d", time.Now().UnixNano()),
		Name:        fmt.Sprintf("Generated_%s", generatorType),
		Description: fmt.Sprintf("Synthetic data generated using %s", generatorType),
		SensorType:  "synthetic",
		Frequency:   frequency,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	output := map[string]interface{}{
		"time_series_id":    timeSeries.ID,
		"generated_points":  generatedPoints,
		"quality_score":     quality,
		"generator_type":    generatorType,
		"generation_time":   time.Now(),
		"status":           "completed",
	}

	return output, nil
}

// Validate validates the generation activity input
func (ga *GenerationActivity) Validate(input map[string]interface{}) error {
	if _, ok := input["generator_type"]; !ok {
		return fmt.Errorf("generator_type is required")
	}

	if length, ok := input["length"].(int); ok && length <= 0 {
		return fmt.Errorf("length must be positive")
	}

	return nil
}

// GetSchema returns the activity schema
func (ga *GenerationActivity) GetSchema() ActivitySchema {
	return ActivitySchema{
		Input: map[string]ParameterDef{
			"generator_type": {
				Type:        "string",
				Required:    true,
				Description: "Type of generator (timegan, arima, statistical, lstm)",
			},
			"length": {
				Type:        "int",
				Required:    false,
				Default:     1000,
				Description: "Number of points to generate",
			},
			"frequency": {
				Type:        "string",
				Required:    false,
				Default:     "1m",
				Description: "Frequency of generated data points",
			},
			"training_data": {
				Type:        "string",
				Required:    false,
				Description: "ID of training data time series",
			},
		},
		Output: map[string]OutputDef{
			"time_series_id": {
				Type:        "string",
				Description: "ID of generated time series",
			},
			"generated_points": {
				Type:        "int",
				Description: "Number of points generated",
			},
			"quality_score": {
				Type:        "float",
				Description: "Quality score of generated data",
			},
		},
	}
}

// Helper methods for GenerationActivity
func (ga *GenerationActivity) generateWithTimeGAN(ctx context.Context, length int) (int, float64) {
	return length, 0.92 // High quality
}

func (ga *GenerationActivity) generateWithARIMA(ctx context.Context, length int) (int, float64) {
	return length, 0.85 // Good quality
}

func (ga *GenerationActivity) generateWithStatistical(ctx context.Context, length int) (int, float64) {
	return length, 0.78 // Moderate quality
}

func (ga *GenerationActivity) generateWithLSTM(ctx context.Context, length int) (int, float64) {
	return length, 0.88 // Good quality
}

// ValidationActivity handles data validation
type ValidationActivity struct {
	logger *logrus.Logger
}

// Name returns the activity name
func (va *ValidationActivity) Name() string {
	return "validation"
}

// Execute executes the validation activity
func (va *ValidationActivity) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	// Extract parameters
	timeSeriesID, ok := input["time_series_id"].(string)
	if !ok {
		return nil, fmt.Errorf("time_series_id is required")
	}

	validationRules, ok := input["validation_rules"].([]interface{})
	if !ok {
		validationRules = []interface{}{"basic", "statistical"} // default
	}

	// Simulate validation
	testResults := make(map[string]interface{})
	var overallScore float64
	passedTests := 0
	totalTests := len(validationRules)

	for _, rule := range validationRules {
		ruleName := rule.(string)
		result := va.runValidationRule(ctx, timeSeriesID, ruleName)
		testResults[ruleName] = result

		if result.(map[string]interface{})["passed"].(bool) {
			passedTests++
		}
	}

	overallScore = float64(passedTests) / float64(totalTests)

	output := map[string]interface{}{
		"time_series_id":   timeSeriesID,
		"test_results":     testResults,
		"overall_score":    overallScore,
		"tests_passed":     passedTests,
		"total_tests":      totalTests,
		"validation_time":  time.Now(),
		"status":          "completed",
	}

	return output, nil
}

// Validate validates the validation activity input
func (va *ValidationActivity) Validate(input map[string]interface{}) error {
	if _, ok := input["time_series_id"]; !ok {
		return fmt.Errorf("time_series_id is required")
	}

	return nil
}

// GetSchema returns the activity schema
func (va *ValidationActivity) GetSchema() ActivitySchema {
	return ActivitySchema{
		Input: map[string]ParameterDef{
			"time_series_id": {
				Type:        "string",
				Required:    true,
				Description: "ID of time series to validate",
			},
			"validation_rules": {
				Type:        "array",
				Required:    false,
				Description: "List of validation rules to apply",
			},
			"reference_data": {
				Type:        "string",
				Required:    false,
				Description: "ID of reference time series for comparison",
			},
		},
		Output: map[string]OutputDef{
			"test_results": {
				Type:        "object",
				Description: "Detailed test results",
			},
			"overall_score": {
				Type:        "float",
				Description: "Overall validation score (0-1)",
			},
			"tests_passed": {
				Type:        "int",
				Description: "Number of tests passed",
			},
			"total_tests": {
				Type:        "int",
				Description: "Total number of tests run",
			},
		},
	}
}

// Helper method for ValidationActivity
func (va *ValidationActivity) runValidationRule(ctx context.Context, timeSeriesID, ruleName string) interface{} {
	// Simulate validation rule execution
	switch ruleName {
	case "basic":
		return map[string]interface{}{
			"rule":        "basic",
			"passed":      true,
			"score":       0.95,
			"description": "Basic validation checks passed",
		}
	case "statistical":
		return map[string]interface{}{
			"rule":        "statistical",
			"passed":      true,
			"score":       0.88,
			"description": "Statistical validation checks passed",
		}
	case "temporal":
		return map[string]interface{}{
			"rule":        "temporal",
			"passed":      false,
			"score":       0.65,
			"description": "Temporal patterns validation failed",
		}
	default:
		return map[string]interface{}{
			"rule":        ruleName,
			"passed":      true,
			"score":       0.80,
			"description": "Custom validation rule",
		}
	}
}

// ExportActivity handles data export
type ExportActivity struct {
	logger *logrus.Logger
}

// Name returns the activity name
func (ea *ExportActivity) Name() string {
	return "export"
}

// Execute executes the export activity
func (ea *ExportActivity) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	// Extract parameters
	timeSeriesID, ok := input["time_series_id"].(string)
	if !ok {
		return nil, fmt.Errorf("time_series_id is required")
	}

	format, ok := input["format"].(string)
	if !ok {
		format = "csv" // default
	}

	destination, ok := input["destination"].(string)
	if !ok {
		return nil, fmt.Errorf("destination is required")
	}

	// Simulate export
	exportedRecords, fileSize := ea.performExport(ctx, timeSeriesID, format, destination)

	output := map[string]interface{}{
		"time_series_id":    timeSeriesID,
		"export_format":     format,
		"destination":       destination,
		"exported_records":  exportedRecords,
		"file_size":         fileSize,
		"export_time":       time.Now(),
		"status":           "completed",
	}

	return output, nil
}

// Validate validates the export activity input
func (ea *ExportActivity) Validate(input map[string]interface{}) error {
	if _, ok := input["time_series_id"]; !ok {
		return fmt.Errorf("time_series_id is required")
	}

	if _, ok := input["destination"]; !ok {
		return fmt.Errorf("destination is required")
	}

	return nil
}

// GetSchema returns the activity schema
func (ea *ExportActivity) GetSchema() ActivitySchema {
	return ActivitySchema{
		Input: map[string]ParameterDef{
			"time_series_id": {
				Type:        "string",
				Required:    true,
				Description: "ID of time series to export",
			},
			"format": {
				Type:        "string",
				Required:    false,
				Default:     "csv",
				Description: "Export format (csv, json, parquet, etc.)",
			},
			"destination": {
				Type:        "string",
				Required:    true,
				Description: "Export destination path or URL",
			},
			"compression": {
				Type:        "string",
				Required:    false,
				Description: "Compression format (gzip, snappy, etc.)",
			},
		},
		Output: map[string]OutputDef{
			"exported_records": {
				Type:        "int",
				Description: "Number of records exported",
			},
			"file_size": {
				Type:        "int",
				Description: "Size of exported file in bytes",
			},
			"file_path": {
				Type:        "string",
				Description: "Path to exported file",
			},
		},
	}
}

// Helper method for ExportActivity
func (ea *ExportActivity) performExport(ctx context.Context, timeSeriesID, format, destination string) (int64, int64) {
	// Simulate export based on format
	switch format {
	case "csv":
		return 5000, 512 * 1024 // 5K records, 512KB
	case "json":
		return 5000, 1024 * 1024 // 5K records, 1MB
	case "parquet":
		return 5000, 256 * 1024 // 5K records, 256KB (compressed)
	default:
		return 5000, 512 * 1024
	}
}

// TransformationActivity handles data transformation
type TransformationActivity struct {
	logger *logrus.Logger
}

// Name returns the activity name
func (ta *TransformationActivity) Name() string {
	return "transformation"
}

// Execute executes the transformation activity
func (ta *TransformationActivity) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	// Extract parameters
	timeSeriesID, ok := input["time_series_id"].(string)
	if !ok {
		return nil, fmt.Errorf("time_series_id is required")
	}

	transformations, ok := input["transformations"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("transformations are required")
	}

	// Simulate transformation
	transformedRecords := ta.applyTransformations(ctx, timeSeriesID, transformations)

	output := map[string]interface{}{
		"original_series_id":    timeSeriesID,
		"transformed_series_id": fmt.Sprintf("transformed_%d", time.Now().UnixNano()),
		"transformed_records":   transformedRecords,
		"transformations":       transformations,
		"transformation_time":   time.Now(),
		"status":               "completed",
	}

	return output, nil
}

// Validate validates the transformation activity input
func (ta *TransformationActivity) Validate(input map[string]interface{}) error {
	if _, ok := input["time_series_id"]; !ok {
		return fmt.Errorf("time_series_id is required")
	}

	if _, ok := input["transformations"]; !ok {
		return fmt.Errorf("transformations are required")
	}

	return nil
}

// GetSchema returns the activity schema
func (ta *TransformationActivity) GetSchema() ActivitySchema {
	return ActivitySchema{
		Input: map[string]ParameterDef{
			"time_series_id": {
				Type:        "string",
				Required:    true,
				Description: "ID of time series to transform",
			},
			"transformations": {
				Type:        "array",
				Required:    true,
				Description: "List of transformations to apply",
			},
		},
		Output: map[string]OutputDef{
			"transformed_series_id": {
				Type:        "string",
				Description: "ID of transformed time series",
			},
			"transformed_records": {
				Type:        "int",
				Description: "Number of transformed records",
			},
		},
	}
}

// Helper method for TransformationActivity
func (ta *TransformationActivity) applyTransformations(ctx context.Context, timeSeriesID string, transformations []interface{}) int64 {
	// Simulate applying transformations
	return 5000 // Assume 5K records transformed
}