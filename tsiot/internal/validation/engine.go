package validation

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

type ValidationEngineConfig struct {
	DefaultValidators    []string               `json:"default_validators"`
	ConcurrentValidation bool                   `json:"concurrent_validation"`
	MaxConcurrency       int                    `json:"max_concurrency"`
	Timeout              time.Duration          `json:"timeout"`
	RetryPolicy          *RetryPolicy           `json:"retry_policy"`
	Metadata             map[string]interface{} `json:"metadata"`
}

type RetryPolicy struct {
	MaxRetries int           `json:"max_retries"`
	Delay      time.Duration `json:"delay"`
	Backoff    float64       `json:"backoff"`
}

type ValidationEngine struct {
	config     *ValidationEngineConfig
	logger     *logrus.Logger
	validators map[string]interfaces.Validator
	mu         sync.RWMutex
}

type ValidationSummary struct {
	TotalValidators    int                    `json:"total_validators"`
	PassedValidators   int                    `json:"passed_validators"`
	FailedValidators   int                    `json:"failed_validators"`
	OverallScore       float64                `json:"overall_score"`
	ExecutionTime      time.Duration          `json:"execution_time"`
	ValidatorResults   map[string]*models.ValidationResult `json:"validator_results"`
	Errors             map[string]error       `json:"errors"`
	Timestamp          int64                  `json:"timestamp"`
}

func NewValidationEngine(config *ValidationEngineConfig, logger *logrus.Logger) *ValidationEngine {
	if config == nil {
		config = getDefaultValidationEngineConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	engine := &ValidationEngine{
		config:     config,
		logger:     logger,
		validators: make(map[string]interfaces.Validator),
	}

	// Initialize default validators
	engine.initializeDefaultValidators()

	return engine
}

func (e *ValidationEngine) initializeDefaultValidators() {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Register default validators
	for _, validatorType := range e.config.DefaultValidators {
		switch validatorType {
		case constants.ValidatorTypeStatistical:
			e.validators[validatorType] = NewStatisticalValidator(nil, e.logger)
		case constants.ValidatorTypePrivacy:
			e.validators[validatorType] = NewPrivacyValidator(nil, e.logger)
		case constants.ValidatorTypeQuality:
			e.validators[validatorType] = NewQualityValidator(nil, e.logger)
		case constants.ValidatorTypeTemporal:
			e.validators[validatorType] = NewTemporalValidator(nil, e.logger)
		default:
			e.logger.Warnf("Unknown validator type: %s", validatorType)
		}
	}
}

func (e *ValidationEngine) RegisterValidator(validatorType string, validator interfaces.Validator) error {
	if validator == nil {
		return errors.NewValidationError("validator cannot be nil", nil)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.validators[validatorType] = validator
	e.logger.WithField("validator", validatorType).Info("Validator registered")

	return nil
}

func (e *ValidationEngine) UnregisterValidator(validatorType string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.validators[validatorType]; !exists {
		return errors.NewValidationError(fmt.Sprintf("validator %s not found", validatorType), nil)
	}

	delete(e.validators, validatorType)
	e.logger.WithField("validator", validatorType).Info("Validator unregistered")

	return nil
}

func (e *ValidationEngine) GetRegisteredValidators() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	validators := make([]string, 0, len(e.validators))
	for validatorType := range e.validators {
		validators = append(validators, validatorType)
	}

	return validators
}

func (e *ValidationEngine) ValidateDataset(ctx context.Context, dataset []*models.TimeSeries, validators []string) (*ValidationSummary, error) {
	if len(dataset) == 0 {
		return nil, errors.NewValidationError("empty dataset provided", nil)
	}

	// Use default validators if none specified
	if len(validators) == 0 {
		validators = e.config.DefaultValidators
	}

	e.logger.WithFields(logrus.Fields{
		"dataset_size": len(dataset),
		"validators":   validators,
	}).Info("Starting dataset validation")

	start := time.Now()

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, e.config.Timeout)
	defer cancel()

	// Validate using specified validators
	var results map[string]*models.ValidationResult
	var validationErrors map[string]error

	if e.config.ConcurrentValidation {
		results, validationErrors = e.validateConcurrently(ctx, dataset, validators)
	} else {
		results, validationErrors = e.validateSequentially(ctx, dataset, validators)
	}

	// Calculate summary
	summary := e.calculateSummary(results, validationErrors, time.Since(start))

	e.logger.WithFields(logrus.Fields{
		"overall_score":     summary.OverallScore,
		"passed_validators": summary.PassedValidators,
		"failed_validators": summary.FailedValidators,
		"execution_time":    summary.ExecutionTime,
	}).Info("Dataset validation completed")

	return summary, nil
}

func (e *ValidationEngine) ValidateSingleSeries(ctx context.Context, series *models.TimeSeries, validators []string) (*ValidationSummary, error) {
	return e.ValidateDataset(ctx, []*models.TimeSeries{series}, validators)
}

func (e *ValidationEngine) validateConcurrently(ctx context.Context, dataset []*models.TimeSeries, validators []string) (map[string]*models.ValidationResult, map[string]error) {
	results := make(map[string]*models.ValidationResult)
	validationErrors := make(map[string]error)
	
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Create semaphore for concurrency control
	semaphore := make(chan struct{}, e.config.MaxConcurrency)

	for _, validatorType := range validators {
		wg.Add(1)
		go func(vType string) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			result, err := e.runValidator(ctx, vType, dataset)
			
			mu.Lock()
			if err != nil {
				validationErrors[vType] = err
			} else {
				results[vType] = result
			}
			mu.Unlock()
		}(validatorType)
	}

	wg.Wait()
	return results, validationErrors
}

func (e *ValidationEngine) validateSequentially(ctx context.Context, dataset []*models.TimeSeries, validators []string) (map[string]*models.ValidationResult, map[string]error) {
	results := make(map[string]*models.ValidationResult)
	validationErrors := make(map[string]error)

	for _, validatorType := range validators {
		select {
		case <-ctx.Done():
			validationErrors[validatorType] = ctx.Err()
			continue
		default:
		}

		result, err := e.runValidator(ctx, validatorType, dataset)
		if err != nil {
			validationErrors[validatorType] = err
		} else {
			results[validatorType] = result
		}
	}

	return results, validationErrors
}

func (e *ValidationEngine) runValidator(ctx context.Context, validatorType string, dataset []*models.TimeSeries) (*models.ValidationResult, error) {
	e.mu.RLock()
	validator, exists := e.validators[validatorType]
	e.mu.RUnlock()

	if !exists {
		return nil, errors.NewValidationError(fmt.Sprintf("validator %s not registered", validatorType), nil)
	}

	// Execute with retry policy
	var result *models.ValidationResult
	var err error

	for attempt := 0; attempt <= e.config.RetryPolicy.MaxRetries; attempt++ {
		if attempt > 0 {
			// Apply backoff delay
			delay := time.Duration(float64(e.config.RetryPolicy.Delay) * 
				math.Pow(e.config.RetryPolicy.Backoff, float64(attempt-1)))
			
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
			
			e.logger.WithFields(logrus.Fields{
				"validator": validatorType,
				"attempt":   attempt + 1,
			}).Info("Retrying validation")
		}

		result, err = e.executeValidator(ctx, validator, dataset)
		if err == nil {
			break
		}

		if attempt == e.config.RetryPolicy.MaxRetries {
			e.logger.WithFields(logrus.Fields{
				"validator": validatorType,
				"error":     err,
				"attempts":  attempt + 1,
			}).Error("Validation failed after retries")
		}
	}

	return result, err
}

func (e *ValidationEngine) executeValidator(ctx context.Context, validator interfaces.Validator, dataset []*models.TimeSeries) (*models.ValidationResult, error) {
	// For now, validate the first series as a representative
	// In a full implementation, this would be more sophisticated
	if len(dataset) == 0 {
		return nil, errors.NewValidationError("empty dataset", nil)
	}

	// Convert to the expected format for individual validators
	// This is a simplified approach - in practice, you'd want validators that can handle datasets
	return e.validateSingleTimeSeries(ctx, validator, dataset[0])
}

func (e *ValidationEngine) validateSingleTimeSeries(ctx context.Context, validator interfaces.Validator, series *models.TimeSeries) (*models.ValidationResult, error) {
	// Create a validation request
	req := &models.ValidationRequest{
		ID:        fmt.Sprintf("validation_%d", time.Now().UnixNano()),
		Timestamp: time.Now().Unix(),
		Parameters: models.ValidationParameters{
			SyntheticData: series,
			// Note: ReferenceData would typically be provided separately
		},
	}

	// Execute validation based on validator type
	switch v := validator.(type) {
	case interfaces.StatisticalValidator:
		return v.Validate(ctx, req)
	case interfaces.PrivacyValidator:
		return v.Validate(ctx, req)
	case interfaces.QualityValidator:
		return v.Validate(ctx, req)
	case interfaces.TemporalValidator:
		return v.Validate(ctx, req)
	default:
		return nil, errors.NewValidationError("unsupported validator type", nil)
	}
}

func (e *ValidationEngine) calculateSummary(results map[string]*models.ValidationResult, validationErrors map[string]error, executionTime time.Duration) *ValidationSummary {
	summary := &ValidationSummary{
		TotalValidators:  len(results) + len(validationErrors),
		PassedValidators: 0,
		FailedValidators: len(validationErrors),
		ValidatorResults: results,
		Errors:           validationErrors,
		ExecutionTime:    executionTime,
		Timestamp:        time.Now().Unix(),
	}

	// Count passed/failed and calculate overall score
	totalScore := 0.0
	scoreCount := 0

	for _, result := range results {
		if result.Status == "passed" {
			summary.PassedValidators++
		} else {
			summary.FailedValidators++
		}

		totalScore += result.QualityScore
		scoreCount++
	}

	if scoreCount > 0 {
		summary.OverallScore = totalScore / float64(scoreCount)
	}

	return summary
}

func (e *ValidationEngine) GetValidatorInfo(validatorType string) (map[string]interface{}, error) {
	e.mu.RLock()
	validator, exists := e.validators[validatorType]
	e.mu.RUnlock()

	if !exists {
		return nil, errors.NewValidationError(fmt.Sprintf("validator %s not found", validatorType), nil)
	}

	info := map[string]interface{}{
		"type":        validatorType,
		"name":        getValidatorName(validator),
		"description": getValidatorDescription(validator),
	}

	return info, nil
}

func (e *ValidationEngine) GetEngineStatus() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	status := map[string]interface{}{
		"registered_validators": len(e.validators),
		"validator_types":       e.GetRegisteredValidators(),
		"concurrent_validation": e.config.ConcurrentValidation,
		"max_concurrency":       e.config.MaxConcurrency,
		"timeout":               e.config.Timeout.String(),
		"default_validators":    e.config.DefaultValidators,
	}

	return status
}

func (e *ValidationEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Close all validators
	for validatorType, validator := range e.validators {
		if closer, ok := validator.(interfaces.Closeable); ok {
			if err := closer.Close(); err != nil {
				e.logger.WithFields(logrus.Fields{
					"validator": validatorType,
					"error":     err,
				}).Error("Failed to close validator")
			}
		}
	}

	e.validators = nil
	e.logger.Info("Validation engine closed")

	return nil
}

// Helper functions

func getValidatorName(validator interfaces.Validator) string {
	switch v := validator.(type) {
	case interfaces.StatisticalValidator:
		return v.GetName()
	case interfaces.PrivacyValidator:
		return v.GetName()
	case interfaces.QualityValidator:
		return v.GetName()
	case interfaces.TemporalValidator:
		return v.GetName()
	default:
		return "Unknown Validator"
	}
}

func getValidatorDescription(validator interfaces.Validator) string {
	switch v := validator.(type) {
	case interfaces.StatisticalValidator:
		return v.GetDescription()
	case interfaces.PrivacyValidator:
		return v.GetDescription()
	case interfaces.QualityValidator:
		return v.GetDescription()
	case interfaces.TemporalValidator:
		return v.GetDescription()
	default:
		return "No description available"
	}
}

// extractValues extracts float64 values from a time series
func extractValues(ts *models.TimeSeries) []float64 {
	if ts == nil || len(ts.DataPoints) == 0 {
		return []float64{}
	}

	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}

	return values
}

func getDefaultValidationEngineConfig() *ValidationEngineConfig {
	return &ValidationEngineConfig{
		DefaultValidators: []string{
			constants.ValidatorTypeStatistical,
			constants.ValidatorTypeQuality,
			constants.ValidatorTypeTemporal,
		},
		ConcurrentValidation: true,
		MaxConcurrency:       4,
		Timeout:              5 * time.Minute,
		RetryPolicy: &RetryPolicy{
			MaxRetries: 2,
			Delay:      1 * time.Second,
			Backoff:    2.0,
		},
		Metadata: make(map[string]interface{}),
	}
}