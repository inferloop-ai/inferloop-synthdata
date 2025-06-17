package privacy

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

type KAnonymityConfig struct {
	K                    int                    `json:"k"`
	QuasiIdentifiers     []string               `json:"quasi_identifiers"`
	GeneralizationRules  map[string][]string    `json:"generalization_rules"`
	SuppressionThreshold float64                `json:"suppression_threshold"`
	SensitiveAttributes  []string               `json:"sensitive_attributes"`
	Metadata             map[string]interface{} `json:"metadata"`
}

type KAnonymityProcessor struct {
	config *KAnonymityConfig
	logger *logrus.Logger
	mu     sync.RWMutex
}

type EquivalenceClass struct {
	Records    []*models.TimeSeries
	Identifier string
	Size       int
}

func NewKAnonymityProcessor(config *KAnonymityConfig, logger *logrus.Logger) *KAnonymityProcessor {
	if config == nil {
		config = getDefaultKAnonymityConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	return &KAnonymityProcessor{
		config: config,
		logger: logger,
	}
}

func (k *KAnonymityProcessor) ApplyKAnonymity(ctx context.Context, dataset []*models.TimeSeries) ([]*models.TimeSeries, error) {
	if len(dataset) == 0 {
		return dataset, nil
	}

	if len(dataset) < k.config.K {
		return nil, errors.NewValidationError(
			fmt.Sprintf("dataset size %d is less than k=%d", len(dataset), k.config.K),
			nil,
		)
	}

	k.logger.WithFields(logrus.Fields{
		"dataset_size": len(dataset),
		"k_value":      k.config.K,
	}).Info("Applying k-anonymity")

	// Step 1: Create equivalence classes
	classes := k.createEquivalenceClasses(dataset)

	// Step 2: Apply generalization to achieve k-anonymity
	anonymized := k.generalizeClasses(classes)

	// Step 3: Suppress small groups if necessary
	final := k.suppressSmallGroups(anonymized)

	return final, nil
}

func (k *KAnonymityProcessor) createEquivalenceClasses(dataset []*models.TimeSeries) []*EquivalenceClass {
	classMap := make(map[string]*EquivalenceClass)

	for _, series := range dataset {
		// Generate equivalence class identifier based on quasi-identifiers
		classID := k.getEquivalenceClassID(series)
		
		if class, exists := classMap[classID]; exists {
			class.Records = append(class.Records, series)
			class.Size++
		} else {
			classMap[classID] = &EquivalenceClass{
				Records:    []*models.TimeSeries{series},
				Identifier: classID,
				Size:       1,
			}
		}
	}

	// Convert map to slice
	classes := make([]*EquivalenceClass, 0, len(classMap))
	for _, class := range classMap {
		classes = append(classes, class)
	}

	// Sort by size for better processing
	sort.Slice(classes, func(i, j int) bool {
		return classes[i].Size > classes[j].Size
	})

	return classes
}

func (k *KAnonymityProcessor) getEquivalenceClassID(series *models.TimeSeries) string {
	values := make([]string, 0, len(k.config.QuasiIdentifiers))
	
	for _, qi := range k.config.QuasiIdentifiers {
		value := k.getQuasiIdentifierValue(series, qi)
		values = append(values, value)
	}
	
	// Create a unique identifier for the equivalence class
	return fmt.Sprintf("%v", values)
}

func (k *KAnonymityProcessor) getQuasiIdentifierValue(series *models.TimeSeries, identifier string) string {
	// Extract value from metadata or properties
	if val, ok := series.Metadata[identifier]; ok {
		return fmt.Sprintf("%v", val)
	}
	
	if val, ok := series.Properties[identifier]; ok {
		return fmt.Sprintf("%v", val)
	}
	
	// Check tags
	for _, tag := range series.Tags {
		if tag == identifier {
			return identifier
		}
	}
	
	return ""
}

func (k *KAnonymityProcessor) generalizeClasses(classes []*EquivalenceClass) []*models.TimeSeries {
	var result []*models.TimeSeries
	
	for _, class := range classes {
		if class.Size >= k.config.K {
			// Class already satisfies k-anonymity
			result = append(result, class.Records...)
		} else {
			// Need to generalize or merge with other classes
			generalized := k.generalizeSmallClass(class, classes)
			result = append(result, generalized...)
		}
	}
	
	return result
}

func (k *KAnonymityProcessor) generalizeSmallClass(small *EquivalenceClass, allClasses []*EquivalenceClass) []*models.TimeSeries {
	// Find the most similar larger class to merge with
	for _, large := range allClasses {
		if large.Size >= k.config.K && large != small {
			// Merge small class into large class
			merged := k.mergeClasses(small, large)
			if merged != nil {
				return merged
			}
		}
	}
	
	// If no suitable class found, apply maximum generalization
	return k.applyMaxGeneralization(small.Records)
}

func (k *KAnonymityProcessor) mergeClasses(small, large *EquivalenceClass) []*models.TimeSeries {
	// Apply generalization rules to make records compatible
	generalized := make([]*models.TimeSeries, 0, len(small.Records))
	
	for _, record := range small.Records {
		gen := k.generalizeRecord(record, large.Identifier)
		generalized = append(generalized, gen)
	}
	
	return generalized
}

func (k *KAnonymityProcessor) generalizeRecord(series *models.TimeSeries, targetClass string) *models.TimeSeries {
	// Create a copy of the series
	generalized := &models.TimeSeries{
		ID:         series.ID,
		Name:       series.Name,
		Points:     series.Points,
		Metadata:   make(map[string]interface{}),
		Properties: make(map[string]interface{}),
		Tags:       series.Tags,
	}
	
	// Copy non-quasi-identifier attributes
	for key, value := range series.Metadata {
		if !k.isQuasiIdentifier(key) && !k.isSensitiveAttribute(key) {
			generalized.Metadata[key] = value
		}
	}
	
	// Apply generalization to quasi-identifiers
	for _, qi := range k.config.QuasiIdentifiers {
		if rules, ok := k.config.GeneralizationRules[qi]; ok && len(rules) > 0 {
			// Apply the first level of generalization
			generalized.Metadata[qi] = rules[0]
		} else {
			// Default generalization
			generalized.Metadata[qi] = "*"
		}
	}
	
	return generalized
}

func (k *KAnonymityProcessor) applyMaxGeneralization(records []*models.TimeSeries) []*models.TimeSeries {
	generalized := make([]*models.TimeSeries, len(records))
	
	for i, record := range records {
		gen := &models.TimeSeries{
			ID:         record.ID,
			Name:       record.Name,
			Points:     record.Points,
			Metadata:   make(map[string]interface{}),
			Properties: make(map[string]interface{}),
			Tags:       record.Tags,
		}
		
		// Copy non-sensitive attributes
		for key, value := range record.Metadata {
			if !k.isQuasiIdentifier(key) && !k.isSensitiveAttribute(key) {
				gen.Metadata[key] = value
			}
		}
		
		// Maximum generalization for all quasi-identifiers
		for _, qi := range k.config.QuasiIdentifiers {
			gen.Metadata[qi] = "*"
		}
		
		generalized[i] = gen
	}
	
	return generalized
}

func (k *KAnonymityProcessor) suppressSmallGroups(dataset []*models.TimeSeries) []*models.TimeSeries {
	// Recompute equivalence classes after generalization
	classes := k.createEquivalenceClasses(dataset)
	
	totalRecords := len(dataset)
	suppressedCount := 0
	result := make([]*models.TimeSeries, 0, totalRecords)
	
	for _, class := range classes {
		if class.Size < k.config.K {
			suppressedCount += class.Size
			// Check if suppression threshold is exceeded
			if float64(suppressedCount)/float64(totalRecords) > k.config.SuppressionThreshold {
				// Add records anyway (with warning)
				k.logger.Warn("Suppression threshold exceeded, including small group")
				result = append(result, class.Records...)
			}
			// Otherwise, suppress (don't include) these records
		} else {
			result = append(result, class.Records...)
		}
	}
	
	k.logger.WithFields(logrus.Fields{
		"suppressed": suppressedCount,
		"total":      totalRecords,
		"threshold":  k.config.SuppressionThreshold,
	}).Info("Suppression complete")
	
	return result
}

func (k *KAnonymityProcessor) isQuasiIdentifier(attribute string) bool {
	for _, qi := range k.config.QuasiIdentifiers {
		if qi == attribute {
			return true
		}
	}
	return false
}

func (k *KAnonymityProcessor) isSensitiveAttribute(attribute string) bool {
	for _, sa := range k.config.SensitiveAttributes {
		if sa == attribute {
			return true
		}
	}
	return false
}

func (k *KAnonymityProcessor) ValidateKAnonymity(dataset []*models.TimeSeries) (bool, error) {
	classes := k.createEquivalenceClasses(dataset)
	
	for _, class := range classes {
		if class.Size < k.config.K && class.Size > 0 {
			return false, fmt.Errorf("equivalence class %s has size %d, less than k=%d", 
				class.Identifier, class.Size, k.config.K)
		}
	}
	
	return true, nil
}

func getDefaultKAnonymityConfig() *KAnonymityConfig {
	return &KAnonymityConfig{
		K:                    5,
		QuasiIdentifiers:     []string{"location", "age_group", "device_type"},
		GeneralizationRules: map[string][]string{
			"location":    {"city", "state", "country", "*"},
			"age_group":   {"5-year", "10-year", "generation", "*"},
			"device_type": {"model", "brand", "category", "*"},
		},
		SuppressionThreshold: 0.1,
		SensitiveAttributes:  []string{"health_data", "financial_data"},
		Metadata:             make(map[string]interface{}),
	}
}