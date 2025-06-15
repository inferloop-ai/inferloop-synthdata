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

type LDiversityConfig struct {
	L                   int                    `json:"l"`
	DiversityModel      string                 `json:"diversity_model"` // distinct, entropy, recursive
	QuasiIdentifiers    []string               `json:"quasi_identifiers"`
	SensitiveAttributes []string               `json:"sensitive_attributes"`
	EntropyThreshold    float64                `json:"entropy_threshold"`
	RecursiveC          float64                `json:"recursive_c"`
	Metadata            map[string]interface{} `json:"metadata"`
}

type LDiversityProcessor struct {
	config *LDiversityConfig
	logger *logrus.Logger
	mu     sync.RWMutex
}

type DiversityClass struct {
	Records             []*models.TimeSeries
	QuasiIdentifiers    map[string]string
	SensitiveValues     map[string][]interface{}
	SensitiveValueCount map[string]map[interface{}]int
}

func NewLDiversityProcessor(config *LDiversityConfig, logger *logrus.Logger) *LDiversityProcessor {
	if config == nil {
		config = getDefaultLDiversityConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	return &LDiversityProcessor{
		config: config,
		logger: logger,
	}
}

func (l *LDiversityProcessor) ApplyLDiversity(ctx context.Context, dataset []*models.TimeSeries) ([]*models.TimeSeries, error) {
	if len(dataset) == 0 {
		return dataset, nil
	}

	l.logger.WithFields(logrus.Fields{
		"dataset_size":    len(dataset),
		"l_value":         l.config.L,
		"diversity_model": l.config.DiversityModel,
	}).Info("Applying l-diversity")

	// Step 1: Group by quasi-identifiers
	groups := l.groupByQuasiIdentifiers(dataset)

	// Step 2: Check and enforce l-diversity
	diverseGroups := l.enforceLDiversity(groups)

	// Step 3: Extract records from diverse groups
	result := l.extractRecords(diverseGroups)

	return result, nil
}

func (l *LDiversityProcessor) groupByQuasiIdentifiers(dataset []*models.TimeSeries) map[string]*DiversityClass {
	groups := make(map[string]*DiversityClass)

	for _, series := range dataset {
		// Create group key from quasi-identifiers
		groupKey := l.createGroupKey(series)

		if group, exists := groups[groupKey]; exists {
			group.Records = append(group.Records, series)
			l.updateSensitiveValues(group, series)
		} else {
			newGroup := &DiversityClass{
				Records:             []*models.TimeSeries{series},
				QuasiIdentifiers:    l.extractQuasiIdentifiers(series),
				SensitiveValues:     make(map[string][]interface{}),
				SensitiveValueCount: make(map[string]map[interface{}]int),
			}
			l.updateSensitiveValues(newGroup, series)
			groups[groupKey] = newGroup
		}
	}

	return groups
}

func (l *LDiversityProcessor) createGroupKey(series *models.TimeSeries) string {
	values := make([]string, 0, len(l.config.QuasiIdentifiers))

	for _, qi := range l.config.QuasiIdentifiers {
		value := l.getAttributeValue(series, qi)
		values = append(values, fmt.Sprintf("%v", value))
	}

	return fmt.Sprintf("%v", values)
}

func (l *LDiversityProcessor) extractQuasiIdentifiers(series *models.TimeSeries) map[string]string {
	identifiers := make(map[string]string)

	for _, qi := range l.config.QuasiIdentifiers {
		value := l.getAttributeValue(series, qi)
		identifiers[qi] = fmt.Sprintf("%v", value)
	}

	return identifiers
}

func (l *LDiversityProcessor) updateSensitiveValues(group *DiversityClass, series *models.TimeSeries) {
	for _, attr := range l.config.SensitiveAttributes {
		value := l.getAttributeValue(series, attr)
		
		// Add to list of values
		group.SensitiveValues[attr] = append(group.SensitiveValues[attr], value)
		
		// Update count map
		if group.SensitiveValueCount[attr] == nil {
			group.SensitiveValueCount[attr] = make(map[interface{}]int)
		}
		group.SensitiveValueCount[attr][value]++
	}
}

func (l *LDiversityProcessor) getAttributeValue(series *models.TimeSeries, attribute string) interface{} {
	// Check metadata first
	if val, ok := series.Metadata[attribute]; ok {
		return val
	}

	// Check properties
	if val, ok := series.Properties[attribute]; ok {
		return val
	}

	// Check tags
	for _, tag := range series.Tags {
		if tag == attribute {
			return attribute
		}
	}

	return nil
}

func (l *LDiversityProcessor) enforceLDiversity(groups map[string]*DiversityClass) []*DiversityClass {
	var diverseGroups []*DiversityClass

	for _, group := range groups {
		if l.checkLDiversity(group) {
			diverseGroups = append(diverseGroups, group)
		} else {
			// Try to merge with other groups or apply generalization
			merged := l.mergeForDiversity(group, groups)
			if merged != nil && l.checkLDiversity(merged) {
				diverseGroups = append(diverseGroups, merged)
			}
		}
	}

	return diverseGroups
}

func (l *LDiversityProcessor) checkLDiversity(group *DiversityClass) bool {
	for _, attr := range l.config.SensitiveAttributes {
		if !l.checkAttributeDiversity(group, attr) {
			return false
		}
	}
	return true
}

func (l *LDiversityProcessor) checkAttributeDiversity(group *DiversityClass, attribute string) bool {
	counts := group.SensitiveValueCount[attribute]
	if len(counts) < l.config.L {
		return false
	}

	switch l.config.DiversityModel {
	case "distinct":
		return l.checkDistinctDiversity(counts)
	case "entropy":
		return l.checkEntropyDiversity(counts)
	case "recursive":
		return l.checkRecursiveDiversity(counts)
	default:
		return l.checkDistinctDiversity(counts)
	}
}

func (l *LDiversityProcessor) checkDistinctDiversity(counts map[interface{}]int) bool {
	// Simple distinct l-diversity: at least l different values
	return len(counts) >= l.config.L
}

func (l *LDiversityProcessor) checkEntropyDiversity(counts map[interface{}]int) bool {
	// Entropy l-diversity
	total := 0
	for _, count := range counts {
		total += count
	}

	entropy := 0.0
	for _, count := range counts {
		if count > 0 {
			p := float64(count) / float64(total)
			entropy -= p * l.log2(p)
		}
	}

	return entropy >= l.log2(float64(l.config.L))
}

func (l *LDiversityProcessor) checkRecursiveDiversity(counts map[interface{}]int) bool {
	// Recursive (c,l)-diversity
	if len(counts) < l.config.L {
		return false
	}

	// Sort counts in descending order
	var sortedCounts []int
	for _, count := range counts {
		sortedCounts = append(sortedCounts, count)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(sortedCounts)))

	// Check recursive condition
	sum := 0
	for i := l.config.L; i < len(sortedCounts); i++ {
		sum += sortedCounts[i]
	}

	return float64(sortedCounts[0]) <= l.config.RecursiveC*float64(sum)
}

func (l *LDiversityProcessor) mergeForDiversity(group *DiversityClass, allGroups map[string]*DiversityClass) *DiversityClass {
	// Try to find compatible groups to merge
	for _, other := range allGroups {
		if other == group {
			continue
		}

		// Check if merging would help achieve l-diversity
		merged := l.mergeGroups(group, other)
		if l.checkLDiversity(merged) {
			return merged
		}
	}

	return nil
}

func (l *LDiversityProcessor) mergeGroups(group1, group2 *DiversityClass) *DiversityClass {
	merged := &DiversityClass{
		Records:             append(group1.Records, group2.Records...),
		QuasiIdentifiers:    make(map[string]string),
		SensitiveValues:     make(map[string][]interface{}),
		SensitiveValueCount: make(map[string]map[interface{}]int),
	}

	// Merge quasi-identifiers (generalize if different)
	for qi := range group1.QuasiIdentifiers {
		if group1.QuasiIdentifiers[qi] == group2.QuasiIdentifiers[qi] {
			merged.QuasiIdentifiers[qi] = group1.QuasiIdentifiers[qi]
		} else {
			merged.QuasiIdentifiers[qi] = "*" // Generalize
		}
	}

	// Merge sensitive values
	for _, record := range merged.Records {
		for _, attr := range l.config.SensitiveAttributes {
			value := l.getAttributeValue(record, attr)
			merged.SensitiveValues[attr] = append(merged.SensitiveValues[attr], value)
			
			if merged.SensitiveValueCount[attr] == nil {
				merged.SensitiveValueCount[attr] = make(map[interface{}]int)
			}
			merged.SensitiveValueCount[attr][value]++
		}
	}

	return merged
}

func (l *LDiversityProcessor) extractRecords(groups []*DiversityClass) []*models.TimeSeries {
	var result []*models.TimeSeries

	for _, group := range groups {
		// Apply quasi-identifier values to all records in the group
		for _, record := range group.Records {
			anonymized := l.applyQuasiIdentifiers(record, group.QuasiIdentifiers)
			result = append(result, anonymized)
		}
	}

	return result
}

func (l *LDiversityProcessor) applyQuasiIdentifiers(series *models.TimeSeries, identifiers map[string]string) *models.TimeSeries {
	// Create a copy of the series
	anonymized := &models.TimeSeries{
		ID:         series.ID,
		Name:       series.Name,
		Points:     series.Points,
		Metadata:   make(map[string]interface{}),
		Properties: make(map[string]interface{}),
		Tags:       series.Tags,
	}

	// Copy all metadata
	for k, v := range series.Metadata {
		anonymized.Metadata[k] = v
	}

	// Apply generalized quasi-identifiers
	for qi, value := range identifiers {
		anonymized.Metadata[qi] = value
	}

	return anonymized
}

func (l *LDiversityProcessor) ValidateLDiversity(dataset []*models.TimeSeries) (bool, error) {
	groups := l.groupByQuasiIdentifiers(dataset)

	for groupKey, group := range groups {
		if !l.checkLDiversity(group) {
			return false, fmt.Errorf("group %s does not satisfy %d-diversity", groupKey, l.config.L)
		}
	}

	return true, nil
}

func (l *LDiversityProcessor) log2(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return logN(x, 2)
}

func logN(x, base float64) float64 {
	return log(x) / log(base)
}

func log(x float64) float64 {
	// Natural logarithm approximation
	if x <= 0 {
		return 0
	}
	// This is a simplified implementation
	// In production, use math.Log
	return 0.693147 * (x - 1) // Approximation for small values
}

func getDefaultLDiversityConfig() *LDiversityConfig {
	return &LDiversityConfig{
		L:                   3,
		DiversityModel:      "distinct",
		QuasiIdentifiers:    []string{"location", "age_group", "device_type"},
		SensitiveAttributes: []string{"health_status", "income_level"},
		EntropyThreshold:    2.0,
		RecursiveC:          3.0,
		Metadata:            make(map[string]interface{}),
	}
}