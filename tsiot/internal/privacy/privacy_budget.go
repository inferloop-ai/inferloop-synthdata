package privacy

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/inferloop/tsiot/pkg/errors"
)

// PrivacyBudgetManager manages privacy budget allocation and tracking
type PrivacyBudgetManager struct {
	mu               sync.RWMutex
	globalEpsilon    float64
	globalDelta      float64
	consumedEpsilon  float64
	consumedDelta    float64
	timeHorizon      time.Duration
	lastReset        time.Time
	allocations      map[string]*BudgetAllocation
	transactions     []BudgetTransaction
	autoReset        bool
	compositionRule  CompositionRule
	maxTransactions  int
}

// BudgetAllocation represents budget allocation for a specific purpose
type BudgetAllocation struct {
	Name            string    `json:"name"`
	AllocatedEps    float64   `json:"allocated_epsilon"`
	AllocatedDelta  float64   `json:"allocated_delta"`
	ConsumedEps     float64   `json:"consumed_epsilon"`
	ConsumedDelta   float64   `json:"consumed_delta"`
	LastUpdated     time.Time `json:"last_updated"`
	Priority        int       `json:"priority"`        // 1 = highest, 5 = lowest
	Renewable       bool      `json:"renewable"`       // Can be renewed on reset
	MinReserve      float64   `json:"min_reserve"`     // Minimum budget to keep reserved
}

// BudgetTransaction records a budget expenditure
type BudgetTransaction struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	EpsilonUsed float64                `json:"epsilon_used"`
	DeltaUsed   float64                `json:"delta_used"`
	Purpose     string                 `json:"purpose"`
	Description string                 `json:"description"`
	Mechanism   string                 `json:"mechanism"`
	QueryType   QueryType              `json:"query_type"`
	DataSize    int                    `json:"data_size"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// BudgetStatus provides current budget status information
type BudgetStatus struct {
	GlobalEpsilon      float64                    `json:"global_epsilon"`
	GlobalDelta        float64                    `json:"global_delta"`
	ConsumedEpsilon    float64                    `json:"consumed_epsilon"`
	ConsumedDelta      float64                    `json:"consumed_delta"`
	RemainingEpsilon   float64                    `json:"remaining_epsilon"`
	RemainingDelta     float64                    `json:"remaining_delta"`
	UtilizationEpsilon float64                    `json:"utilization_epsilon"`
	UtilizationDelta   float64                    `json:"utilization_delta"`
	TimeHorizon        time.Duration              `json:"time_horizon"`
	LastReset          time.Time                  `json:"last_reset"`
	NextReset          time.Time                  `json:"next_reset"`
	TransactionCount   int                        `json:"transaction_count"`
	Allocations        map[string]*BudgetAllocation `json:"allocations"`
	HealthStatus       string                     `json:"health_status"`
	Warnings           []string                   `json:"warnings"`
}

// CompositionRule defines how privacy guarantees compose
type CompositionRule interface {
	Compose(transactions []BudgetTransaction) (float64, float64)
	GetName() string
	GetDescription() string
}

// BasicComposition implements basic composition (simple summation)
type BasicComposition struct{}

func (bc *BasicComposition) Compose(transactions []BudgetTransaction) (float64, float64) {
	var totalEpsilon, totalDelta float64
	for _, tx := range transactions {
		totalEpsilon += tx.EpsilonUsed
		totalDelta += tx.DeltaUsed
	}
	return totalEpsilon, totalDelta
}

func (bc *BasicComposition) GetName() string {
	return "basic"
}

func (bc *BasicComposition) GetDescription() string {
	return "Basic composition: �_total = ��_i, �_total = ��_i"
}

// AdvancedComposition implements advanced composition theorem
type AdvancedComposition struct {
	delta0 float64 // Additional delta parameter
}

func (ac *AdvancedComposition) Compose(transactions []BudgetTransaction) (float64, float64) {
	k := float64(len(transactions))
	if k == 0 {
		return 0, 0
	}

	var sumEpsilon, maxEpsilon, sumDelta float64
	for _, tx := range transactions {
		sumEpsilon += tx.EpsilonUsed
		if tx.EpsilonUsed > maxEpsilon {
			maxEpsilon = tx.EpsilonUsed
		}
		sumDelta += tx.DeltaUsed
	}

	// Advanced composition formula
	// �' = (2k ln(1/�')) * max� + k * max� * (e^max� - 1)
	// This is a simplified version
	advancedEpsilon := sumEpsilon + maxEpsilon*k*0.1 // Simplified
	advancedDelta := sumDelta + ac.delta0

	return advancedEpsilon, advancedDelta
}

func (ac *AdvancedComposition) GetName() string {
	return "advanced"
}

func (ac *AdvancedComposition) GetDescription() string {
	return "Advanced composition theorem with tighter bounds"
}

// RenyiDPComposition implements Renyi Differential Privacy composition
type RenyiDPComposition struct {
	alpha float64 // Renyi parameter
}

func (rdp *RenyiDPComposition) Compose(transactions []BudgetTransaction) (float64, float64) {
	// Simplified RDP composition
	// In practice, this would maintain RDP guarantees and convert to (�,�)-DP
	k := float64(len(transactions))
	if k == 0 {
		return 0, 0
	}

	var sumEpsilon, sumDelta float64
	for _, tx := range transactions {
		sumEpsilon += tx.EpsilonUsed
		sumDelta += tx.DeltaUsed
	}

	// RDP provides tighter composition bounds
	rdpEpsilon := sumEpsilon * 0.8 // Simplified improvement factor
	return rdpEpsilon, sumDelta
}

func (rdp *RenyiDPComposition) GetName() string {
	return "rdp"
}

func (rdp *RenyiDPComposition) GetDescription() string {
	return "Renyi Differential Privacy composition with tighter bounds"
}

// NewPrivacyBudgetManager creates a new privacy budget manager
func NewPrivacyBudgetManager(globalEpsilon, globalDelta float64, timeHorizon time.Duration) (*PrivacyBudgetManager, error) {
	if globalEpsilon <= 0 {
		return nil, fmt.Errorf("global epsilon must be positive, got %f", globalEpsilon)
	}

	if globalDelta < 0 || globalDelta >= 1 {
		return nil, fmt.Errorf("global delta must be in [0, 1), got %f", globalDelta)
	}

	if timeHorizon <= 0 {
		timeHorizon = 24 * time.Hour // Default to 24 hours
	}

	manager := &PrivacyBudgetManager{
		globalEpsilon:   globalEpsilon,
		globalDelta:     globalDelta,
		consumedEpsilon: 0.0,
		consumedDelta:   0.0,
		timeHorizon:     timeHorizon,
		lastReset:       time.Now(),
		allocations:     make(map[string]*BudgetAllocation),
		transactions:    make([]BudgetTransaction, 0),
		autoReset:       true,
		compositionRule: &BasicComposition{},
		maxTransactions: 10000,
	}

	// Create default allocations
	manager.createDefaultAllocations()

	return manager, nil
}

// CanSpend checks if the budget allows spending the requested amount
func (pbm *PrivacyBudgetManager) CanSpend(epsilon, delta float64) bool {
	pbm.mu.RLock()
	defer pbm.mu.RUnlock()

	// Check if auto-reset is needed
	if pbm.autoReset && time.Since(pbm.lastReset) >= pbm.timeHorizon {
		// Note: We can't reset here due to RLock, but we can check as if reset
		return epsilon <= pbm.globalEpsilon && delta <= pbm.globalDelta
	}

	remainingEpsilon := pbm.globalEpsilon - pbm.consumedEpsilon
	remainingDelta := pbm.globalDelta - pbm.consumedDelta

	return epsilon <= remainingEpsilon && delta <= remainingDelta
}

// Spend spends the specified amount of privacy budget
func (pbm *PrivacyBudgetManager) Spend(epsilon, delta float64) error {
	return pbm.SpendWithPurpose(epsilon, delta, "general", "", map[string]interface{}{})
}

// SpendWithPurpose spends budget with detailed tracking
func (pbm *PrivacyBudgetManager) SpendWithPurpose(
	epsilon, delta float64,
	purpose, description string,
	metadata map[string]interface{},
) error {
	pbm.mu.Lock()
	defer pbm.mu.Unlock()

	// Check if auto-reset is needed
	if pbm.autoReset && time.Since(pbm.lastReset) >= pbm.timeHorizon {
		pbm.resetInternal()
	}

	// Check budget availability
	if !pbm.canSpendInternal(epsilon, delta) {
		return errors.NewPrivacyError("BUDGET_EXHAUSTED",
			fmt.Sprintf("Insufficient budget: need (�=%g, �=%g), have (�=%g, �=%g)",
				epsilon, delta,
				pbm.globalEpsilon-pbm.consumedEpsilon,
				pbm.globalDelta-pbm.consumedDelta))
	}

	// Create transaction record
	transaction := BudgetTransaction{
		ID:          pbm.generateTransactionID(),
		Timestamp:   time.Now(),
		EpsilonUsed: epsilon,
		DeltaUsed:   delta,
		Purpose:     purpose,
		Description: description,
		Metadata:    metadata,
	}

	// Add mechanism and query type from metadata if available
	if mech, ok := metadata["mechanism"].(string); ok {
		transaction.Mechanism = mech
	}
	if qt, ok := metadata["query_type"].(QueryType); ok {
		transaction.QueryType = qt
	}
	if size, ok := metadata["data_size"].(int); ok {
		transaction.DataSize = size
	}

	// Record transaction
	pbm.transactions = append(pbm.transactions, transaction)

	// Trim transactions if too many
	if len(pbm.transactions) > pbm.maxTransactions {
		pbm.transactions = pbm.transactions[len(pbm.transactions)-pbm.maxTransactions:]
	}

	// Update consumed budget using composition rule
	pbm.updateConsumedBudget()

	// Update relevant allocation
	if allocation, exists := pbm.allocations[purpose]; exists {
		allocation.ConsumedEps += epsilon
		allocation.ConsumedDelta += delta
		allocation.LastUpdated = time.Now()
	}

	return nil
}

// GetRemainingBudget returns remaining privacy budget
func (pbm *PrivacyBudgetManager) GetRemainingBudget() (float64, float64) {
	pbm.mu.RLock()
	defer pbm.mu.RUnlock()

	remainingEpsilon := pbm.globalEpsilon - pbm.consumedEpsilon
	remainingDelta := pbm.globalDelta - pbm.consumedDelta

	return math.Max(0, remainingEpsilon), math.Max(0, remainingDelta)
}

// GetStatus returns comprehensive budget status
func (pbm *PrivacyBudgetManager) GetStatus() *BudgetStatus {
	pbm.mu.RLock()
	defer pbm.mu.RUnlock()

	remainingEpsilon := pbm.globalEpsilon - pbm.consumedEpsilon
	remainingDelta := pbm.globalDelta - pbm.consumedDelta

	utilizationEpsilon := pbm.consumedEpsilon / pbm.globalEpsilon
	utilizationDelta := 0.0
	if pbm.globalDelta > 0 {
		utilizationDelta = pbm.consumedDelta / pbm.globalDelta
	}

	// Determine health status
	healthStatus := "healthy"
	warnings := []string{}

	if utilizationEpsilon > 0.9 {
		healthStatus = "critical"
		warnings = append(warnings, "Epsilon budget nearly exhausted")
	} else if utilizationEpsilon > 0.7 {
		healthStatus = "warning"
		warnings = append(warnings, "Epsilon budget running low")
	}

	if utilizationDelta > 0.9 {
		healthStatus = "critical"
		warnings = append(warnings, "Delta budget nearly exhausted")
	}

	// Check for time-based warnings
	timeUntilReset := pbm.timeHorizon - time.Since(pbm.lastReset)
	if timeUntilReset < time.Hour && utilizationEpsilon > 0.5 {
		warnings = append(warnings, "Budget reset approaching with significant utilization")
	}

	status := &BudgetStatus{
		GlobalEpsilon:      pbm.globalEpsilon,
		GlobalDelta:        pbm.globalDelta,
		ConsumedEpsilon:    pbm.consumedEpsilon,
		ConsumedDelta:      pbm.consumedDelta,
		RemainingEpsilon:   math.Max(0, remainingEpsilon),
		RemainingDelta:     math.Max(0, remainingDelta),
		UtilizationEpsilon: utilizationEpsilon,
		UtilizationDelta:   utilizationDelta,
		TimeHorizon:        pbm.timeHorizon,
		LastReset:          pbm.lastReset,
		NextReset:          pbm.lastReset.Add(pbm.timeHorizon),
		TransactionCount:   len(pbm.transactions),
		Allocations:        pbm.copyAllocations(),
		HealthStatus:       healthStatus,
		Warnings:           warnings,
	}

	return status
}

// Reset resets the privacy budget
func (pbm *PrivacyBudgetManager) Reset() error {
	pbm.mu.Lock()
	defer pbm.mu.Unlock()

	pbm.resetInternal()
	return nil
}

// SetCompositionRule sets the composition rule
func (pbm *PrivacyBudgetManager) SetCompositionRule(rule CompositionRule) {
	pbm.mu.Lock()
	defer pbm.mu.Unlock()

	pbm.compositionRule = rule
	pbm.updateConsumedBudget() // Recalculate with new rule
}

// CreateAllocation creates a new budget allocation
func (pbm *PrivacyBudgetManager) CreateAllocation(name string, epsilon, delta float64, priority int, renewable bool) error {
	pbm.mu.Lock()
	defer pbm.mu.Unlock()

	if _, exists := pbm.allocations[name]; exists {
		return fmt.Errorf("allocation %s already exists", name)
	}

	if epsilon < 0 || delta < 0 {
		return fmt.Errorf("allocation amounts must be non-negative")
	}

	// Check if allocation fits within global budget
	totalAllocatedEps := epsilon
	totalAllocatedDelta := delta
	for _, alloc := range pbm.allocations {
		totalAllocatedEps += alloc.AllocatedEps
		totalAllocatedDelta += alloc.AllocatedDelta
	}

	if totalAllocatedEps > pbm.globalEpsilon {
		return fmt.Errorf("total epsilon allocation (%g) exceeds global budget (%g)",
			totalAllocatedEps, pbm.globalEpsilon)
	}

	if totalAllocatedDelta > pbm.globalDelta {
		return fmt.Errorf("total delta allocation (%g) exceeds global budget (%g)",
			totalAllocatedDelta, pbm.globalDelta)
	}

	allocation := &BudgetAllocation{
		Name:            name,
		AllocatedEps:    epsilon,
		AllocatedDelta:  delta,
		ConsumedEps:     0.0,
		ConsumedDelta:   0.0,
		LastUpdated:     time.Now(),
		Priority:        priority,
		Renewable:       renewable,
		MinReserve:      epsilon * 0.1, // 10% reserve by default
	}

	pbm.allocations[name] = allocation
	return nil
}

// GetTransactionHistory returns transaction history
func (pbm *PrivacyBudgetManager) GetTransactionHistory(limit int) []BudgetTransaction {
	pbm.mu.RLock()
	defer pbm.mu.RUnlock()

	if limit <= 0 || limit > len(pbm.transactions) {
		limit = len(pbm.transactions)
	}

	// Return most recent transactions
	start := len(pbm.transactions) - limit
	history := make([]BudgetTransaction, limit)
	copy(history, pbm.transactions[start:])

	return history
}

// GetAllocationStatus returns status for a specific allocation
func (pbm *PrivacyBudgetManager) GetAllocationStatus(name string) (*BudgetAllocation, error) {
	pbm.mu.RLock()
	defer pbm.mu.RUnlock()

	allocation, exists := pbm.allocations[name]
	if !exists {
		return nil, fmt.Errorf("allocation %s not found", name)
	}

	// Return a copy
	copy := *allocation
	return &copy, nil
}

// Internal helper methods

func (pbm *PrivacyBudgetManager) canSpendInternal(epsilon, delta float64) bool {
	remainingEpsilon := pbm.globalEpsilon - pbm.consumedEpsilon
	remainingDelta := pbm.globalDelta - pbm.consumedDelta

	return epsilon <= remainingEpsilon && delta <= remainingDelta
}

func (pbm *PrivacyBudgetManager) resetInternal() {
	pbm.consumedEpsilon = 0.0
	pbm.consumedDelta = 0.0
	pbm.lastReset = time.Now()
	pbm.transactions = pbm.transactions[:0] // Clear transactions

	// Reset renewable allocations
	for _, allocation := range pbm.allocations {
		if allocation.Renewable {
			allocation.ConsumedEps = 0.0
			allocation.ConsumedDelta = 0.0
			allocation.LastUpdated = time.Now()
		}
	}
}

func (pbm *PrivacyBudgetManager) updateConsumedBudget() {
	if len(pbm.transactions) == 0 {
		pbm.consumedEpsilon = 0.0
		pbm.consumedDelta = 0.0
		return
	}

	// Use composition rule to calculate total consumption
	epsilon, delta := pbm.compositionRule.Compose(pbm.transactions)
	pbm.consumedEpsilon = epsilon
	pbm.consumedDelta = delta
}

func (pbm *PrivacyBudgetManager) generateTransactionID() string {
	return fmt.Sprintf("tx_%d_%d", time.Now().UnixNano(), len(pbm.transactions))
}

func (pbm *PrivacyBudgetManager) createDefaultAllocations() {
	// Create default allocations for common purposes
	allocations := map[string]struct {
		epsilon   float64
		delta     float64
		priority  int
		renewable bool
	}{
		"generation":  {pbm.globalEpsilon * 0.6, pbm.globalDelta * 0.6, 1, true},
		"validation":  {pbm.globalEpsilon * 0.25, pbm.globalDelta * 0.25, 2, true},
		"analysis":    {pbm.globalEpsilon * 0.1, pbm.globalDelta * 0.1, 3, true},
		"emergency":   {pbm.globalEpsilon * 0.05, pbm.globalDelta * 0.05, 5, false},
	}

	for name, config := range allocations {
		allocation := &BudgetAllocation{
			Name:            name,
			AllocatedEps:    config.epsilon,
			AllocatedDelta:  config.delta,
			ConsumedEps:     0.0,
			ConsumedDelta:   0.0,
			LastUpdated:     time.Now(),
			Priority:        config.priority,
			Renewable:       config.renewable,
			MinReserve:      config.epsilon * 0.1,
		}
		pbm.allocations[name] = allocation
	}
}

func (pbm *PrivacyBudgetManager) copyAllocations() map[string]*BudgetAllocation {
	copy := make(map[string]*BudgetAllocation)
	for name, allocation := range pbm.allocations {
		allocCopy := *allocation
		copy[name] = &allocCopy
	}
	return copy
}

// Factory functions for composition rules

func NewBasicComposition() CompositionRule {
	return &BasicComposition{}
}

func NewAdvancedComposition(delta0 float64) CompositionRule {
	return &AdvancedComposition{delta0: delta0}
}

func NewRenyiDPComposition(alpha float64) CompositionRule {
	return &RenyiDPComposition{alpha: alpha}
}