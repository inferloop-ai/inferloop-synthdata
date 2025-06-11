package api

import (
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/internal/api/handlers"
	"github.com/inferloop/tsiot/internal/generators"
	"github.com/inferloop/tsiot/internal/storage"
)

// Handlers contains all HTTP handlers for the API
type Handlers struct {
	Generation *handlers.GenerationHandler
	Analytics  *handlers.AnalyticsHandler  
	Health     *handlers.HealthHandler
	TimeSeries *handlers.TimeSeriesHandler
	Validation *handlers.ValidationHandler
	Workflows  *handlers.WorkflowsHandler
	Agents     *handlers.AgentsHandler
}

// HandlerConfig contains configuration for handlers
type HandlerConfig struct {
	GeneratorFactory *generators.Factory
	StorageFactory   *storage.Factory
	Logger           *logrus.Logger
}

// NewHandlers creates a new handlers instance with all HTTP handlers
func NewHandlers(config *HandlerConfig) *Handlers {
	if config.Logger == nil {
		config.Logger = logrus.New()
	}

	return &Handlers{
		Generation: handlers.NewGenerationHandler(),
		Analytics:  handlers.NewAnalyticsHandler(),
		Health:     handlers.NewHealthHandler(),
		TimeSeries: handlers.NewTimeSeriesHandler(),
		Validation: handlers.NewValidationHandler(),
		Workflows:  handlers.NewWorkflowsHandler(),
		Agents:     handlers.NewAgentsHandler(),
	}
}

// Close closes all handlers and cleans up resources
func (h *Handlers) Close() error {
	// Close any resources held by handlers
	if h.Generation != nil {
		// Generation handler cleanup if needed
	}
	
	if h.Analytics != nil {
		// Analytics handler cleanup if needed
	}
	
	if h.Health != nil {
		// Health handler cleanup if needed
	}
	
	if h.TimeSeries != nil {
		// TimeSeries handler cleanup if needed  
	}
	
	if h.Validation != nil {
		// Validation handler cleanup if needed
	}
	
	if h.Workflows != nil {
		// Workflows handler cleanup if needed
	}
	
	if h.Agents != nil {
		// Agents handler cleanup if needed
	}
	
	return nil
}