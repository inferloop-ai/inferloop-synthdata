package api

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/inferloop/tsiot/internal/api/handlers"
	"github.com/inferloop/tsiot/pkg/models"
)

type Router struct {
	generationHandler *handlers.GenerationHandler
	validationHandler *handlers.ValidationHandler
	timeseriesHandler *handlers.TimeSeriesHandler
	analyticsHandler  *handlers.AnalyticsHandler
	healthHandler     *handlers.HealthHandler
	workflowsHandler  *handlers.WorkflowsHandler
	agentsHandler     *handlers.AgentsHandler
}

func NewRouter() *Router {
	timeseriesHandler := handlers.NewTimeSeriesHandler()
	analyticsHandler := handlers.NewAnalyticsHandler()
	
	// Share storage between handlers
	analyticsHandler.SetTimeSeriesStorage(timeseriesHandler.GetStorage())
	
	return &Router{
		generationHandler: handlers.NewGenerationHandler(),
		validationHandler: handlers.NewValidationHandler(),
		timeseriesHandler: timeseriesHandler,
		analyticsHandler:  analyticsHandler,
		healthHandler:     handlers.NewHealthHandler("1.0.0", "development"),
		workflowsHandler:  handlers.NewWorkflowsHandler(),
		agentsHandler:     handlers.NewAgentsHandler(),
	}
}

func (router *Router) SetupRoutes() *mux.Router {
	r := mux.NewRouter()
	
	// Apply middleware
	config := DefaultMiddlewareConfig()
	r = ApplyMiddleware(r, config)
	
	// Add metrics endpoint
	r.HandleFunc("/metrics", MetricsHandler).Methods("GET")
	
	// API version prefix
	api := r.PathPrefix("/api/v1").Subrouter()
	
	// Health endpoints
	health := api.PathPrefix("/health").Subrouter()
	health.HandleFunc("", router.healthHandler.GetHealth).Methods("GET")
	health.HandleFunc("/live", router.healthHandler.GetLiveness).Methods("GET")
	health.HandleFunc("/ready", router.healthHandler.GetReadiness).Methods("GET")
	health.HandleFunc("/metrics", router.healthHandler.GetMetrics).Methods("GET")
	health.HandleFunc("/dependencies", router.healthHandler.GetDependencies).Methods("GET")
	health.HandleFunc("/version", router.healthHandler.GetVersion).Methods("GET")
	
	// Generation endpoints
	generation := api.PathPrefix("/generate").Subrouter()
	generation.HandleFunc("", router.generationHandler.GenerateTimeSeries).Methods("POST")
	generation.HandleFunc("/batch", router.generationHandler.GenerateBatch).Methods("POST")
	generation.HandleFunc("/generators", router.generationHandler.GetGenerators).Methods("GET")
	generation.HandleFunc("/generators/{id}", router.generationHandler.GetGeneratorDetails).Methods("GET")
	generation.HandleFunc("/templates", router.generationHandler.GetTemplates).Methods("GET")
	generation.HandleFunc("/templates/{id}", router.generationHandler.GenerateFromTemplate).Methods("POST")
	
	// Validation endpoints
	validation := api.PathPrefix("/validate").Subrouter()
	validation.HandleFunc("", router.validationHandler.ValidateTimeSeries).Methods("POST")
	validation.HandleFunc("/batch", router.validationHandler.ValidateBatch).Methods("POST")
	validation.HandleFunc("/metrics", router.validationHandler.GetValidationMetrics).Methods("GET")
	validation.HandleFunc("/compare", router.validationHandler.CompareTimeSeries).Methods("POST")
	validation.HandleFunc("/reports/{id}", router.validationHandler.GetQualityReport).Methods("GET")
	
	// TimeSeries endpoints
	timeseries := api.PathPrefix("/timeseries").Subrouter()
	timeseries.HandleFunc("", router.timeseriesHandler.CreateTimeSeries).Methods("POST")
	timeseries.HandleFunc("", router.timeseriesHandler.ListTimeSeries).Methods("GET")
	timeseries.HandleFunc("/search", router.timeseriesHandler.SearchTimeSeries).Methods("GET")
	timeseries.HandleFunc("/{id}", router.timeseriesHandler.GetTimeSeries).Methods("GET")
	timeseries.HandleFunc("/{id}", router.timeseriesHandler.UpdateTimeSeries).Methods("PUT")
	timeseries.HandleFunc("/{id}", router.timeseriesHandler.DeleteTimeSeries).Methods("DELETE")
	timeseries.HandleFunc("/{id}/data", router.timeseriesHandler.GetTimeSeriesData).Methods("GET")
	timeseries.HandleFunc("/{id}/data", router.timeseriesHandler.AppendData).Methods("POST")
	timeseries.HandleFunc("/{id}/stats", router.timeseriesHandler.GetTimeSeriesStats).Methods("GET")
	
	// Analytics endpoints
	analytics := api.PathPrefix("/analytics").Subrouter()
	analytics.HandleFunc("/system", router.analyticsHandler.GetSystemMetrics).Methods("GET")
	analytics.HandleFunc("/usage", router.analyticsHandler.GetUsageStatistics).Methods("GET")
	analytics.HandleFunc("/quality", router.analyticsHandler.GetQualityMetrics).Methods("GET")
	analytics.HandleFunc("/performance", router.analyticsHandler.GetPerformanceMetrics).Methods("GET")
	analytics.HandleFunc("/timeseries/{id}", router.analyticsHandler.AnalyzeTimeSeries).Methods("GET")
	analytics.HandleFunc("/compare", router.analyticsHandler.CompareMultipleTimeSeries).Methods("POST")
	analytics.HandleFunc("/correlation", router.analyticsHandler.GetCorrelationMatrix).Methods("POST")
	analytics.HandleFunc("/anomalies/{id}", router.analyticsHandler.GetAnomalyDetection).Methods("GET")
	analytics.HandleFunc("/trends/{id}", router.analyticsHandler.GetTrendAnalysis).Methods("GET")
	
	// Workflows endpoints
	workflows := api.PathPrefix("/workflows").Subrouter()
	workflows.HandleFunc("", router.workflowsHandler.CreateWorkflow).Methods("POST")
	workflows.HandleFunc("", router.workflowsHandler.ListWorkflows).Methods("GET")
	workflows.HandleFunc("/templates", router.workflowsHandler.GetWorkflowTemplates).Methods("GET")
	workflows.HandleFunc("/templates/{templateId}", router.workflowsHandler.CreateFromTemplate).Methods("POST")
	workflows.HandleFunc("/{id}", router.workflowsHandler.GetWorkflow).Methods("GET")
	workflows.HandleFunc("/{id}", router.workflowsHandler.UpdateWorkflow).Methods("PUT")
	workflows.HandleFunc("/{id}", router.workflowsHandler.DeleteWorkflow).Methods("DELETE")
	workflows.HandleFunc("/{id}/execute", router.workflowsHandler.ExecuteWorkflow).Methods("POST")
	
	// Workflow executions endpoints
	executions := api.PathPrefix("/executions").Subrouter()
	executions.HandleFunc("", router.workflowsHandler.ListExecutions).Methods("GET")
	executions.HandleFunc("/{id}", router.workflowsHandler.GetExecution).Methods("GET")
	executions.HandleFunc("/{id}/stop", router.workflowsHandler.StopExecution).Methods("POST")
	
	// Agents endpoints
	agents := api.PathPrefix("/agents").Subrouter()
	agents.HandleFunc("", router.agentsHandler.CreateAgent).Methods("POST")
	agents.HandleFunc("", router.agentsHandler.ListAgents).Methods("GET")
	agents.HandleFunc("/types", router.agentsHandler.GetAgentTypes).Methods("GET")
	agents.HandleFunc("/{id}", router.agentsHandler.GetAgent).Methods("GET")
	agents.HandleFunc("/{id}", router.agentsHandler.UpdateAgent).Methods("PUT")
	agents.HandleFunc("/{id}", router.agentsHandler.DeleteAgent).Methods("DELETE")
	agents.HandleFunc("/{id}/start", router.agentsHandler.StartAgent).Methods("POST")
	agents.HandleFunc("/{id}/stop", router.agentsHandler.StopAgent).Methods("POST")
	agents.HandleFunc("/{id}/health", router.agentsHandler.GetAgentHealth).Methods("GET")
	agents.HandleFunc("/{id}/metrics", router.agentsHandler.GetAgentMetrics).Methods("GET")
	
	// Agent coordinators endpoints
	coordinators := api.PathPrefix("/coordinators").Subrouter()
	coordinators.HandleFunc("", router.agentsHandler.CreateCoordinator).Methods("POST")
	coordinators.HandleFunc("", router.agentsHandler.ListCoordinators).Methods("GET")
	
	// Agent tasks endpoints
	tasks := api.PathPrefix("/tasks").Subrouter()
	tasks.HandleFunc("", router.agentsHandler.AssignTask).Methods("POST")
	tasks.HandleFunc("", router.agentsHandler.ListTasks).Methods("GET")
	tasks.HandleFunc("/{id}", router.agentsHandler.GetTask).Methods("GET")
	
	// Root endpoints
	r.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"service": "TimeSeries Synthetic MCP",
			"version": "1.0.0",
			"status":  "running",
			"endpoints": map[string]string{
				"health":     "/api/v1/health",
				"generate":   "/api/v1/generate",
				"validate":   "/api/v1/validate",
				"timeseries": "/api/v1/timeseries",
				"analytics":  "/api/v1/analytics",
				"workflows":  "/api/v1/workflows",
				"agents":     "/api/v1/agents",
			},
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}).Methods("GET")
	
	// CORS preflight for all routes
	r.Methods("OPTIONS").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.WriteHeader(http.StatusOK)
	})
	
	return r
}

