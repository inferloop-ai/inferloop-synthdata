package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/inferloop/tsiot/internal/generators"
	"github.com/inferloop/tsiot/pkg/models"
)

type GenerationHandler struct {
	generatorFactory *generators.Factory
}

func NewGenerationHandler() *GenerationHandler {
	return &GenerationHandler{
		generatorFactory: generators.NewFactory(nil),
	}
}

func (h *GenerationHandler) GenerateTimeSeries(w http.ResponseWriter, r *http.Request) {
	var request models.GenerationRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if err := h.validateGenerationRequest(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	generator, err := h.generatorFactory.CreateGenerator(request.GeneratorType)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create generator: %v", err), http.StatusBadRequest)
		return
	}

	timeSeries, err := generator.Generate(r.Context(), &request)
	if err != nil {
		http.Error(w, fmt.Sprintf("Generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":     "success",
		"timeSeries": timeSeries,
		"metadata": map[string]interface{}{
			"generatorType": request.GeneratorType,
			"parameters":    request.Parameters,
			"generatedAt":   time.Now(),
			"dataPoints":    len(timeSeries.DataPoints),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *GenerationHandler) GenerateBatch(w http.ResponseWriter, r *http.Request) {
	var batchRequest struct {
		Requests []models.GenerationRequest `json:"requests"`
		Parallel bool                       `json:"parallel"`
	}

	if err := json.NewDecoder(r.Body).Decode(&batchRequest); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if len(batchRequest.Requests) == 0 {
		http.Error(w, "No generation requests provided", http.StatusBadRequest)
		return
	}

	if len(batchRequest.Requests) > 100 {
		http.Error(w, "Too many requests (max 100)", http.StatusBadRequest)
		return
	}

	var results []map[string]interface{}
	var errors []string

	if batchRequest.Parallel {
		results, errors = h.processBatchParallel(batchRequest.Requests)
	} else {
		results, errors = h.processBatchSequential(batchRequest.Requests)
	}

	response := map[string]interface{}{
		"status":  "completed",
		"results": results,
		"summary": map[string]interface{}{
			"total":     len(batchRequest.Requests),
			"succeeded": len(results),
			"failed":    len(errors),
		},
	}

	if len(errors) > 0 {
		response["errors"] = errors
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *GenerationHandler) GetGenerators(w http.ResponseWriter, r *http.Request) {
	generators := []map[string]interface{}{
		{
			"id":          "statistical",
			"name":        "Statistical Generator",
			"description": "Generates time series using statistical methods",
			"methods":     []string{"gaussian", "ar", "ma", "arma"},
			"parameters": map[string]interface{}{
				"mean": map[string]interface{}{
					"type":        "number",
					"description": "Mean value",
					"default":     0.0,
				},
				"std": map[string]interface{}{
					"type":        "number",
					"description": "Standard deviation",
					"default":     1.0,
				},
			},
		},
		{
			"id":          "arima",
			"name":        "ARIMA Generator",
			"description": "Generates time series using ARIMA models",
			"methods":     []string{"arima", "sarima"},
			"parameters": map[string]interface{}{
				"p": map[string]interface{}{
					"type":        "integer",
					"description": "AR order",
					"default":     1,
				},
				"d": map[string]interface{}{
					"type":        "integer",
					"description": "Differencing degree",
					"default":     0,
				},
				"q": map[string]interface{}{
					"type":        "integer",
					"description": "MA order",
					"default":     1,
				},
			},
		},
		{
			"id":          "timegan",
			"name":        "TimeGAN Generator",
			"description": "Neural network-based generator",
			"methods":     []string{"standard", "conditional"},
			"parameters": map[string]interface{}{
				"sequence_length": map[string]interface{}{
					"type":        "integer",
					"description": "Sequence length",
					"default":     24,
				},
				"hidden_dim": map[string]interface{}{
					"type":        "integer",
					"description": "Hidden dimensions",
					"default":     24,
				},
			},
		},
	}

	response := map[string]interface{}{
		"generators": generators,
		"count":      len(generators),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *GenerationHandler) GetGeneratorDetails(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	generatorID := vars["id"]

	if generatorID == "" {
		http.Error(w, "Generator ID is required", http.StatusBadRequest)
		return
	}

	generator, err := h.generatorFactory.CreateGenerator(generatorID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Generator not found: %v", err), http.StatusNotFound)
		return
	}

	details := map[string]interface{}{
		"id":   generatorID,
		"name": h.getGeneratorName(generatorID),
		"description": h.getGeneratorDescription(generatorID),
		"capabilities": h.getGeneratorCapabilities(generatorID),
		"parameters": h.getGeneratorParameters(generatorID),
		"examples": h.getGeneratorExamples(generatorID),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(details)
}

func (h *GenerationHandler) GetTemplates(w http.ResponseWriter, r *http.Request) {
	templates := []map[string]interface{}{
		{
			"id":          "sensor_temperature",
			"name":        "Temperature Sensor",
			"description": "IoT temperature sensor data",
			"generator":   "statistical",
			"parameters": map[string]interface{}{
				"mean":      22.0,
				"std":       2.0,
				"method":    "gaussian",
				"frequency": "1m",
			},
		},
		{
			"id":          "stock_prices",
			"name":        "Stock Prices",
			"description": "Financial market data",
			"generator":   "arima",
			"parameters": map[string]interface{}{
				"p":         1,
				"d":         1,
				"q":         1,
				"frequency": "1d",
			},
		},
		{
			"id":          "network_traffic",
			"name":        "Network Traffic",
			"description": "Network utilization patterns",
			"generator":   "timegan",
			"parameters": map[string]interface{}{
				"sequence_length": 96,
				"frequency":       "15m",
			},
		},
	}

	response := map[string]interface{}{
		"templates": templates,
		"count":     len(templates),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *GenerationHandler) GenerateFromTemplate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	templateID := vars["id"]

	var customParams map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&customParams); err != nil {
		customParams = make(map[string]interface{})
	}

	template := h.getTemplate(templateID)
	if template == nil {
		http.Error(w, "Template not found", http.StatusNotFound)
		return
	}

	request := models.GenerationRequest{
		GeneratorType: template["generator"].(string),
		Parameters:    h.mergeParameters(template["parameters"].(map[string]interface{}), customParams),
		StartTime:     time.Now(),
		EndTime:       time.Now().Add(24 * time.Hour),
		Frequency:     h.getFrequency(template["parameters"].(map[string]interface{})),
	}

	generator, err := h.generatorFactory.CreateGenerator(request.GeneratorType)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create generator: %v", err), http.StatusInternalServerError)
		return
	}

	timeSeries, err := generator.Generate(r.Context(), &request)
	if err != nil {
		http.Error(w, fmt.Sprintf("Generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":     "success",
		"template":   templateID,
		"timeSeries": timeSeries,
		"metadata": map[string]interface{}{
			"template":     template,
			"parameters":   request.Parameters,
			"generatedAt":  time.Now(),
			"dataPoints":   len(timeSeries.DataPoints),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *GenerationHandler) validateGenerationRequest(request *models.GenerationRequest) error {
	if request.GeneratorType == "" {
		return fmt.Errorf("generator type is required")
	}

	if request.StartTime.IsZero() {
		request.StartTime = time.Now()
	}

	if request.EndTime.IsZero() {
		request.EndTime = request.StartTime.Add(24 * time.Hour)
	}

	if request.EndTime.Before(request.StartTime) {
		return fmt.Errorf("end time must be after start time")
	}

	if request.Frequency == "" {
		request.Frequency = "1m"
	}

	return nil
}

func (h *GenerationHandler) processBatchParallel(requests []models.GenerationRequest) ([]map[string]interface{}, []string) {
	resultChan := make(chan map[string]interface{}, len(requests))
	errorChan := make(chan string, len(requests))

	for i, req := range requests {
		go func(index int, request models.GenerationRequest) {
			generator, err := h.generatorFactory.CreateGenerator(request.GeneratorType)
			if err != nil {
				errorChan <- fmt.Sprintf("Request %d: %v", index, err)
				return
			}

			timeSeries, err := generator.Generate(context.Background(), &request)
			if err != nil {
				errorChan <- fmt.Sprintf("Request %d: %v", index, err)
				return
			}

			result := map[string]interface{}{
				"index":      index,
				"timeSeries": timeSeries,
				"generatedAt": time.Now(),
			}
			resultChan <- result
		}(i, req)
	}

	var results []map[string]interface{}
	var errors []string

	for i := 0; i < len(requests); i++ {
		select {
		case result := <-resultChan:
			results = append(results, result)
		case err := <-errorChan:
			errors = append(errors, err)
		case <-time.After(30 * time.Second):
			errors = append(errors, fmt.Sprintf("Request timeout"))
		}
	}

	return results, errors
}

func (h *GenerationHandler) processBatchSequential(requests []models.GenerationRequest) ([]map[string]interface{}, []string) {
	var results []map[string]interface{}
	var errors []string

	for i, request := range requests {
		generator, err := h.generatorFactory.CreateGenerator(request.GeneratorType)
		if err != nil {
			errors = append(errors, fmt.Sprintf("Request %d: %v", i, err))
			continue
		}

		timeSeries, err := generator.Generate(context.Background(), &request)
		if err != nil {
			errors = append(errors, fmt.Sprintf("Request %d: %v", i, err))
			continue
		}

		result := map[string]interface{}{
			"index":      i,
			"timeSeries": timeSeries,
			"generatedAt": time.Now(),
		}
		results = append(results, result)
	}

	return results, errors
}

func (h *GenerationHandler) getGeneratorName(id string) string {
	names := map[string]string{
		"statistical": "Statistical Generator",
		"arima":       "ARIMA Generator",
		"timegan":     "TimeGAN Generator",
	}
	return names[id]
}

func (h *GenerationHandler) getGeneratorDescription(id string) string {
	descriptions := map[string]string{
		"statistical": "Generates time series using statistical methods and distributions",
		"arima":       "Generates time series using ARIMA and SARIMA models",
		"timegan":     "Generates time series using neural networks and GANs",
	}
	return descriptions[id]
}

func (h *GenerationHandler) getGeneratorCapabilities(id string) []string {
	capabilities := map[string][]string{
		"statistical": {"gaussian", "ar", "ma", "arma", "white_noise"},
		"arima":       {"arima", "sarima", "trend", "seasonality"},
		"timegan":     {"standard", "conditional", "multivariate"},
	}
	return capabilities[id]
}

func (h *GenerationHandler) getGeneratorParameters(id string) map[string]interface{} {
	parameters := map[string]map[string]interface{}{
		"statistical": {
			"mean": map[string]interface{}{"type": "number", "default": 0.0},
			"std":  map[string]interface{}{"type": "number", "default": 1.0},
		},
		"arima": {
			"p": map[string]interface{}{"type": "integer", "default": 1},
			"d": map[string]interface{}{"type": "integer", "default": 0},
			"q": map[string]interface{}{"type": "integer", "default": 1},
		},
		"timegan": {
			"sequence_length": map[string]interface{}{"type": "integer", "default": 24},
			"hidden_dim":      map[string]interface{}{"type": "integer", "default": 24},
		},
	}
	return parameters[id]
}

func (h *GenerationHandler) getGeneratorExamples(id string) []map[string]interface{} {
	examples := map[string][]map[string]interface{}{
		"statistical": {
			{
				"name":        "Gaussian Noise",
				"description": "Simple white noise with normal distribution",
				"parameters":  map[string]interface{}{"method": "gaussian", "mean": 0.0, "std": 1.0},
			},
		},
		"arima": {
			{
				"name":        "Simple ARIMA(1,1,1)",
				"description": "Basic ARIMA model with trend",
				"parameters":  map[string]interface{}{"p": 1, "d": 1, "q": 1},
			},
		},
		"timegan": {
			{
				"name":        "Standard TimeGAN",
				"description": "Default TimeGAN configuration",
				"parameters":  map[string]interface{}{"sequence_length": 24, "hidden_dim": 24},
			},
		},
	}
	return examples[id]
}

func (h *GenerationHandler) getTemplate(id string) map[string]interface{} {
	templates := map[string]map[string]interface{}{
		"sensor_temperature": {
			"generator": "statistical",
			"parameters": map[string]interface{}{
				"mean": 22.0, "std": 2.0, "method": "gaussian", "frequency": "1m",
			},
		},
		"stock_prices": {
			"generator": "arima",
			"parameters": map[string]interface{}{
				"p": 1, "d": 1, "q": 1, "frequency": "1d",
			},
		},
		"network_traffic": {
			"generator": "timegan",
			"parameters": map[string]interface{}{
				"sequence_length": 96, "frequency": "15m",
			},
		},
	}
	return templates[id]
}

func (h *GenerationHandler) mergeParameters(template, custom map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	
	for k, v := range template {
		result[k] = v
	}
	
	for k, v := range custom {
		result[k] = v
	}
	
	return result
}

func (h *GenerationHandler) getFrequency(params map[string]interface{}) string {
	if freq, ok := params["frequency"].(string); ok {
		return freq
	}
	return "1m"
}