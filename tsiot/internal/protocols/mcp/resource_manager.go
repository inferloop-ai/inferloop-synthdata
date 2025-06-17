package mcp

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/inferloop/tsiot/internal/generators"
	"github.com/inferloop/tsiot/pkg/models"
)

type ResourceManager struct {
	resources      map[string]ResourceProvider
	subscriptions  map[string][]Subscription
	mu             sync.RWMutex
	
	generatorFactory *generators.GeneratorFactory
	dataCache        map[string]*CachedData
	cacheMu          sync.RWMutex
}

type ResourceProvider interface {
	GetResource() (interface{}, error)
	GetMimeType() string
}

type Subscription struct {
	ClientID     string
	ResourceURI  string
	SubscribedAt time.Time
	Callback     func(update ResourceUpdate)
}

type ResourceUpdate struct {
	URI         string      `json:"uri"`
	UpdateType  string      `json:"updateType"`
	Data        interface{} `json:"data"`
	Timestamp   time.Time   `json:"timestamp"`
}

type CachedData struct {
	Data      interface{}
	CachedAt  time.Time
	ExpiresAt time.Time
}

func NewResourceManager() *ResourceManager {
	rm := &ResourceManager{
		resources:        make(map[string]ResourceProvider),
		subscriptions:    make(map[string][]Subscription),
		dataCache:        make(map[string]*CachedData),
		generatorFactory: generators.NewGeneratorFactory(),
	}
	
	rm.registerBuiltinResources()
	return rm
}

func (rm *ResourceManager) registerBuiltinResources() {
	rm.RegisterResourceProvider("timeseries://generators", &GeneratorsResource{
		factory: rm.generatorFactory,
	})
	
	rm.RegisterResourceProvider("timeseries://templates", &TemplatesResource{})
	
	rm.RegisterResourceProvider("timeseries://schemas", &SchemasResource{})
	
	rm.RegisterResourceProvider("timeseries://metrics", &MetricsResource{})
	
	rm.RegisterResourceProvider("timeseries://datasets", &DatasetsResource{})
}

func (rm *ResourceManager) RegisterResourceProvider(uri string, provider ResourceProvider) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.resources[uri] = provider
}

func (rm *ResourceManager) GetResource(uri string) (interface{}, error) {
	rm.cacheMu.RLock()
	if cached, ok := rm.dataCache[uri]; ok && time.Now().Before(cached.ExpiresAt) {
		rm.cacheMu.RUnlock()
		return cached.Data, nil
	}
	rm.cacheMu.RUnlock()
	
	rm.mu.RLock()
	provider, ok := rm.resources[uri]
	rm.mu.RUnlock()
	
	if !ok {
		return nil, fmt.Errorf("resource not found: %s", uri)
	}
	
	data, err := provider.GetResource()
	if err != nil {
		return nil, err
	}
	
	rm.cacheMu.Lock()
	rm.dataCache[uri] = &CachedData{
		Data:      data,
		CachedAt:  time.Now(),
		ExpiresAt: time.Now().Add(5 * time.Minute),
	}
	rm.cacheMu.Unlock()
	
	return data, nil
}

func (rm *ResourceManager) Subscribe(clientID, uri string, callback func(ResourceUpdate)) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	
	if _, ok := rm.resources[uri]; !ok {
		return fmt.Errorf("resource not found: %s", uri)
	}
	
	subscription := Subscription{
		ClientID:     clientID,
		ResourceURI:  uri,
		SubscribedAt: time.Now(),
		Callback:     callback,
	}
	
	rm.subscriptions[uri] = append(rm.subscriptions[uri], subscription)
	
	return nil
}

func (rm *ResourceManager) Unsubscribe(clientID, uri string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	
	subs := rm.subscriptions[uri]
	newSubs := make([]Subscription, 0, len(subs))
	
	for _, sub := range subs {
		if sub.ClientID != clientID {
			newSubs = append(newSubs, sub)
		}
	}
	
	rm.subscriptions[uri] = newSubs
}

func (rm *ResourceManager) NotifySubscribers(uri string, update ResourceUpdate) {
	rm.mu.RLock()
	subs := rm.subscriptions[uri]
	rm.mu.RUnlock()
	
	for _, sub := range subs {
		go sub.Callback(update)
	}
}

func (rm *ResourceManager) InvalidateCache(uri string) {
	rm.cacheMu.Lock()
	defer rm.cacheMu.Unlock()
	delete(rm.dataCache, uri)
}

// Built-in Resource Providers

type GeneratorsResource struct {
	factory *generators.GeneratorFactory
}

func (r *GeneratorsResource) GetResource() (interface{}, error) {
	return map[string]interface{}{
		"generators": []map[string]interface{}{
			{
				"id":          "statistical",
				"name":        "Statistical Generator",
				"description": "Generates time series using statistical methods",
				"methods": []string{
					"gaussian",
					"ar",
					"ma",
					"arma",
				},
				"parameters": map[string]interface{}{
					"mean": map[string]interface{}{
						"type":        "number",
						"description": "Mean value for the distribution",
						"default":     0.0,
					},
					"std": map[string]interface{}{
						"type":        "number",
						"description": "Standard deviation",
						"default":     1.0,
					},
					"order": map[string]interface{}{
						"type":        "integer",
						"description": "Order for AR/MA models",
						"default":     1,
					},
				},
			},
			{
				"id":          "arima",
				"name":        "ARIMA Generator",
				"description": "Generates time series using ARIMA models",
				"methods": []string{
					"arima",
					"sarima",
				},
				"parameters": map[string]interface{}{
					"p": map[string]interface{}{
						"type":        "integer",
						"description": "AR order",
						"default":     1,
					},
					"d": map[string]interface{}{
						"type":        "integer",
						"description": "Degree of differencing",
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
				"description": "Generates time series using neural networks",
				"methods": []string{
					"standard",
					"conditional",
				},
				"parameters": map[string]interface{}{
					"sequence_length": map[string]interface{}{
						"type":        "integer",
						"description": "Length of generated sequences",
						"default":     24,
					},
					"hidden_dim": map[string]interface{}{
						"type":        "integer",
						"description": "Hidden dimension size",
						"default":     24,
					},
					"num_layers": map[string]interface{}{
						"type":        "integer",
						"description": "Number of RNN layers",
						"default":     3,
					},
				},
			},
		},
	}, nil
}

func (r *GeneratorsResource) GetMimeType() string {
	return "application/json"
}

type TemplatesResource struct{}

func (r *TemplatesResource) GetResource() (interface{}, error) {
	return map[string]interface{}{
		"templates": []map[string]interface{}{
			{
				"id":          "sensor_temperature",
				"name":        "Temperature Sensor",
				"description": "Template for temperature sensor data",
				"generator":   "statistical",
				"parameters": map[string]interface{}{
					"mean":      22.0,
					"std":       2.0,
					"method":    "gaussian",
					"frequency": "1m",
				},
				"metadata": map[string]interface{}{
					"unit":     "celsius",
					"location": "office",
					"sensor":   "DHT22",
				},
			},
			{
				"id":          "stock_prices",
				"name":        "Stock Price Data",
				"description": "Template for financial stock price data",
				"generator":   "arima",
				"parameters": map[string]interface{}{
					"p":         1,
					"d":         1,
					"q":         1,
					"frequency": "1d",
				},
				"metadata": map[string]interface{}{
					"market":   "NYSE",
					"currency": "USD",
				},
			},
			{
				"id":          "network_traffic",
				"name":        "Network Traffic",
				"description": "Template for network traffic patterns",
				"generator":   "timegan",
				"parameters": map[string]interface{}{
					"sequence_length": 96,
					"hidden_dim":      48,
					"frequency":       "15m",
				},
				"metadata": map[string]interface{}{
					"protocol": "HTTP",
					"unit":     "requests/sec",
				},
			},
		},
	}, nil
}

func (r *TemplatesResource) GetMimeType() string {
	return "application/json"
}

type SchemasResource struct{}

func (r *SchemasResource) GetResource() (interface{}, error) {
	return map[string]interface{}{
		"schemas": map[string]interface{}{
			"TimeSeries": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the time series",
					},
					"dataPoints": map[string]interface{}{
						"type": "array",
						"items": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"timestamp": map[string]interface{}{
									"type":   "string",
									"format": "date-time",
								},
								"value": map[string]interface{}{
									"type": "number",
								},
							},
						},
					},
					"metadata": map[string]interface{}{
						"type":        "object",
						"description": "Additional metadata",
					},
				},
			},
			"GenerationRequest": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"generatorType": map[string]interface{}{
						"type": "string",
						"enum": []string{"statistical", "arima", "timegan"},
					},
					"parameters": map[string]interface{}{
						"type": "object",
					},
					"startTime": map[string]interface{}{
						"type":   "string",
						"format": "date-time",
					},
					"endTime": map[string]interface{}{
						"type":   "string",
						"format": "date-time",
					},
					"frequency": map[string]interface{}{
						"type": "string",
					},
				},
			},
		},
	}, nil
}

func (r *SchemasResource) GetMimeType() string {
	return "application/json"
}

type MetricsResource struct{}

func (r *MetricsResource) GetResource() (interface{}, error) {
	return map[string]interface{}{
		"metrics": map[string]interface{}{
			"generation": map[string]interface{}{
				"total_requests":     1234,
				"successful_requests": 1200,
				"failed_requests":     34,
				"average_duration_ms": 150,
			},
			"validation": map[string]interface{}{
				"total_validations": 567,
				"passed":            520,
				"failed":            47,
			},
			"resources": map[string]interface{}{
				"cpu_usage_percent":    45.2,
				"memory_usage_mb":      512,
				"active_generators":    5,
				"cached_timeseries":    23,
			},
			"timestamp": time.Now().Format(time.RFC3339),
		},
	}, nil
}

func (r *MetricsResource) GetMimeType() string {
	return "application/json"
}

type DatasetsResource struct{}

func (r *DatasetsResource) GetResource() (interface{}, error) {
	return map[string]interface{}{
		"datasets": []map[string]interface{}{
			{
				"id":          "sample_sensor_data",
				"name":        "Sample Sensor Dataset",
				"description": "Pre-generated sensor data for testing",
				"size":        1000,
				"generator":   "statistical",
				"created_at":  "2024-01-15T10:00:00Z",
			},
			{
				"id":          "financial_demo",
				"name":        "Financial Demo Dataset",
				"description": "Stock market simulation data",
				"size":        5000,
				"generator":   "arima",
				"created_at":  "2024-01-20T14:30:00Z",
			},
		},
	}, nil
}

func (r *DatasetsResource) GetMimeType() string {
	return "application/json"
}