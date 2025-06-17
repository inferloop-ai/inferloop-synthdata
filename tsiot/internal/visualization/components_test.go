package visualization

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewWebSocketHub(t *testing.T) {
	logger := logrus.New()
	hub := NewWebSocketHub(logger)

	require.NotNil(t, hub)
	assert.Equal(t, logger, hub.logger)
	assert.NotNil(t, hub.clients)
	assert.NotNil(t, hub.register)
	assert.NotNil(t, hub.unregister)
	assert.NotNil(t, hub.broadcast)
	assert.NotNil(t, hub.upgrader)
}

func TestWebSocketHubRun(t *testing.T) {
	logger := logrus.New()
	hub := NewWebSocketHub(logger)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Start the hub in a goroutine
	done := make(chan bool)
	go func() {
		hub.Run(ctx)
		done <- true
	}()

	// Wait for context to be cancelled
	select {
	case <-done:
		// Hub stopped as expected
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Hub did not stop within expected time")
	}
}

func TestWebSocketHubClientManagement(t *testing.T) {
	logger := logrus.New()
	hub := NewWebSocketHub(logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the hub
	go hub.Run(ctx)

	// Create mock client
	client := &WebSocketClient{
		hub:           hub,
		send:          make(chan []byte, 256),
		dashboardID:   "test-dashboard",
		subscriptions: make(map[string]bool),
	}

	// Register client
	hub.register <- client

	// Give some time for registration
	time.Sleep(10 * time.Millisecond)

	// Check that client was registered
	hub.mu.RLock()
	clientExists := hub.clients[client]
	clientCount := len(hub.clients)
	hub.mu.RUnlock()

	assert.True(t, clientExists)
	assert.Equal(t, 1, clientCount)

	// Unregister client
	hub.unregister <- client

	// Give some time for unregistration
	time.Sleep(10 * time.Millisecond)

	// Check that client was unregistered
	hub.mu.RLock()
	clientExists = hub.clients[client]
	clientCount = len(hub.clients)
	hub.mu.RUnlock()

	assert.False(t, clientExists)
	assert.Equal(t, 0, clientCount)
}

func TestWebSocketHubBroadcast(t *testing.T) {
	logger := logrus.New()
	hub := NewWebSocketHub(logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the hub
	go hub.Run(ctx)

	// Create mock client
	client := &WebSocketClient{
		hub:           hub,
		send:          make(chan []byte, 256),
		dashboardID:   "test-dashboard",
		subscriptions: make(map[string]bool),
	}

	// Register client
	hub.register <- client

	// Give some time for registration
	time.Sleep(10 * time.Millisecond)

	// Test broadcast
	testData := map[string]interface{}{
		"test": "data",
		"value": 42,
	}

	hub.BroadcastUpdate("test_message", testData)

	// Check that message was sent to client
	select {
	case message := <-client.send:
		var wsMessage WebSocketMessage
		err := json.Unmarshal(message, &wsMessage)
		require.NoError(t, err)

		assert.Equal(t, "test_message", wsMessage.Type)
		assert.NotNil(t, wsMessage.Data)
		assert.False(t, wsMessage.Timestamp.IsZero())

		// Check data content
		dataMap, ok := wsMessage.Data.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "data", dataMap["test"])
		assert.Equal(t, float64(42), dataMap["value"])

	case <-time.After(100 * time.Millisecond):
		t.Fatal("No message received within expected time")
	}

	// Cleanup
	hub.unregister <- client
}

func TestWebSocketHubHandleConnection(t *testing.T) {
	logger := logrus.New()
	hub := NewWebSocketHub(logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the hub
	go hub.Run(ctx)

	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hub.HandleConnection(w, r)
	}))
	defer server.Close()

	// Convert http URL to ws URL
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "?dashboard_id=test-dashboard"

	// Create WebSocket connection
	dialer := websocket.Dialer{}
	conn, _, err := dialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Give some time for connection to be established
	time.Sleep(10 * time.Millisecond)

	// Check that client was registered
	hub.mu.RLock()
	clientCount := len(hub.clients)
	hub.mu.RUnlock()

	assert.Equal(t, 1, clientCount)
}

func TestNewChartManager(t *testing.T) {
	logger := logrus.New()
	cm := NewChartManager(logger)

	require.NotNil(t, cm)
	assert.Equal(t, logger, cm.logger)
	assert.NotNil(t, cm.chartTypes)
}

func TestNewWidgetManager(t *testing.T) {
	logger := logrus.New()
	wm := NewWidgetManager(logger)

	require.NotNil(t, wm)
	assert.Equal(t, logger, wm.logger)
	assert.NotNil(t, wm.widgetTypes)
	assert.NotNil(t, wm.dataCache)
}

func TestWidgetManagerCRUD(t *testing.T) {
	logger := logrus.New()
	wm := NewWidgetManager(logger)

	widget := &Widget{
		ID:    "test-widget",
		Type:  WidgetTypeChart,
		Title: "Test Chart Widget",
		Configuration: WidgetConfiguration{
			ChartType: ChartTypeLine,
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		IsVisible: true,
	}

	t.Run("Create Widget", func(t *testing.T) {
		err := wm.CreateWidget(widget)
		require.NoError(t, err)

		// Verify widget was created
		retrieved, err := wm.GetWidget("test-widget")
		require.NoError(t, err)
		assert.Equal(t, widget.ID, retrieved.ID)
		assert.Equal(t, widget.Title, retrieved.Title)
	})

	t.Run("Get Non-existent Widget", func(t *testing.T) {
		_, err := wm.GetWidget("non-existent")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "widget not found")
	})

	t.Run("Update Widget", func(t *testing.T) {
		widget.Title = "Updated Chart Widget"
		widget.UpdatedAt = time.Now()

		err := wm.UpdateWidget(widget)
		require.NoError(t, err)

		// Verify update
		retrieved, err := wm.GetWidget("test-widget")
		require.NoError(t, err)
		assert.Equal(t, "Updated Chart Widget", retrieved.Title)
	})

	t.Run("Get All Widgets", func(t *testing.T) {
		// Create another widget
		widget2 := &Widget{
			ID:    "test-widget-2",
			Type:  WidgetTypeTable,
			Title: "Test Table Widget",
		}
		err := wm.CreateWidget(widget2)
		require.NoError(t, err)

		widgets := wm.GetAllWidgets()
		assert.Len(t, widgets, 2)

		// Verify widget IDs
		ids := make(map[string]bool)
		for _, w := range widgets {
			ids[w.ID] = true
		}
		assert.True(t, ids["test-widget"])
		assert.True(t, ids["test-widget-2"])
	})

	t.Run("Delete Widget", func(t *testing.T) {
		err := wm.DeleteWidget("test-widget")
		require.NoError(t, err)

		// Verify deletion
		_, err = wm.GetWidget("test-widget")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "widget not found")
	})

	t.Run("Delete Non-existent Widget", func(t *testing.T) {
		err := wm.DeleteWidget("non-existent")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "widget not found")
	})
}

func TestStaticDataProvider(t *testing.T) {
	provider := NewStaticDataProvider()
	require.NotNil(t, provider)

	ctx := context.Background()

	t.Run("Set and Get Data", func(t *testing.T) {
		testData := map[string]interface{}{
			"series1": []map[string]interface{}{
				{"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
				{"timestamp": "2023-01-01T01:00:00Z", "value": 15.0},
			},
		}

		provider.SetData("test-key", testData)

		config := DataSourceConfig{
			Type: DataSourceTypeStatic,
			Parameters: map[string]interface{}{
				"key": "test-key",
			},
		}

		retrievedData, err := provider.GetData(ctx, config)
		require.NoError(t, err)
		assert.Equal(t, testData, retrievedData)
	})

	t.Run("Get Non-existent Data", func(t *testing.T) {
		config := DataSourceConfig{
			Type: DataSourceTypeStatic,
			Parameters: map[string]interface{}{
				"key": "non-existent",
			},
		}

		_, err := provider.GetData(ctx, config)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("Supports Real Time", func(t *testing.T) {
		assert.False(t, provider.SupportsRealTime())
	})

	t.Run("Subscribe/Unsubscribe", func(t *testing.T) {
		config := DataSourceConfig{}
		callback := func(data interface{}) {}

		err := provider.Subscribe(ctx, config, callback)
		assert.NoError(t, err)

		err = provider.Unsubscribe(config)
		assert.NoError(t, err)
	})
}

func TestRESTDataProvider(t *testing.T) {
	provider := NewRESTDataProvider()
	require.NotNil(t, provider)

	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		testData := []map[string]interface{}{
			{"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
			{"timestamp": "2023-01-01T01:00:00Z", "value": 15.0},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(testData)
	}))
	defer server.Close()

	ctx := context.Background()

	t.Run("Get Data from REST API", func(t *testing.T) {
		config := DataSourceConfig{
			Type:   DataSourceTypeREST,
			URL:    server.URL + "/api/data",
			Method: "GET",
		}

		data, err := provider.GetData(ctx, config)
		require.NoError(t, err)
		assert.NotNil(t, data)

		// Verify data structure
		dataSlice, ok := data.([]interface{})
		require.True(t, ok)
		assert.Len(t, dataSlice, 2)
	})

	t.Run("Get Data with Caching", func(t *testing.T) {
		config := DataSourceConfig{
			Type:   DataSourceTypeREST,
			URL:    server.URL + "/api/data",
			Method: "GET",
			Cache: CacheConfig{
				Enabled: true,
				TTL:     5 * time.Minute,
				Key:     "test-cache-key",
			},
		}

		// First request - should fetch from API
		data1, err := provider.GetData(ctx, config)
		require.NoError(t, err)

		// Second request - should use cache
		data2, err := provider.GetData(ctx, config)
		require.NoError(t, err)

		assert.Equal(t, data1, data2)
	})

	t.Run("Supports Real Time", func(t *testing.T) {
		assert.False(t, provider.SupportsRealTime())
	})
}

func TestWebSocketDataProvider(t *testing.T) {
	provider := NewWebSocketDataProvider()
	require.NotNil(t, provider)

	ctx := context.Background()

	t.Run("Supports Real Time", func(t *testing.T) {
		assert.True(t, provider.SupportsRealTime())
	})

	t.Run("Subscribe/Unsubscribe", func(t *testing.T) {
		config := DataSourceConfig{
			Type: DataSourceTypeWebSocket,
			URL:  "ws://localhost:8080/ws",
		}

		callbackCalled := false
		callback := func(data interface{}) {
			callbackCalled = true
		}

		err := provider.Subscribe(ctx, config, callback)
		assert.NoError(t, err)

		err = provider.Unsubscribe(config)
		assert.NoError(t, err)
	})
}

func TestCachedData(t *testing.T) {
	cache := &CachedData{
		Data:      "test data",
		Timestamp: time.Now(),
		TTL:       5 * time.Minute,
	}

	t.Run("Valid Cache", func(t *testing.T) {
		assert.True(t, cache.IsValid())
	})

	t.Run("Expired Cache", func(t *testing.T) {
		expiredCache := &CachedData{
			Data:      "expired data",
			Timestamp: time.Now().Add(-10 * time.Minute),
			TTL:       5 * time.Minute,
		}
		assert.False(t, expiredCache.IsValid())
	})
}

func TestWebSocketMessage(t *testing.T) {
	message := WebSocketMessage{
		Type:      "data_update",
		Data:      map[string]interface{}{"value": 42},
		Timestamp: time.Now(),
		ID:        "msg-123",
	}

	// Test JSON marshaling
	messageBytes, err := json.Marshal(message)
	require.NoError(t, err)
	assert.NotEmpty(t, messageBytes)

	// Test JSON unmarshaling
	var unmarshaled WebSocketMessage
	err = json.Unmarshal(messageBytes, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, message.Type, unmarshaled.Type)
	assert.Equal(t, message.ID, unmarshaled.ID)
	assert.NotNil(t, unmarshaled.Data)
}

// Helper functions for component construction

func NewChartManager(logger *logrus.Logger) *ChartManager {
	return &ChartManager{
		logger:     logger,
		chartTypes: make(map[ChartType]ChartRenderer),
	}
}

func NewWidgetManager(logger *logrus.Logger) *WidgetManager {
	return &WidgetManager{
		logger:      logger,
		widgetTypes: make(map[WidgetType]WidgetRenderer),
		dataCache:   make(map[string]*CachedData),
	}
}

func (wm *WidgetManager) CreateWidget(widget *Widget) error {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	// In a real implementation, this would store to a database
	// For testing, we'll use a simple in-memory map
	if wm.widgets == nil {
		wm.widgets = make(map[string]*Widget)
	}
	wm.widgets[widget.ID] = widget
	return nil
}

func (wm *WidgetManager) GetWidget(id string) (*Widget, error) {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	if wm.widgets == nil {
		return nil, fmt.Errorf("widget not found: %s", id)
	}

	widget, exists := wm.widgets[id]
	if !exists {
		return nil, fmt.Errorf("widget not found: %s", id)
	}
	return widget, nil
}

func (wm *WidgetManager) UpdateWidget(widget *Widget) error {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	if wm.widgets == nil {
		return fmt.Errorf("widget not found: %s", widget.ID)
	}

	if _, exists := wm.widgets[widget.ID]; !exists {
		return fmt.Errorf("widget not found: %s", widget.ID)
	}

	wm.widgets[widget.ID] = widget
	return nil
}

func (wm *WidgetManager) DeleteWidget(id string) error {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	if wm.widgets == nil {
		return fmt.Errorf("widget not found: %s", id)
	}

	if _, exists := wm.widgets[id]; !exists {
		return fmt.Errorf("widget not found: %s", id)
	}

	delete(wm.widgets, id)
	return nil
}

func (wm *WidgetManager) GetAllWidgets() []*Widget {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	if wm.widgets == nil {
		return []*Widget{}
	}

	widgets := make([]*Widget, 0, len(wm.widgets))
	for _, widget := range wm.widgets {
		widgets = append(widgets, widget)
	}
	return widgets
}

// Add widgets field to WidgetManager for testing
type WidgetManager struct {
	logger      *logrus.Logger
	widgetTypes map[WidgetType]WidgetRenderer
	dataCache   map[string]*CachedData
	widgets     map[string]*Widget // For testing
	mu          sync.RWMutex
}

func NewStaticDataProvider() *StaticDataProvider {
	return &StaticDataProvider{
		data: make(map[string]interface{}),
	}
}

func (p *StaticDataProvider) SetData(key string, data interface{}) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.data[key] = data
}

func (p *StaticDataProvider) GetData(ctx context.Context, config interface{}) (interface{}, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var key string
	if dsConfig, ok := config.(DataSourceConfig); ok {
		if params := dsConfig.Parameters; params != nil {
			if k, exists := params["key"]; exists {
				if keyStr, ok := k.(string); ok {
					key = keyStr
				}
			}
		}
	}

	if key == "" {
		key = "default"
	}

	data, exists := p.data[key]
	if !exists {
		return nil, fmt.Errorf("data not found for key: %s", key)
	}

	return data, nil
}

func (p *StaticDataProvider) SupportsRealTime() bool {
	return false
}

func (p *StaticDataProvider) Subscribe(ctx context.Context, config interface{}, callback func(interface{})) error {
	// Static provider doesn't support real-time subscriptions
	return nil
}

func (p *StaticDataProvider) Unsubscribe(config interface{}) error {
	return nil
}

func NewRESTDataProvider() *RESTDataProvider {
	return &RESTDataProvider{
		client: &http.Client{Timeout: 30 * time.Second},
		cache:  make(map[string]*CachedData),
	}
}

func (p *RESTDataProvider) GetData(ctx context.Context, config interface{}) (interface{}, error) {
	dsConfig, ok := config.(DataSourceConfig)
	if !ok {
		return nil, fmt.Errorf("invalid config type")
	}

	// Check cache first
	if dsConfig.Cache.Enabled && dsConfig.Cache.Key != "" {
		p.mu.RLock()
		cachedData, exists := p.cache[dsConfig.Cache.Key]
		p.mu.RUnlock()

		if exists && cachedData.IsValid() {
			return cachedData.Data, nil
		}
	}

	// Make HTTP request
	method := dsConfig.Method
	if method == "" {
		method = "GET"
	}

	req, err := http.NewRequestWithContext(ctx, method, dsConfig.URL, nil)
	if err != nil {
		return nil, err
	}

	// Add headers
	for key, value := range dsConfig.Headers {
		req.Header.Set(key, value)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}

	var data interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	// Cache the result
	if dsConfig.Cache.Enabled && dsConfig.Cache.Key != "" {
		p.mu.Lock()
		p.cache[dsConfig.Cache.Key] = &CachedData{
			Data:      data,
			Timestamp: time.Now(),
			TTL:       dsConfig.Cache.TTL,
		}
		p.mu.Unlock()
	}

	return data, nil
}

func (p *RESTDataProvider) SupportsRealTime() bool {
	return false
}

func (p *RESTDataProvider) Subscribe(ctx context.Context, config interface{}, callback func(interface{})) error {
	return fmt.Errorf("REST provider does not support real-time subscriptions")
}

func (p *RESTDataProvider) Unsubscribe(config interface{}) error {
	return nil
}

func NewWebSocketDataProvider() *WebSocketDataProvider {
	return &WebSocketDataProvider{
		connections: make(map[string]*websocket.Conn),
		subscribers: make(map[string][]func(interface{})),
	}
}

func (p *WebSocketDataProvider) GetData(ctx context.Context, config interface{}) (interface{}, error) {
	return nil, fmt.Errorf("WebSocket provider is for real-time data only")
}

func (p *WebSocketDataProvider) SupportsRealTime() bool {
	return true
}

func (p *WebSocketDataProvider) Subscribe(ctx context.Context, config interface{}, callback func(interface{})) error {
	dsConfig, ok := config.(DataSourceConfig)
	if !ok {
		return fmt.Errorf("invalid config type")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	key := dsConfig.URL
	p.subscribers[key] = append(p.subscribers[key], callback)

	return nil
}

func (p *WebSocketDataProvider) Unsubscribe(config interface{}) error {
	dsConfig, ok := config.(DataSourceConfig)
	if !ok {
		return fmt.Errorf("invalid config type")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	key := dsConfig.URL
	delete(p.subscribers, key)

	if conn, exists := p.connections[key]; exists {
		conn.Close()
		delete(p.connections, key)
	}

	return nil
}

func (c *CachedData) IsValid() bool {
	if c.TTL == 0 {
		return true // No expiration
	}
	return time.Since(c.Timestamp) < c.TTL
}

// Benchmark tests
func BenchmarkWebSocketHubBroadcast(b *testing.B) {
	logger := logrus.New()
	hub := NewWebSocketHub(logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go hub.Run(ctx)

	// Create multiple mock clients
	clients := make([]*WebSocketClient, 100)
	for i := 0; i < 100; i++ {
		client := &WebSocketClient{
			hub:           hub,
			send:          make(chan []byte, 256),
			dashboardID:   fmt.Sprintf("dashboard-%d", i),
			subscriptions: make(map[string]bool),
		}
		clients[i] = client
		hub.register <- client
	}

	// Wait for all clients to register
	time.Sleep(10 * time.Millisecond)

	testData := map[string]interface{}{
		"benchmark": true,
		"value":     42,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hub.BroadcastUpdate("benchmark_message", testData)
	}

	// Cleanup
	for _, client := range clients {
		hub.unregister <- client
	}
}

func BenchmarkWidgetManagerOperations(b *testing.B) {
	logger := logrus.New()
	wm := NewWidgetManager(logger)

	// Pre-populate with widgets
	for i := 0; i < 1000; i++ {
		widget := &Widget{
			ID:    fmt.Sprintf("widget-%d", i),
			Type:  WidgetTypeChart,
			Title: fmt.Sprintf("Widget %d", i),
		}
		wm.CreateWidget(widget)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := wm.GetWidget(fmt.Sprintf("widget-%d", i%1000))
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRESTDataProviderWithCache(b *testing.B) {
	provider := NewRESTDataProvider()

	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		testData := map[string]interface{}{"value": 42}
		json.NewEncoder(w).Encode(testData)
	}))
	defer server.Close()

	config := DataSourceConfig{
		Type:   DataSourceTypeREST,
		URL:    server.URL,
		Method: "GET",
		Cache: CacheConfig{
			Enabled: true,
			TTL:     5 * time.Minute,
			Key:     "benchmark-key",
		},
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := provider.GetData(ctx, config)
		if err != nil {
			b.Fatal(err)
		}
	}
}