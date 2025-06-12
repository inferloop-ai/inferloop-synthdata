package visualization

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// WebSocketHub manages WebSocket connections for real-time updates
type WebSocketHub struct {
	logger      *logrus.Logger
	clients     map[*WebSocketClient]bool
	register    chan *WebSocketClient
	unregister  chan *WebSocketClient
	broadcast   chan []byte
	mu          sync.RWMutex
	upgrader    websocket.Upgrader
}

// WebSocketClient represents a WebSocket client connection
type WebSocketClient struct {
	hub        *WebSocketHub
	conn       *websocket.Conn
	send       chan []byte
	dashboardID string
	subscriptions map[string]bool
}

// WebSocketMessage represents a WebSocket message
type WebSocketMessage struct {
	Type      string                 `json:"type"`
	Data      interface{}            `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
	ID        string                 `json:"id,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// ChartManager manages chart rendering and data
type ChartManager struct {
	logger     *logrus.Logger
	chartTypes map[ChartType]ChartRenderer
	mu         sync.RWMutex
}

// ChartRenderer interface for rendering charts
type ChartRenderer interface {
	Render(data interface{}, options ChartOptions) (interface{}, error)
	GetDefaultOptions() ChartOptions
	ValidateData(data interface{}) error
}

// WidgetManager manages widget lifecycle and rendering
type WidgetManager struct {
	logger        *logrus.Logger
	widgetTypes   map[WidgetType]WidgetRenderer
	dataCache     map[string]*CachedData
	mu            sync.RWMutex
}

// WidgetRenderer interface for rendering widgets
type WidgetRenderer interface {
	Render(widget *Widget, data interface{}) (interface{}, error)
	GetRequiredFields() []string
	ValidateConfiguration(config WidgetConfiguration) error
}

// CachedData represents cached widget data
type CachedData struct {
	Data      interface{}
	Timestamp time.Time
	TTL       time.Duration
}

// StaticDataProvider provides static data
type StaticDataProvider struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

// RESTDataProvider provides data from REST APIs
type RESTDataProvider struct {
	client *http.Client
	cache  map[string]*CachedData
	mu     sync.RWMutex
}

// WebSocketDataProvider provides real-time data via WebSocket
type WebSocketDataProvider struct {
	connections map[string]*websocket.Conn
	subscribers map[string][]func(interface{})
	mu          sync.RWMutex
}

// NewWebSocketHub creates a new WebSocket hub
func NewWebSocketHub(logger *logrus.Logger) *WebSocketHub {
	return &WebSocketHub{
		logger:     logger,
		clients:    make(map[*WebSocketClient]bool),
		register:   make(chan *WebSocketClient),
		unregister: make(chan *WebSocketClient),
		broadcast:  make(chan []byte),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins in development
			},
		},
	}
}

// Run starts the WebSocket hub
func (hub *WebSocketHub) Run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case client := <-hub.register:
			hub.mu.Lock()
			hub.clients[client] = true
			hub.mu.Unlock()
			hub.logger.WithField("client_count", len(hub.clients)).Info("WebSocket client connected")
		case client := <-hub.unregister:
			hub.mu.Lock()
			if _, ok := hub.clients[client]; ok {
				delete(hub.clients, client)
				close(client.send)
			}
			hub.mu.Unlock()
			hub.logger.WithField("client_count", len(hub.clients)).Info("WebSocket client disconnected")
		case message := <-hub.broadcast:
			hub.mu.RLock()
			for client := range hub.clients {
				select {
				case client.send <- message:
				default:
					delete(hub.clients, client)
					close(client.send)
				}
			}
			hub.mu.RUnlock()
		}
	}
}

// HandleConnection handles new WebSocket connections
func (hub *WebSocketHub) HandleConnection(w http.ResponseWriter, r *http.Request) {
	conn, err := hub.upgrader.Upgrade(w, r, nil)
	if err != nil {
		hub.logger.WithError(err).Error("WebSocket upgrade failed")
		return
	}

	dashboardID := r.URL.Query().Get("dashboard_id")
	client := &WebSocketClient{
		hub:           hub,
		conn:          conn,
		send:          make(chan []byte, 256),
		dashboardID:   dashboardID,
		subscriptions: make(map[string]bool),
	}

	hub.register <- client

	go client.writePump()
	go client.readPump()
}

// BroadcastUpdate broadcasts an update to all connected clients
func (hub *WebSocketHub) BroadcastUpdate(messageType string, data interface{}) {
	message := WebSocketMessage{
		Type:      messageType,
		Data:      data,
		Timestamp: time.Now(),
	}

	messageBytes, err := json.Marshal(message)
	if err != nil {
		hub.logger.WithError(err).Error("Failed to marshal WebSocket message")
		return
	}

	hub.broadcast <- messageBytes
}

func (client *WebSocketClient) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		client.conn.Close()
	}()

	for {
		select {
		case message, ok := <-client.send:
			client.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				client.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := client.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			if err := w.Close(); err != nil {
				return
			}
		case <-ticker.C:
			client.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (client *WebSocketClient) readPump() {
	defer func() {
		client.hub.unregister <- client
		client.conn.Close()
	}()

	client.conn.SetReadLimit(512)
	client.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	client.conn.SetPongHandler(func(string) error {
		client.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, message, err := client.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				client.hub.logger.WithError(err).Error("WebSocket error")
			}
			break
		}

		// Handle incoming messages (subscriptions, etc.)
		var wsMessage WebSocketMessage
		if err := json.Unmarshal(message, &wsMessage); err != nil {
			client.hub.logger.WithError(err).Error("Failed to unmarshal WebSocket message")
			continue
		}

		client.handleMessage(wsMessage)
	}
}

func (client *WebSocketClient) handleMessage(message WebSocketMessage) {
	switch message.Type {
	case "subscribe":
		if dataID, ok := message.Data.(string); ok {
			client.subscriptions[dataID] = true
		}
	case "unsubscribe":
		if dataID, ok := message.Data.(string); ok {
			delete(client.subscriptions, dataID)
		}
	}
}

// NewChartManager creates a new chart manager
func NewChartManager(logger *logrus.Logger) *ChartManager {
	cm := &ChartManager{
		logger:     logger,
		chartTypes: make(map[ChartType]ChartRenderer),
	}

	// Register default chart renderers
	cm.registerDefaultRenderers()

	return cm
}

// RenderChart renders a chart with given data and options
func (cm *ChartManager) RenderChart(chartType ChartType, data interface{}, options ChartOptions) (interface{}, error) {
	cm.mu.RLock()
	renderer, exists := cm.chartTypes[chartType]
	cm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unsupported chart type: %s", chartType)
	}

	return renderer.Render(data, options)
}

// RegisterRenderer registers a chart renderer
func (cm *ChartManager) RegisterRenderer(chartType ChartType, renderer ChartRenderer) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.chartTypes[chartType] = renderer
	cm.logger.WithField("chart_type", chartType).Info("Registered chart renderer")
}

func (cm *ChartManager) registerDefaultRenderers() {
	cm.RegisterRenderer(ChartTypeLine, NewLineChartRenderer())
	cm.RegisterRenderer(ChartTypeBar, NewBarChartRenderer())
	cm.RegisterRenderer(ChartTypePie, NewPieChartRenderer())
	cm.RegisterRenderer(ChartTypeArea, NewAreaChartRenderer())
}

// NewWidgetManager creates a new widget manager
func NewWidgetManager(logger *logrus.Logger) *WidgetManager {
	wm := &WidgetManager{
		logger:      logger,
		widgetTypes: make(map[WidgetType]WidgetRenderer),
		dataCache:   make(map[string]*CachedData),
	}

	// Register default widget renderers
	wm.registerDefaultRenderers()

	return wm
}

// RenderWidget renders a widget with given data
func (wm *WidgetManager) RenderWidget(widget *Widget, data interface{}) (interface{}, error) {
	wm.mu.RLock()
	renderer, exists := wm.widgetTypes[widget.Type]
	wm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unsupported widget type: %s", widget.Type)
	}

	return renderer.Render(widget, data)
}

// CacheData caches widget data
func (wm *WidgetManager) CacheData(key string, data interface{}, ttl time.Duration) {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	wm.dataCache[key] = &CachedData{
		Data:      data,
		Timestamp: time.Now(),
		TTL:       ttl,
	}
}

// GetCachedData retrieves cached data
func (wm *WidgetManager) GetCachedData(key string) (interface{}, bool) {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	cached, exists := wm.dataCache[key]
	if !exists {
		return nil, false
	}

	if time.Since(cached.Timestamp) > cached.TTL {
		delete(wm.dataCache, key)
		return nil, false
	}

	return cached.Data, true
}

func (wm *WidgetManager) registerDefaultRenderers() {
	wm.RegisterRenderer(WidgetTypeChart, NewChartWidgetRenderer())
	wm.RegisterRenderer(WidgetTypeTable, NewTableWidgetRenderer())
	wm.RegisterRenderer(WidgetTypeMetric, NewMetricWidgetRenderer())
	wm.RegisterRenderer(WidgetTypeGauge, NewGaugeWidgetRenderer())
}

// RegisterRenderer registers a widget renderer
func (wm *WidgetManager) RegisterRenderer(widgetType WidgetType, renderer WidgetRenderer) {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	wm.widgetTypes[widgetType] = renderer
}

// Data Provider Implementations

// NewStaticDataProvider creates a new static data provider
func NewStaticDataProvider() *StaticDataProvider {
	return &StaticDataProvider{
		data: make(map[string]interface{}),
	}
}

// GetData returns static data
func (sdp *StaticDataProvider) GetData(ctx context.Context, config DataSourceConfig) (interface{}, error) {
	sdp.mu.RLock()
	defer sdp.mu.RUnlock()

	// Return mock time series data
	mockData := []*models.TimeSeries{
		{
			ID:         "static_1",
			Name:       "Static Series 1",
			SensorType: "temperature",
			DataPoints: []models.DataPoint{
				{Timestamp: time.Now().Add(-10 * time.Minute), Value: 20.5, Quality: 1.0},
				{Timestamp: time.Now().Add(-8 * time.Minute), Value: 21.2, Quality: 1.0},
				{Timestamp: time.Now().Add(-6 * time.Minute), Value: 22.1, Quality: 1.0},
				{Timestamp: time.Now().Add(-4 * time.Minute), Value: 21.8, Quality: 1.0},
				{Timestamp: time.Now().Add(-2 * time.Minute), Value: 23.0, Quality: 1.0},
			},
		},
	}

	return mockData, nil
}

// SupportsRealTime returns false for static data
func (sdp *StaticDataProvider) SupportsRealTime() bool {
	return false
}

// Subscribe is not supported for static data
func (sdp *StaticDataProvider) Subscribe(ctx context.Context, config DataSourceConfig, callback func(interface{})) error {
	return fmt.Errorf("static data provider does not support subscriptions")
}

// Unsubscribe is not supported for static data
func (sdp *StaticDataProvider) Unsubscribe(config DataSourceConfig) error {
	return fmt.Errorf("static data provider does not support subscriptions")
}

// NewRESTDataProvider creates a new REST data provider
func NewRESTDataProvider() *RESTDataProvider {
	return &RESTDataProvider{
		client: &http.Client{Timeout: 30 * time.Second},
		cache:  make(map[string]*CachedData),
	}
}

// GetData fetches data from REST API
func (rdp *RESTDataProvider) GetData(ctx context.Context, config DataSourceConfig) (interface{}, error) {
	// Check cache first
	if config.Cache.Enabled {
		rdp.mu.RLock()
		cached, exists := rdp.cache[config.Cache.Key]
		rdp.mu.RUnlock()

		if exists && time.Since(cached.Timestamp) < config.Cache.TTL {
			return cached.Data, nil
		}
	}

	// Make HTTP request
	req, err := http.NewRequestWithContext(ctx, config.Method, config.URL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	for key, value := range config.Headers {
		req.Header.Set(key, value)
	}

	resp, err := rdp.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP request failed with status: %d", resp.StatusCode)
	}

	var data interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Cache the data
	if config.Cache.Enabled {
		rdp.mu.Lock()
		rdp.cache[config.Cache.Key] = &CachedData{
			Data:      data,
			Timestamp: time.Now(),
			TTL:       config.Cache.TTL,
		}
		rdp.mu.Unlock()
	}

	return data, nil
}

// SupportsRealTime returns false for REST data (polling only)
func (rdp *RESTDataProvider) SupportsRealTime() bool {
	return false
}

// Subscribe is not supported for REST data provider
func (rdp *RESTDataProvider) Subscribe(ctx context.Context, config DataSourceConfig, callback func(interface{})) error {
	return fmt.Errorf("REST data provider does not support subscriptions")
}

// Unsubscribe is not supported for REST data provider
func (rdp *RESTDataProvider) Unsubscribe(config DataSourceConfig) error {
	return fmt.Errorf("REST data provider does not support subscriptions")
}

// NewWebSocketDataProvider creates a new WebSocket data provider
func NewWebSocketDataProvider() *WebSocketDataProvider {
	return &WebSocketDataProvider{
		connections: make(map[string]*websocket.Conn),
		subscribers: make(map[string][]func(interface{})),
	}
}

// GetData is not supported for WebSocket data provider (real-time only)
func (wdp *WebSocketDataProvider) GetData(ctx context.Context, config DataSourceConfig) (interface{}, error) {
	return nil, fmt.Errorf("WebSocket data provider only supports real-time subscriptions")
}

// SupportsRealTime returns true for WebSocket data
func (wdp *WebSocketDataProvider) SupportsRealTime() bool {
	return true
}

// Subscribe subscribes to real-time data via WebSocket
func (wdp *WebSocketDataProvider) Subscribe(ctx context.Context, config DataSourceConfig, callback func(interface{})) error {
	wdp.mu.Lock()
	defer wdp.mu.Unlock()

	// Add callback to subscribers
	key := config.URL
	wdp.subscribers[key] = append(wdp.subscribers[key], callback)

	// Connect to WebSocket if not already connected
	if _, exists := wdp.connections[key]; !exists {
		conn, _, err := websocket.DefaultDialer.Dial(config.URL, nil)
		if err != nil {
			return fmt.Errorf("WebSocket connection failed: %w", err)
		}

		wdp.connections[key] = conn

		// Start reading messages
		go wdp.readMessages(key, conn)
	}

	return nil
}

// Unsubscribe unsubscribes from real-time data
func (wdp *WebSocketDataProvider) Unsubscribe(config DataSourceConfig) error {
	wdp.mu.Lock()
	defer wdp.mu.Unlock()

	key := config.URL
	delete(wdp.subscribers, key)

	// Close connection if no more subscribers
	if len(wdp.subscribers[key]) == 0 {
		if conn, exists := wdp.connections[key]; exists {
			conn.Close()
			delete(wdp.connections, key)
		}
	}

	return nil
}

func (wdp *WebSocketDataProvider) readMessages(key string, conn *websocket.Conn) {
	defer conn.Close()

	for {
		var data interface{}
		err := conn.ReadJSON(&data)
		if err != nil {
			break
		}

		wdp.mu.RLock()
		callbacks := wdp.subscribers[key]
		wdp.mu.RUnlock()

		for _, callback := range callbacks {
			go callback(data)
		}
	}
}

// Chart Renderer Implementations

// LineChartRenderer renders line charts
type LineChartRenderer struct{}

// NewLineChartRenderer creates a new line chart renderer
func NewLineChartRenderer() *LineChartRenderer {
	return &LineChartRenderer{}
}

// Render renders a line chart
func (lcr *LineChartRenderer) Render(data interface{}, options ChartOptions) (interface{}, error) {
	// Convert data to chart format
	chartData := map[string]interface{}{
		"type": "line",
		"data": data,
		"options": map[string]interface{}{
			"responsive": options.Responsive,
			"scales": map[string]interface{}{
				"x": map[string]interface{}{
					"title": map[string]interface{}{
						"display": true,
						"text":    options.XAxis.Title,
					},
				},
				"y": map[string]interface{}{
					"title": map[string]interface{}{
						"display": true,
						"text":    options.YAxis.Title,
					},
				},
			},
		},
	}

	return chartData, nil
}

// GetDefaultOptions returns default options for line charts
func (lcr *LineChartRenderer) GetDefaultOptions() ChartOptions {
	return ChartOptions{
		Responsive: true,
		ShowGrid:   true,
		ShowLabels: true,
		Colors:     []string{"#1976d2", "#4caf50", "#ff9800"},
	}
}

// ValidateData validates chart data
func (lcr *LineChartRenderer) ValidateData(data interface{}) error {
	// Basic validation
	if data == nil {
		return fmt.Errorf("chart data cannot be nil")
	}
	return nil
}

// BarChartRenderer renders bar charts
type BarChartRenderer struct{}

func NewBarChartRenderer() *BarChartRenderer {
	return &BarChartRenderer{}
}

func (bcr *BarChartRenderer) Render(data interface{}, options ChartOptions) (interface{}, error) {
	chartData := map[string]interface{}{
		"type":    "bar",
		"data":    data,
		"options": options,
	}
	return chartData, nil
}

func (bcr *BarChartRenderer) GetDefaultOptions() ChartOptions {
	return ChartOptions{
		Responsive: true,
		ShowGrid:   true,
		ShowLabels: true,
	}
}

func (bcr *BarChartRenderer) ValidateData(data interface{}) error {
	return nil
}

// PieChartRenderer renders pie charts
type PieChartRenderer struct{}

func NewPieChartRenderer() *PieChartRenderer {
	return &PieChartRenderer{}
}

func (pcr *PieChartRenderer) Render(data interface{}, options ChartOptions) (interface{}, error) {
	chartData := map[string]interface{}{
		"type":    "pie",
		"data":    data,
		"options": options,
	}
	return chartData, nil
}

func (pcr *PieChartRenderer) GetDefaultOptions() ChartOptions {
	return ChartOptions{
		Responsive: true,
		Legend: LegendConfig{
			Show:     true,
			Position: "right",
		},
	}
}

func (pcr *PieChartRenderer) ValidateData(data interface{}) error {
	return nil
}

// AreaChartRenderer renders area charts
type AreaChartRenderer struct{}

func NewAreaChartRenderer() *AreaChartRenderer {
	return &AreaChartRenderer{}
}

func (acr *AreaChartRenderer) Render(data interface{}, options ChartOptions) (interface{}, error) {
	chartData := map[string]interface{}{
		"type":    "area",
		"data":    data,
		"options": options,
	}
	return chartData, nil
}

func (acr *AreaChartRenderer) GetDefaultOptions() ChartOptions {
	return ChartOptions{
		Responsive: true,
		FillArea:   true,
		ShowGrid:   true,
	}
}

func (acr *AreaChartRenderer) ValidateData(data interface{}) error {
	return nil
}

// Widget Renderer Implementations

// ChartWidgetRenderer renders chart widgets
type ChartWidgetRenderer struct{}

func NewChartWidgetRenderer() *ChartWidgetRenderer {
	return &ChartWidgetRenderer{}
}

func (cwr *ChartWidgetRenderer) Render(widget *Widget, data interface{}) (interface{}, error) {
	return map[string]interface{}{
		"widget_id": widget.ID,
		"type":      "chart",
		"data":      data,
		"config":    widget.Configuration.ChartOptions,
	}, nil
}

func (cwr *ChartWidgetRenderer) GetRequiredFields() []string {
	return []string{"chart_type", "data"}
}

func (cwr *ChartWidgetRenderer) ValidateConfiguration(config WidgetConfiguration) error {
	return nil
}

// TableWidgetRenderer renders table widgets
type TableWidgetRenderer struct{}

func NewTableWidgetRenderer() *TableWidgetRenderer {
	return &TableWidgetRenderer{}
}

func (twr *TableWidgetRenderer) Render(widget *Widget, data interface{}) (interface{}, error) {
	return map[string]interface{}{
		"widget_id": widget.ID,
		"type":      "table",
		"data":      data,
		"config":    widget.Configuration.TableOptions,
	}, nil
}

func (twr *TableWidgetRenderer) GetRequiredFields() []string {
	return []string{"columns", "data"}
}

func (twr *TableWidgetRenderer) ValidateConfiguration(config WidgetConfiguration) error {
	return nil
}

// MetricWidgetRenderer renders metric widgets
type MetricWidgetRenderer struct{}

func NewMetricWidgetRenderer() *MetricWidgetRenderer {
	return &MetricWidgetRenderer{}
}

func (mwr *MetricWidgetRenderer) Render(widget *Widget, data interface{}) (interface{}, error) {
	return map[string]interface{}{
		"widget_id": widget.ID,
		"type":      "metric",
		"data":      data,
		"config":    widget.Configuration.MetricOptions,
	}, nil
}

func (mwr *MetricWidgetRenderer) GetRequiredFields() []string {
	return []string{"value"}
}

func (mwr *MetricWidgetRenderer) ValidateConfiguration(config WidgetConfiguration) error {
	return nil
}

// GaugeWidgetRenderer renders gauge widgets
type GaugeWidgetRenderer struct{}

func NewGaugeWidgetRenderer() *GaugeWidgetRenderer {
	return &GaugeWidgetRenderer{}
}

func (gwr *GaugeWidgetRenderer) Render(widget *Widget, data interface{}) (interface{}, error) {
	return map[string]interface{}{
		"widget_id": widget.ID,
		"type":      "gauge",
		"data":      data,
		"config":    widget.Configuration,
	}, nil
}

func (gwr *GaugeWidgetRenderer) GetRequiredFields() []string {
	return []string{"value", "min", "max"}
}

func (gwr *GaugeWidgetRenderer) ValidateConfiguration(config WidgetConfiguration) error {
	return nil
}