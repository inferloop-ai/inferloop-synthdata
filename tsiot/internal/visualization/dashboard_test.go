package visualization

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inferloop/tsiot/pkg/models"
)

func TestNewDashboardManager(t *testing.T) {
	config := &DashboardConfig{
		Enabled: true,
		Port:    8080,
		Host:    "localhost",
	}
	logger := logrus.New()

	dm, err := NewDashboardManager(config, logger)
	require.NoError(t, err)
	require.NotNil(t, dm)

	assert.Equal(t, config, dm.config)
	assert.Equal(t, logger, dm.logger)
	assert.NotNil(t, dm.webSocketHub)
	assert.NotNil(t, dm.chartManager)
	assert.NotNil(t, dm.widgetManager)
	assert.NotNil(t, dm.dataProviders)
	assert.NotNil(t, dm.dashboards)
	assert.NotNil(t, dm.themes)
}

func TestNewDashboardManagerDefaults(t *testing.T) {
	dm, err := NewDashboardManager(nil, nil)
	require.NoError(t, err)
	require.NotNil(t, dm)

	// Check default config was applied
	assert.True(t, dm.config.Enabled)
	assert.Equal(t, 8080, dm.config.Port)
	assert.Equal(t, "localhost", dm.config.Host)
	assert.NotNil(t, dm.logger)
}

func TestDashboardManagerCreateDashboard(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	dashboard := &Dashboard{
		ID:          "test-dashboard",
		Name:        "Test Dashboard",
		Description: "Test dashboard for unit tests",
		Layout: &DashboardLayout{
			Type:    LayoutTypeGrid,
			Columns: 12,
			Rows:    8,
		},
		Widgets: []*Widget{
			{
				ID:    "widget-1",
				Type:  WidgetTypeChart,
				Title: "Test Chart",
			},
		},
		Theme:    "light",
		IsPublic: true,
		Owner:    "test-user",
	}

	err = dm.CreateDashboard(dashboard)
	require.NoError(t, err)

	// Verify dashboard was created
	retrieved, err := dm.GetDashboard("test-dashboard")
	require.NoError(t, err)
	assert.Equal(t, dashboard.ID, retrieved.ID)
	assert.Equal(t, dashboard.Name, retrieved.Name)
	assert.False(t, retrieved.CreatedAt.IsZero())
	assert.False(t, retrieved.UpdatedAt.IsZero())
}

func TestDashboardManagerCreateDashboardValidation(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	t.Run("Missing ID", func(t *testing.T) {
		dashboard := &Dashboard{
			Name: "Test Dashboard",
		}
		err := dm.CreateDashboard(dashboard)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "dashboard ID is required")
	})

	t.Run("Missing Name", func(t *testing.T) {
		dashboard := &Dashboard{
			ID: "test-dashboard",
		}
		err := dm.CreateDashboard(dashboard)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "dashboard name is required")
	})

	t.Run("Too Many Widgets", func(t *testing.T) {
		config := &DashboardConfig{
			MaxWidgetsPerDashboard: 2,
		}
		dm, err := NewDashboardManager(config, logrus.New())
		require.NoError(t, err)

		dashboard := &Dashboard{
			ID:   "test-dashboard",
			Name: "Test Dashboard",
			Widgets: []*Widget{
				{ID: "w1", Type: WidgetTypeChart},
				{ID: "w2", Type: WidgetTypeTable},
				{ID: "w3", Type: WidgetTypeMetric},
			},
		}
		err = dm.CreateDashboard(dashboard)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "too many widgets")
	})
}

func TestDashboardManagerGetDashboard(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	// Test getting non-existent dashboard
	_, err = dm.GetDashboard("non-existent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dashboard not found")

	// Create and get dashboard
	dashboard := createTestDashboard()
	err = dm.CreateDashboard(dashboard)
	require.NoError(t, err)

	retrieved, err := dm.GetDashboard(dashboard.ID)
	require.NoError(t, err)
	assert.Equal(t, dashboard.ID, retrieved.ID)
	assert.Equal(t, dashboard.Name, retrieved.Name)
}

func TestDashboardManagerUpdateDashboard(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	// Test updating non-existent dashboard
	dashboard := createTestDashboard()
	err = dm.UpdateDashboard(dashboard)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dashboard not found")

	// Create dashboard first
	err = dm.CreateDashboard(dashboard)
	require.NoError(t, err)

	// Update dashboard
	originalCreatedAt := dashboard.CreatedAt
	dashboard.Name = "Updated Dashboard"
	dashboard.Description = "Updated description"

	err = dm.UpdateDashboard(dashboard)
	require.NoError(t, err)

	// Verify update
	retrieved, err := dm.GetDashboard(dashboard.ID)
	require.NoError(t, err)
	assert.Equal(t, "Updated Dashboard", retrieved.Name)
	assert.Equal(t, "Updated description", retrieved.Description)
	assert.Equal(t, originalCreatedAt, retrieved.CreatedAt)
	assert.True(t, retrieved.UpdatedAt.After(originalCreatedAt))
}

func TestDashboardManagerDeleteDashboard(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	// Test deleting non-existent dashboard
	err = dm.DeleteDashboard("non-existent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dashboard not found")

	// Create and delete dashboard
	dashboard := createTestDashboard()
	err = dm.CreateDashboard(dashboard)
	require.NoError(t, err)

	err = dm.DeleteDashboard(dashboard.ID)
	require.NoError(t, err)

	// Verify deletion
	_, err = dm.GetDashboard(dashboard.ID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dashboard not found")
}

func TestDashboardManagerListDashboards(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	// Initially empty
	dashboards := dm.ListDashboards()
	assert.Empty(t, dashboards)

	// Create multiple dashboards
	dashboard1 := createTestDashboard()
	dashboard1.ID = "dashboard-1"
	dashboard1.Name = "Dashboard 1"

	dashboard2 := createTestDashboard()
	dashboard2.ID = "dashboard-2"
	dashboard2.Name = "Dashboard 2"

	err = dm.CreateDashboard(dashboard1)
	require.NoError(t, err)
	err = dm.CreateDashboard(dashboard2)
	require.NoError(t, err)

	// List dashboards
	dashboards = dm.ListDashboards()
	assert.Len(t, dashboards, 2)

	// Verify dashboard IDs are present
	ids := make(map[string]bool)
	for _, d := range dashboards {
		ids[d.ID] = true
	}
	assert.True(t, ids["dashboard-1"])
	assert.True(t, ids["dashboard-2"])
}

func TestDashboardManagerRegisterDataProvider(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	// Register custom data provider
	provider := &MockDataProvider{}
	dm.RegisterDataProvider("mock", provider)

	// Verify provider was registered
	dm.mu.RLock()
	registeredProvider, exists := dm.dataProviders["mock"]
	dm.mu.RUnlock()

	assert.True(t, exists)
	assert.Equal(t, provider, registeredProvider)
}

func TestDashboardManagerHTTPHandlers(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	t.Run("GET /api/dashboards", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/dashboards", nil)
		w := httptest.NewRecorder()

		dm.handleDashboards(w, req)

		assert.Equal(t, http.StatusOK, w.Code)
		assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

		var dashboards []Dashboard
		err := json.NewDecoder(w.Body).Decode(&dashboards)
		require.NoError(t, err)
		assert.Empty(t, dashboards)
	})

	t.Run("POST /api/dashboards", func(t *testing.T) {
		dashboard := createTestDashboard()
		dashboardJSON, err := json.Marshal(dashboard)
		require.NoError(t, err)

		req := httptest.NewRequest("POST", "/api/dashboards", strings.NewReader(string(dashboardJSON)))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		dm.handleDashboards(w, req)

		assert.Equal(t, http.StatusOK, w.Code)

		var responseDashboard Dashboard
		err = json.NewDecoder(w.Body).Decode(&responseDashboard)
		require.NoError(t, err)
		assert.Equal(t, dashboard.ID, responseDashboard.ID)
		assert.Equal(t, dashboard.Name, responseDashboard.Name)
	})

	t.Run("Invalid Method", func(t *testing.T) {
		req := httptest.NewRequest("PATCH", "/api/dashboards", nil)
		w := httptest.NewRecorder()

		dm.handleDashboards(w, req)

		assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	})
}

func TestDashboardManagerDataHandling(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	// Register mock data provider
	mockProvider := &MockDataProvider{
		data: []interface{}{
			map[string]interface{}{"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
			map[string]interface{}{"timestamp": "2023-01-01T01:00:00Z", "value": 15.0},
		},
	}
	dm.RegisterDataProvider("mock", mockProvider)

	t.Run("Fetch Data", func(t *testing.T) {
		request := DataRequest{
			Source:   "mock",
			WidgetID: "test-widget",
		}

		data, err := dm.fetchData(context.Background(), request)
		require.NoError(t, err)
		assert.Len(t, data, 2)
	})

	t.Run("Fetch Data - Unknown Source", func(t *testing.T) {
		request := DataRequest{
			Source: "unknown",
		}

		_, err := dm.fetchData(context.Background(), request)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unknown data source")
	})

	t.Run("Apply Filters", func(t *testing.T) {
		data := []interface{}{
			map[string]interface{}{"category": "A", "value": 10.0},
			map[string]interface{}{"category": "B", "value": 20.0},
			map[string]interface{}{"category": "A", "value": 30.0},
		}

		filters := map[string]interface{}{"category": "A"}
		filtered := dm.applyDataFilters(data, filters)

		assert.Len(t, filtered, 2)
		for _, item := range filtered {
			point := item.(map[string]interface{})
			assert.Equal(t, "A", point["category"])
		}
	})
}

func TestDashboardManagerDataTransformations(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	t.Run("Transform for Time Series Chart", func(t *testing.T) {
		data := []interface{}{
			map[string]interface{}{"timestamp": "2023-01-01T00:00:00Z", "value": 10.0, "series": "temp"},
			map[string]interface{}{"timestamp": "2023-01-01T01:00:00Z", "value": 15.0, "series": "temp"},
		}

		widget := &Widget{Type: WidgetTypeChart}
		result, err := dm.transformForTimeSeriesChart(data, widget)
		require.NoError(t, err)

		resultMap, ok := result.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "time_series", resultMap["type"])
		assert.Contains(t, resultMap, "series")
	})

	t.Run("Transform for Bar Chart", func(t *testing.T) {
		data := []interface{}{
			map[string]interface{}{"category": "A", "value": 10.0},
			map[string]interface{}{"category": "B", "value": 20.0},
		}

		widget := &Widget{Type: WidgetTypeChart}
		result, err := dm.transformForBarChart(data, widget)
		require.NoError(t, err)

		resultMap, ok := result.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "bar_chart", resultMap["type"])
		assert.Contains(t, resultMap, "categories")
		assert.Contains(t, resultMap, "values")
	})

	t.Run("Transform for Pie Chart", func(t *testing.T) {
		data := []interface{}{
			map[string]interface{}{"label": "A", "value": 30.0, "color": "#ff0000"},
			map[string]interface{}{"label": "B", "value": 70.0, "color": "#00ff00"},
		}

		widget := &Widget{Type: WidgetTypePie}
		result, err := dm.transformForPieChart(data, widget)
		require.NoError(t, err)

		resultMap, ok := result.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "pie_chart", resultMap["type"])
		assert.Contains(t, resultMap, "segments")
	})

	t.Run("Transform for Metric", func(t *testing.T) {
		data := []interface{}{
			map[string]interface{}{"value": 42.0},
		}

		widget := &Widget{Type: WidgetTypeMetric}
		result, err := dm.transformForMetric(data, widget)
		require.NoError(t, err)

		resultMap, ok := result.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "metric", resultMap["type"])
		assert.Equal(t, 42.0, resultMap["value"])
	})

	t.Run("Transform for Table", func(t *testing.T) {
		data := []interface{}{
			map[string]interface{}{"name": "John", "age": 30, "city": "NYC"},
			map[string]interface{}{"name": "Jane", "age": 25, "city": "LA"},
		}

		widget := &Widget{Type: WidgetTypeTable}
		result, err := dm.transformForTable(data, widget)
		require.NoError(t, err)

		resultMap, ok := result.(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "table", resultMap["type"])
		assert.Contains(t, resultMap, "rows")
		assert.Contains(t, resultMap, "columns")

		columns, ok := resultMap["columns"].([]string)
		require.True(t, ok)
		assert.Contains(t, columns, "name")
		assert.Contains(t, columns, "age")
		assert.Contains(t, columns, "city")
	})
}

func TestDashboardManagerUtilityFunctions(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	t.Run("Extract ID from Path", func(t *testing.T) {
		id := dm.extractIDFromPath("/api/v1/widgets/widget-123", "/api/v1/widgets/")
		assert.Equal(t, "widget-123", id)

		id = dm.extractIDFromPath("/api/v1/widgets/", "/api/v1/widgets/")
		assert.Equal(t, "", id)
	})

	t.Run("Generate Widget ID", func(t *testing.T) {
		id1 := dm.generateWidgetID()
		id2 := dm.generateWidgetID()

		assert.True(t, strings.HasPrefix(id1, "widget_"))
		assert.True(t, strings.HasPrefix(id2, "widget_"))
		assert.NotEqual(t, id1, id2)
	})

	t.Run("Generate Client ID", func(t *testing.T) {
		id1 := dm.generateClientID()
		id2 := dm.generateClientID()

		assert.True(t, strings.HasPrefix(id1, "client_"))
		assert.True(t, strings.HasPrefix(id2, "client_"))
		assert.NotEqual(t, id1, id2)
	})

	t.Run("Parse Time Parameter", func(t *testing.T) {
		// Valid time
		timeStr := "2023-01-01T00:00:00Z"
		parsedTime := dm.parseTimeParam(timeStr)
		require.NotNil(t, parsedTime)
		assert.Equal(t, 2023, parsedTime.Year())

		// Invalid time
		parsedTime = dm.parseTimeParam("invalid-time")
		assert.Nil(t, parsedTime)

		// Empty time
		parsedTime = dm.parseTimeParam("")
		assert.Nil(t, parsedTime)
	})

	t.Run("Parse Filters", func(t *testing.T) {
		query := map[string][]string{
			"category": {"A"},
			"tags":     {"tag1", "tag2"},
			"status":   {"active"},
		}

		filters := dm.parseFilters(query)
		assert.Equal(t, "A", filters["category"])
		assert.Equal(t, []string{"tag1", "tag2"}, filters["tags"])
		assert.Equal(t, "active", filters["status"])
	})

	t.Run("Compare Values", func(t *testing.T) {
		assert.Equal(t, -1, dm.compareValues(5.0, 10.0))
		assert.Equal(t, 1, dm.compareValues(15.0, 10.0))
		assert.Equal(t, 0, dm.compareValues(10.0, 10.0))
		assert.Equal(t, 0, dm.compareValues("string", 10.0)) // Non-numeric comparison
	})
}

func TestDashboardManagerFilterEvaluation(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	item := map[string]interface{}{
		"value":    15.0,
		"category": "A",
		"status":   "active",
	}

	t.Run("Equality Filter", func(t *testing.T) {
		filter := FilterConfig{Field: "category", Operator: "eq", Value: "A"}
		assert.True(t, dm.evaluateFilter(item, filter))

		filter = FilterConfig{Field: "category", Operator: "eq", Value: "B"}
		assert.False(t, dm.evaluateFilter(item, filter))
	})

	t.Run("Not Equal Filter", func(t *testing.T) {
		filter := FilterConfig{Field: "category", Operator: "ne", Value: "B"}
		assert.True(t, dm.evaluateFilter(item, filter))

		filter = FilterConfig{Field: "category", Operator: "ne", Value: "A"}
		assert.False(t, dm.evaluateFilter(item, filter))
	})

	t.Run("Greater Than Filter", func(t *testing.T) {
		filter := FilterConfig{Field: "value", Operator: "gt", Value: 10.0}
		assert.True(t, dm.evaluateFilter(item, filter))

		filter = FilterConfig{Field: "value", Operator: "gt", Value: 20.0}
		assert.False(t, dm.evaluateFilter(item, filter))
	})

	t.Run("Less Than Filter", func(t *testing.T) {
		filter := FilterConfig{Field: "value", Operator: "lt", Value: 20.0}
		assert.True(t, dm.evaluateFilter(item, filter))

		filter = FilterConfig{Field: "value", Operator: "lt", Value: 10.0}
		assert.False(t, dm.evaluateFilter(item, filter))
	})

	t.Run("Non-existent Field", func(t *testing.T) {
		filter := FilterConfig{Field: "non_existent", Operator: "eq", Value: "test"}
		assert.False(t, dm.evaluateFilter(item, filter))
	})

	t.Run("Unknown Operator", func(t *testing.T) {
		filter := FilterConfig{Field: "value", Operator: "unknown", Value: 10.0}
		assert.False(t, dm.evaluateFilter(item, filter))
	})
}

func TestDashboardManagerThemes(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	themes := dm.getThemes()
	assert.Len(t, themes, 2) // light and dark themes

	// Check theme IDs
	themeIDs := make(map[string]bool)
	for _, theme := range themes {
		themeIDs[theme.ID] = true
	}
	assert.True(t, themeIDs["light"])
	assert.True(t, themeIDs["dark"])

	// Verify theme structure
	for _, theme := range themes {
		assert.NotEmpty(t, theme.Name)
		assert.NotEmpty(t, theme.Colors.Primary)
		assert.NotEmpty(t, theme.Colors.Background)
		assert.NotEmpty(t, theme.Colors.Text)
		assert.NotEmpty(t, theme.Colors.Chart)
	}
}

func TestGetDefaultDashboardConfig(t *testing.T) {
	config := getDefaultDashboardConfig()
	require.NotNil(t, config)

	assert.True(t, config.Enabled)
	assert.Equal(t, 8080, config.Port)
	assert.Equal(t, "localhost", config.Host)
	assert.True(t, config.EnableRealTime)
	assert.Equal(t, time.Second, config.UpdateInterval)
	assert.Equal(t, 1000, config.MaxConnections)
	assert.True(t, config.EnableCompression)
	assert.Equal(t, "light", config.DefaultTheme)
	assert.True(t, config.EnableCustomDashboards)
	assert.Equal(t, 50, config.MaxWidgetsPerDashboard)
	assert.Equal(t, 5*time.Minute, config.CacheTimeout)
	assert.True(t, config.EnableMetrics)
	assert.Equal(t, 24*time.Hour, config.SessionTimeout)
}

func TestWidgetValidation(t *testing.T) {
	dm, err := NewDashboardManager(nil, logrus.New())
	require.NoError(t, err)

	t.Run("Valid Widget", func(t *testing.T) {
		widget := &Widget{
			ID:    "test-widget",
			Name:  "Test Widget",
			Type:  WidgetTypeChart,
			Title: "Test Chart",
		}
		err := dm.validateWidget(widget)
		assert.NoError(t, err)
	})

	t.Run("Missing Name", func(t *testing.T) {
		widget := &Widget{
			ID:   "test-widget",
			Type: WidgetTypeChart,
		}
		err := dm.validateWidget(widget)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "widget name is required")
	})

	t.Run("Missing Type", func(t *testing.T) {
		widget := &Widget{
			ID:   "test-widget",
			Name: "Test Widget",
		}
		err := dm.validateWidget(widget)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "widget type is required")
	})
}

// Helper functions and mocks

func createTestDashboard() *Dashboard {
	return &Dashboard{
		ID:          "test-dashboard-1",
		Name:        "Test Dashboard",
		Description: "A test dashboard for unit tests",
		Layout: &DashboardLayout{
			Type:    LayoutTypeGrid,
			Columns: 12,
			Rows:    8,
			GridSize: GridSize{
				Width:  100,
				Height: 100,
			},
			Responsive: true,
		},
		Widgets: []*Widget{
			{
				ID:    "widget-1",
				Type:  WidgetTypeChart,
				Title: "Temperature Chart",
				Position: WidgetPosition{
					X: 0,
					Y: 0,
					Z: 1,
				},
				Size: WidgetSize{
					Width:  6,
					Height: 4,
				},
				Configuration: WidgetConfiguration{
					ChartType: ChartTypeLine,
					ChartOptions: ChartOptions{
						Responsive: true,
						ShowGrid:   true,
						ShowLabels: true,
					},
				},
				IsVisible: true,
			},
			{
				ID:    "widget-2",
				Type:  WidgetTypeMetric,
				Title: "Current Temperature",
				Position: WidgetPosition{
					X: 6,
					Y: 0,
					Z: 1,
				},
				Size: WidgetSize{
					Width:  3,
					Height: 2,
				},
				Configuration: WidgetConfiguration{
					MetricOptions: MetricOptions{
						Unit:      "°C",
						Precision: 1,
						Color:     "#1976d2",
					},
				},
				IsVisible: true,
			},
		},
		Theme:    "light",
		IsPublic: true,
		Owner:    "test-user",
		Tags:     []string{"test", "temperature", "monitoring"},
		Configuration: map[string]interface{}{
			"refresh_rate": "30s",
			"auto_layout":  true,
		},
		AutoRefresh:     true,
		RefreshInterval: 30 * time.Second,
	}
}

// MockDataProvider for testing
type MockDataProvider struct {
	data      []interface{}
	supportsRT bool
	subscribers map[string][]func(interface{})
}

func (m *MockDataProvider) GetData(ctx context.Context, config interface{}) (interface{}, error) {
	return m.data, nil
}

func (m *MockDataProvider) SupportsRealTime() bool {
	return m.supportsRT
}

func (m *MockDataProvider) Subscribe(ctx context.Context, config interface{}, callback func(interface{})) error {
	if m.subscribers == nil {
		m.subscribers = make(map[string][]func(interface{}))
	}
	key := "default"
	m.subscribers[key] = append(m.subscribers[key], callback)
	return nil
}

func (m *MockDataProvider) Unsubscribe(config interface{}) error {
	return nil
}

// Benchmark tests
func BenchmarkDashboardManagerCreateDashboard(b *testing.B) {
	dm, err := NewDashboardManager(nil, logrus.New())
	if err != nil {
		b.Fatal(err)
	}

	dashboard := createTestDashboard()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dashboard.ID = fmt.Sprintf("benchmark-dashboard-%d", i)
		err := dm.CreateDashboard(dashboard)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDashboardManagerGetDashboard(b *testing.B) {
	dm, err := NewDashboardManager(nil, logrus.New())
	if err != nil {
		b.Fatal(err)
	}

	// Pre-populate dashboards
	for i := 0; i < 100; i++ {
		dashboard := createTestDashboard()
		dashboard.ID = fmt.Sprintf("benchmark-dashboard-%d", i)
		dm.CreateDashboard(dashboard)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := dm.GetDashboard(fmt.Sprintf("benchmark-dashboard-%d", i%100))
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDashboardManagerListDashboards(b *testing.B) {
	dm, err := NewDashboardManager(nil, logrus.New())
	if err != nil {
		b.Fatal(err)
	}

	// Pre-populate dashboards
	for i := 0; i < 1000; i++ {
		dashboard := createTestDashboard()
		dashboard.ID = fmt.Sprintf("benchmark-dashboard-%d", i)
		dm.CreateDashboard(dashboard)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dashboards := dm.ListDashboards()
		if len(dashboards) != 1000 {
			b.Fatal("Expected 1000 dashboards")
		}
	}
}

func BenchmarkDashboardManagerDataTransformation(b *testing.B) {
	dm, err := NewDashboardManager(nil, logrus.New())
	if err != nil {
		b.Fatal(err)
	}

	data := make([]interface{}, 1000)
	for i := 0; i < 1000; i++ {
		data[i] = map[string]interface{}{
			"timestamp": time.Now().Add(time.Duration(i) * time.Minute).Format(time.RFC3339),
			"value":     float64(i),
			"series":    "test",
		}
	}

	widget := &Widget{Type: WidgetTypeChart}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := dm.transformForTimeSeriesChart(data, widget)
		if err != nil {
			b.Fatal(err)
		}
	}
}