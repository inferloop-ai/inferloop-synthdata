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

// DashboardManager manages the visualization dashboard
type DashboardManager struct {
	logger         *logrus.Logger
	config         *DashboardConfig
	server         *http.Server
	webSocketHub   *WebSocketHub
	chartManager   *ChartManager
	widgetManager  *WidgetManager
	dataProviders  map[string]DataProvider
	dashboards     map[string]*Dashboard
	themes         map[string]*Theme
	mu             sync.RWMutex
	stopCh         chan struct{}
}

// DashboardConfig configures the dashboard
type DashboardConfig struct {
	Enabled               bool              `json:"enabled"`
	Port                  int               `json:"port"`
	Host                  string            `json:"host"`
	EnableTLS             bool              `json:"enable_tls"`
	TLSCertFile           string            `json:"tls_cert_file"`
	TLSKeyFile            string            `json:"tls_key_file"`
	EnableAuth            bool              `json:"enable_auth"`
	JWTSecret             string            `json:"jwt_secret"`
	SessionTimeout        time.Duration     `json:"session_timeout"`
	EnableRealTime        bool              `json:"enable_real_time"`
	UpdateInterval        time.Duration     `json:"update_interval"`
	MaxConnections        int               `json:"max_connections"`
	EnableCompression     bool              `json:"enable_compression"`
	StaticFilePath        string            `json:"static_file_path"`
	TemplateFilePath      string            `json:"template_file_path"`
	DefaultTheme          string            `json:"default_theme"`
	EnableCustomDashboards bool             `json:"enable_custom_dashboards"`
	MaxWidgetsPerDashboard int              `json:"max_widgets_per_dashboard"`
	CacheTimeout          time.Duration     `json:"cache_timeout"`
	EnableMetrics         bool              `json:"enable_metrics"`
}

// Dashboard represents a visualization dashboard
type Dashboard struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Layout          *DashboardLayout       `json:"layout"`
	Widgets         []*Widget              `json:"widgets"`
	Theme           string                 `json:"theme"`
	IsPublic        bool                   `json:"is_public"`
	Owner           string                 `json:"owner"`
	Permissions     []Permission           `json:"permissions"`
	Tags            []string               `json:"tags"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	LastViewedAt    *time.Time             `json:"last_viewed_at,omitempty"`
	ViewCount       int64                  `json:"view_count"`
	Configuration   map[string]interface{} `json:"configuration"`
	AutoRefresh     bool                   `json:"auto_refresh"`
	RefreshInterval time.Duration          `json:"refresh_interval"`
}

// DashboardLayout defines the dashboard layout
type DashboardLayout struct {
	Type        LayoutType `json:"type"`
	Columns     int        `json:"columns"`
	Rows        int        `json:"rows"`
	GridSize    GridSize   `json:"grid_size"`
	Responsive  bool       `json:"responsive"`
	Breakpoints map[string]Breakpoint `json:"breakpoints"`
}

// LayoutType defines dashboard layout types
type LayoutType string

const (
	LayoutTypeGrid     LayoutType = "grid"
	LayoutTypeFlex     LayoutType = "flex"
	LayoutTypeMasonry  LayoutType = "masonry"
	LayoutTypeCustom   LayoutType = "custom"
)

// GridSize defines grid dimensions
type GridSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Breakpoint defines responsive breakpoints
type Breakpoint struct {
	MinWidth int `json:"min_width"`
	Columns  int `json:"columns"`
}

// Widget represents a dashboard widget
type Widget struct {
	ID            string                 `json:"id"`
	Type          WidgetType             `json:"type"`
	Title         string                 `json:"title"`
	Description   string                 `json:"description"`
	Position      WidgetPosition         `json:"position"`
	Size          WidgetSize             `json:"size"`
	Configuration WidgetConfiguration    `json:"configuration"`
	DataSource    DataSourceConfig       `json:"data_source"`
	Filters       []Filter               `json:"filters"`
	Aggregations  []Aggregation          `json:"aggregations"`
	Transformations []Transformation     `json:"transformations"`
	Styling       WidgetStyling          `json:"styling"`
	Interactions  []Interaction          `json:"interactions"`
	Alerts        []WidgetAlert          `json:"alerts"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	IsVisible     bool                   `json:"is_visible"`
	RefreshRate   time.Duration          `json:"refresh_rate"`
}

// WidgetType defines widget types
type WidgetType string

const (
	WidgetTypeChart          WidgetType = "chart"
	WidgetTypeTable          WidgetType = "table"
	WidgetTypeMetric         WidgetType = "metric"
	WidgetTypeGauge          WidgetType = "gauge"
	WidgetTypeMap            WidgetType = "map"
	WidgetTypeText           WidgetType = "text"
	WidgetTypeImage          WidgetType = "image"
	WidgetTypeVideo          WidgetType = "video"
	WidgetTypeHeatmap        WidgetType = "heatmap"
	WidgetTypeTimeline       WidgetType = "timeline"
	WidgetTypeHistogram      WidgetType = "histogram"
	WidgetTypeScatterPlot    WidgetType = "scatter_plot"
	WidgetTypeTreeMap        WidgetType = "tree_map"
	WidgetTypeSankey         WidgetType = "sankey"
	WidgetTypeWordCloud      WidgetType = "word_cloud"
	WidgetTypeProgress       WidgetType = "progress"
	WidgetTypeStat           WidgetType = "stat"
	WidgetTypeAlert          WidgetType = "alert"
	WidgetTypeCustom         WidgetType = "custom"
)

// WidgetPosition defines widget position
type WidgetPosition struct {
	X int `json:"x"`
	Y int `json:"y"`
	Z int `json:"z"` // Layer order
}

// WidgetSize defines widget dimensions
type WidgetSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
	MinWidth  int `json:"min_width"`
	MinHeight int `json:"min_height"`
	MaxWidth  int `json:"max_width"`
	MaxHeight int `json:"max_height"`
}

// WidgetConfiguration contains widget-specific configuration
type WidgetConfiguration struct {
	ChartType       ChartType              `json:"chart_type,omitempty"`
	ChartOptions    ChartOptions           `json:"chart_options,omitempty"`
	TableOptions    TableOptions           `json:"table_options,omitempty"`
	MetricOptions   MetricOptions          `json:"metric_options,omitempty"`
	MapOptions      MapOptions             `json:"map_options,omitempty"`
	CustomOptions   map[string]interface{} `json:"custom_options,omitempty"`
}

// ChartType defines chart types
type ChartType string

const (
	ChartTypeLine        ChartType = "line"
	ChartTypeArea        ChartType = "area"
	ChartTypeBar         ChartType = "bar"
	ChartTypeColumn      ChartType = "column"
	ChartTypePie         ChartType = "pie"
	ChartTypeDoughnut    ChartType = "doughnut"
	ChartTypeScatter     ChartType = "scatter"
	ChartTypeBubble      ChartType = "bubble"
	ChartTypeRadar       ChartType = "radar"
	ChartTypePolar       ChartType = "polar"
	ChartTypeCandle      ChartType = "candle"
	ChartTypeWaterfall   ChartType = "waterfall"
	ChartTypeFunnel      ChartType = "funnel"
	ChartTypeGantt       ChartType = "gantt"
)

// ChartOptions contains chart-specific options
type ChartOptions struct {
	XAxis         AxisConfig    `json:"x_axis"`
	YAxis         AxisConfig    `json:"y_axis"`
	Legend        LegendConfig  `json:"legend"`
	Tooltip       TooltipConfig `json:"tooltip"`
	Animation     AnimationConfig `json:"animation"`
	Colors        []string      `json:"colors"`
	Responsive    bool          `json:"responsive"`
	MaintainAspectRatio bool    `json:"maintain_aspect_ratio"`
	ShowGrid      bool          `json:"show_grid"`
	ShowLabels    bool          `json:"show_labels"`
	Stacked       bool          `json:"stacked"`
	Smooth        bool          `json:"smooth"`
	FillArea      bool          `json:"fill_area"`
	ShowPoints    bool          `json:"show_points"`
	PointRadius   int           `json:"point_radius"`
	LineWidth     int           `json:"line_width"`
	BorderWidth   int           `json:"border_width"`
}

// AxisConfig configures chart axes
type AxisConfig struct {
	Title       string      `json:"title"`
	Type        string      `json:"type"` // linear, logarithmic, time, category
	Min         *float64    `json:"min,omitempty"`
	Max         *float64    `json:"max,omitempty"`
	StepSize    *float64    `json:"step_size,omitempty"`
	Format      string      `json:"format"`
	Show        bool        `json:"show"`
	Position    string      `json:"position"` // top, bottom, left, right
	GridLines   bool        `json:"grid_lines"`
	TickMarks   bool        `json:"tick_marks"`
	Labels      []string    `json:"labels,omitempty"`
}

// LegendConfig configures chart legend
type LegendConfig struct {
	Show     bool   `json:"show"`
	Position string `json:"position"` // top, bottom, left, right
	Align    string `json:"align"`    // start, center, end
	Labels   LabelConfig `json:"labels"`
}

// LabelConfig configures labels
type LabelConfig struct {
	FontSize   int    `json:"font_size"`
	FontColor  string `json:"font_color"`
	FontFamily string `json:"font_family"`
}

// TooltipConfig configures chart tooltips
type TooltipConfig struct {
	Enabled     bool   `json:"enabled"`
	Mode        string `json:"mode"` // point, nearest, index, dataset
	Intersect   bool   `json:"intersect"`
	Position    string `json:"position"`
	Format      string `json:"format"`
	Background  string `json:"background"`
	BorderColor string `json:"border_color"`
	BorderWidth int    `json:"border_width"`
}

// AnimationConfig configures animations
type AnimationConfig struct {
	Enabled  bool          `json:"enabled"`
	Duration time.Duration `json:"duration"`
	Easing   string        `json:"easing"`
	Delay    time.Duration `json:"delay"`
}

// TableOptions contains table-specific options
type TableOptions struct {
	Pagination  PaginationConfig `json:"pagination"`
	Sorting     SortingConfig    `json:"sorting"`
	Filtering   FilteringConfig  `json:"filtering"`
	Columns     []ColumnConfig   `json:"columns"`
	RowHeight   int              `json:"row_height"`
	Striped     bool             `json:"striped"`
	Bordered    bool             `json:"bordered"`
	Hover       bool             `json:"hover"`
	Compact     bool             `json:"compact"`
	Responsive  bool             `json:"responsive"`
	Selection   bool             `json:"selection"`
	Export      bool             `json:"export"`
}

// PaginationConfig configures table pagination
type PaginationConfig struct {
	Enabled   bool `json:"enabled"`
	PageSize  int  `json:"page_size"`
	ShowInfo  bool `json:"show_info"`
	ShowSizer bool `json:"show_sizer"`
}

// SortingConfig configures table sorting
type SortingConfig struct {
	Enabled     bool     `json:"enabled"`
	MultiColumn bool     `json:"multi_column"`
	DefaultSort []string `json:"default_sort"`
}

// FilteringConfig configures table filtering
type FilteringConfig struct {
	Enabled    bool `json:"enabled"`
	GlobalSearch bool `json:"global_search"`
	ColumnFilters bool `json:"column_filters"`
}

// ColumnConfig configures table columns
type ColumnConfig struct {
	Field      string `json:"field"`
	Title      string `json:"title"`
	Width      int    `json:"width"`
	MinWidth   int    `json:"min_width"`
	MaxWidth   int    `json:"max_width"`
	Sortable   bool   `json:"sortable"`
	Filterable bool   `json:"filterable"`
	Resizable  bool   `json:"resizable"`
	Align      string `json:"align"`
	Format     string `json:"format"`
	Visible    bool   `json:"visible"`
}

// MetricOptions contains metric widget options
type MetricOptions struct {
	Value       string          `json:"value"`
	Unit        string          `json:"unit"`
	Precision   int             `json:"precision"`
	Threshold   MetricThreshold `json:"threshold"`
	Comparison  MetricComparison `json:"comparison"`
	Sparkline   bool            `json:"sparkline"`
	Icon        string          `json:"icon"`
	Color       string          `json:"color"`
	FontSize    int             `json:"font_size"`
	Alignment   string          `json:"alignment"`
}

// MetricThreshold defines metric thresholds
type MetricThreshold struct {
	Critical float64 `json:"critical"`
	Warning  float64 `json:"warning"`
	Good     float64 `json:"good"`
}

// MetricComparison defines metric comparison
type MetricComparison struct {
	Enabled        bool    `json:"enabled"`
	PreviousValue  float64 `json:"previous_value"`
	ChangePercent  float64 `json:"change_percent"`
	ChangeAbsolute float64 `json:"change_absolute"`
	Direction      string  `json:"direction"` // up, down, neutral
}

// MapOptions contains map widget options
type MapOptions struct {
	Center      LatLng          `json:"center"`
	Zoom        int             `json:"zoom"`
	MapType     string          `json:"map_type"` // roadmap, satellite, hybrid, terrain
	Markers     []MapMarker     `json:"markers"`
	Layers      []MapLayer      `json:"layers"`
	Controls    MapControls     `json:"controls"`
	Clustering  bool            `json:"clustering"`
	HeatMap     bool            `json:"heat_map"`
	Boundaries  bool            `json:"boundaries"`
}

// LatLng represents latitude and longitude
type LatLng struct {
	Lat float64 `json:"lat"`
	Lng float64 `json:"lng"`
}

// MapMarker represents a map marker
type MapMarker struct {
	Position LatLng                 `json:"position"`
	Title    string                 `json:"title"`
	Content  string                 `json:"content"`
	Icon     string                 `json:"icon"`
	Color    string                 `json:"color"`
	Data     map[string]interface{} `json:"data"`
}

// MapLayer represents a map layer
type MapLayer struct {
	ID      string                 `json:"id"`
	Type    string                 `json:"type"` // geojson, wms, tile
	Source  string                 `json:"source"`
	Style   map[string]interface{} `json:"style"`
	Visible bool                   `json:"visible"`
}

// MapControls configures map controls
type MapControls struct {
	Zoom       bool `json:"zoom"`
	FullScreen bool `json:"full_screen"`
	Scale      bool `json:"scale"`
	Layers     bool `json:"layers"`
	Search     bool `json:"search"`
}

// DataSourceConfig configures widget data source
type DataSourceConfig struct {
	Type       DataSourceType         `json:"type"`
	URL        string                 `json:"url,omitempty"`
	Query      string                 `json:"query,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Headers    map[string]string      `json:"headers,omitempty"`
	Method     string                 `json:"method,omitempty"`
	Body       string                 `json:"body,omitempty"`
	Polling    PollingConfig          `json:"polling"`
	Cache      CacheConfig            `json:"cache"`
}

// DataSourceType defines data source types
type DataSourceType string

const (
	DataSourceTypeStatic    DataSourceType = "static"
	DataSourceTypeREST      DataSourceType = "rest"
	DataSourceTypeWebSocket DataSourceType = "websocket"
	DataSourceTypeSSE       DataSourceType = "sse"
	DataSourceTypeDatabase  DataSourceType = "database"
	DataSourceTypeFile      DataSourceType = "file"
	DataSourceTypeStream    DataSourceType = "stream"
)

// PollingConfig configures data polling
type PollingConfig struct {
	Enabled  bool          `json:"enabled"`
	Interval time.Duration `json:"interval"`
	Timeout  time.Duration `json:"timeout"`
}

// CacheConfig configures data caching
type CacheConfig struct {
	Enabled bool          `json:"enabled"`
	TTL     time.Duration `json:"ttl"`
	Key     string        `json:"key"`
}

// Filter defines data filtering
type Filter struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"` // eq, ne, gt, lt, gte, lte, in, contains
	Value    interface{} `json:"value"`
	Logic    string      `json:"logic"` // and, or
}

// Aggregation defines data aggregation
type Aggregation struct {
	Field    string `json:"field"`
	Function string `json:"function"` // sum, avg, min, max, count, distinct
	GroupBy  string `json:"group_by,omitempty"`
}

// Transformation defines data transformation
type Transformation struct {
	Type   string                 `json:"type"` // map, filter, reduce, sort, group
	Config map[string]interface{} `json:"config"`
}

// WidgetStyling defines widget appearance
type WidgetStyling struct {
	Background    string            `json:"background"`
	Border        BorderStyle       `json:"border"`
	Padding       SpacingStyle      `json:"padding"`
	Margin        SpacingStyle      `json:"margin"`
	BorderRadius  int               `json:"border_radius"`
	BoxShadow     string            `json:"box_shadow"`
	Opacity       float64           `json:"opacity"`
	CustomCSS     string            `json:"custom_css"`
	Classes       []string          `json:"classes"`
}

// BorderStyle defines border styling
type BorderStyle struct {
	Width int    `json:"width"`
	Style string `json:"style"` // solid, dashed, dotted
	Color string `json:"color"`
}

// SpacingStyle defines spacing
type SpacingStyle struct {
	Top    int `json:"top"`
	Right  int `json:"right"`
	Bottom int `json:"bottom"`
	Left   int `json:"left"`
}

// Interaction defines widget interactions
type Interaction struct {
	Type   InteractionType        `json:"type"`
	Target string                 `json:"target"`
	Action string                 `json:"action"`
	Config map[string]interface{} `json:"config"`
}

// InteractionType defines interaction types
type InteractionType string

const (
	InteractionTypeClick       InteractionType = "click"
	InteractionTypeHover       InteractionType = "hover"
	InteractionTypeDoubleClick InteractionType = "double_click"
	InteractionTypeDrag        InteractionType = "drag"
	InteractionTypeZoom        InteractionType = "zoom"
	InteractionTypeSelect      InteractionType = "select"
)

// WidgetAlert defines widget alerts
type WidgetAlert struct {
	ID          string              `json:"id"`
	Name        string              `json:"name"`
	Condition   AlertCondition      `json:"condition"`
	Actions     []AlertAction       `json:"actions"`
	Enabled     bool                `json:"enabled"`
	Severity    AlertSeverity       `json:"severity"`
	Throttle    time.Duration       `json:"throttle"`
	LastTriggered *time.Time        `json:"last_triggered,omitempty"`
}

// AlertCondition defines alert conditions
type AlertCondition struct {
	Field     string      `json:"field"`
	Operator  string      `json:"operator"`
	Value     interface{} `json:"value"`
	Duration  time.Duration `json:"duration"`
	Frequency int         `json:"frequency"`
}

// AlertAction defines alert actions
type AlertAction struct {
	Type   string                 `json:"type"` // email, webhook, notification
	Config map[string]interface{} `json:"config"`
}

// AlertSeverity defines alert severity levels
type AlertSeverity string

const (
	AlertSeverityInfo     AlertSeverity = "info"
	AlertSeverityWarning  AlertSeverity = "warning"
	AlertSeverityError    AlertSeverity = "error"
	AlertSeverityCritical AlertSeverity = "critical"
)

// Permission defines dashboard permissions
type Permission struct {
	User        string           `json:"user"`
	Role        string           `json:"role"`
	Permissions []PermissionType `json:"permissions"`
}

// PermissionType defines permission types
type PermissionType string

const (
	PermissionTypeView   PermissionType = "view"
	PermissionTypeEdit   PermissionType = "edit"
	PermissionTypeDelete PermissionType = "delete"
	PermissionTypeShare  PermissionType = "share"
)

// Theme defines dashboard themes
type Theme struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Colors      ThemeColors            `json:"colors"`
	Fonts       ThemeFonts             `json:"fonts"`
	Spacing     ThemeSpacing           `json:"spacing"`
	Components  map[string]interface{} `json:"components"`
	CustomCSS   string                 `json:"custom_css"`
}

// ThemeColors defines theme colors
type ThemeColors struct {
	Primary     string   `json:"primary"`
	Secondary   string   `json:"secondary"`
	Success     string   `json:"success"`
	Warning     string   `json:"warning"`
	Error       string   `json:"error"`
	Info        string   `json:"info"`
	Background  string   `json:"background"`
	Surface     string   `json:"surface"`
	Text        string   `json:"text"`
	TextSecondary string `json:"text_secondary"`
	Border      string   `json:"border"`
	Chart       []string `json:"chart"`
}

// ThemeFonts defines theme fonts
type ThemeFonts struct {
	Primary    FontConfig `json:"primary"`
	Secondary  FontConfig `json:"secondary"`
	Monospace  FontConfig `json:"monospace"`
}

// FontConfig configures fonts
type FontConfig struct {
	Family string `json:"family"`
	Size   int    `json:"size"`
	Weight string `json:"weight"`
}

// ThemeSpacing defines theme spacing
type ThemeSpacing struct {
	XSmall int `json:"x_small"`
	Small  int `json:"small"`
	Medium int `json:"medium"`
	Large  int `json:"large"`
	XLarge int `json:"x_large"`
}

// DataProvider interface for providing data to widgets
type DataProvider interface {
	GetData(ctx context.Context, config DataSourceConfig) (interface{}, error)
	SupportsRealTime() bool
	Subscribe(ctx context.Context, config DataSourceConfig, callback func(interface{})) error
	Unsubscribe(config DataSourceConfig) error
}

// NewDashboardManager creates a new dashboard manager
func NewDashboardManager(config *DashboardConfig, logger *logrus.Logger) (*DashboardManager, error) {
	if config == nil {
		config = getDefaultDashboardConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	dm := &DashboardManager{
		logger:        logger,
		config:        config,
		dataProviders: make(map[string]DataProvider),
		dashboards:    make(map[string]*Dashboard),
		themes:        make(map[string]*Theme),
		stopCh:        make(chan struct{}),
	}

	// Initialize components
	dm.webSocketHub = NewWebSocketHub(logger)
	dm.chartManager = NewChartManager(logger)
	dm.widgetManager = NewWidgetManager(logger)

	// Load default themes
	dm.loadDefaultThemes()

	// Register default data providers
	dm.registerDefaultDataProviders()

	return dm, nil
}

// Start starts the dashboard manager
func (dm *DashboardManager) Start(ctx context.Context) error {
	if !dm.config.Enabled {
		dm.logger.Info("Dashboard disabled")
		return nil
	}

	dm.logger.Info("Starting dashboard manager")

	// Start WebSocket hub
	go dm.webSocketHub.Run(ctx)

	// Start HTTP server
	return dm.startHTTPServer()
}

// Stop stops the dashboard manager
func (dm *DashboardManager) Stop(ctx context.Context) error {
	dm.logger.Info("Stopping dashboard manager")

	close(dm.stopCh)

	if dm.server != nil {
		return dm.server.Shutdown(ctx)
	}

	return nil
}

// CreateDashboard creates a new dashboard
func (dm *DashboardManager) CreateDashboard(dashboard *Dashboard) error {
	if err := dm.validateDashboard(dashboard); err != nil {
		return fmt.Errorf("invalid dashboard: %w", err)
	}

	dm.mu.Lock()
	defer dm.mu.Unlock()

	dashboard.CreatedAt = time.Now()
	dashboard.UpdatedAt = time.Now()
	dm.dashboards[dashboard.ID] = dashboard

	dm.logger.WithField("dashboard_id", dashboard.ID).Info("Created dashboard")
	return nil
}

// GetDashboard returns a dashboard by ID
func (dm *DashboardManager) GetDashboard(id string) (*Dashboard, error) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	dashboard, exists := dm.dashboards[id]
	if !exists {
		return nil, fmt.Errorf("dashboard not found: %s", id)
	}

	return dashboard, nil
}

// UpdateDashboard updates a dashboard
func (dm *DashboardManager) UpdateDashboard(dashboard *Dashboard) error {
	if err := dm.validateDashboard(dashboard); err != nil {
		return fmt.Errorf("invalid dashboard: %w", err)
	}

	dm.mu.Lock()
	defer dm.mu.Unlock()

	existing, exists := dm.dashboards[dashboard.ID]
	if !exists {
		return fmt.Errorf("dashboard not found: %s", dashboard.ID)
	}

	dashboard.CreatedAt = existing.CreatedAt
	dashboard.UpdatedAt = time.Now()
	dm.dashboards[dashboard.ID] = dashboard

	dm.logger.WithField("dashboard_id", dashboard.ID).Info("Updated dashboard")
	return nil
}

// DeleteDashboard deletes a dashboard
func (dm *DashboardManager) DeleteDashboard(id string) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	if _, exists := dm.dashboards[id]; !exists {
		return fmt.Errorf("dashboard not found: %s", id)
	}

	delete(dm.dashboards, id)

	dm.logger.WithField("dashboard_id", id).Info("Deleted dashboard")
	return nil
}

// ListDashboards returns all dashboards
func (dm *DashboardManager) ListDashboards() []*Dashboard {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	dashboards := make([]*Dashboard, 0, len(dm.dashboards))
	for _, dashboard := range dm.dashboards {
		dashboards = append(dashboards, dashboard)
	}

	return dashboards
}

// RegisterDataProvider registers a data provider
func (dm *DashboardManager) RegisterDataProvider(name string, provider DataProvider) {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.dataProviders[name] = provider
	dm.logger.WithField("provider", name).Info("Registered data provider")
}

// Helper methods

func (dm *DashboardManager) validateDashboard(dashboard *Dashboard) error {
	if dashboard.ID == "" {
		return fmt.Errorf("dashboard ID is required")
	}

	if dashboard.Name == "" {
		return fmt.Errorf("dashboard name is required")
	}

	if len(dashboard.Widgets) > dm.config.MaxWidgetsPerDashboard {
		return fmt.Errorf("too many widgets: max %d allowed", dm.config.MaxWidgetsPerDashboard)
	}

	return nil
}

func (dm *DashboardManager) startHTTPServer() error {
	mux := http.NewServeMux()

	// API routes
	mux.HandleFunc("/api/dashboards", dm.handleDashboards)
	mux.HandleFunc("/api/dashboards/", dm.handleDashboard)
	mux.HandleFunc("/api/widgets/", dm.handleWidget)
	mux.HandleFunc("/api/data", dm.handleData)
	mux.HandleFunc("/api/themes", dm.handleThemes)
	
	// WebSocket endpoint
	mux.HandleFunc("/ws", dm.handleWebSocket)

	// Static files
	if dm.config.StaticFilePath != "" {
		mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir(dm.config.StaticFilePath))))
	}

	addr := fmt.Sprintf("%s:%d", dm.config.Host, dm.config.Port)
	dm.server = &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	dm.logger.WithField("addr", addr).Info("Starting dashboard HTTP server")

	if dm.config.EnableTLS {
		return dm.server.ListenAndServeTLS(dm.config.TLSCertFile, dm.config.TLSKeyFile)
	}

	return dm.server.ListenAndServe()
}

func (dm *DashboardManager) handleDashboards(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		dashboards := dm.ListDashboards()
		dm.writeJSON(w, dashboards)
	case http.MethodPost:
		var dashboard Dashboard
		if err := json.NewDecoder(r.Body).Decode(&dashboard); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		if err := dm.CreateDashboard(&dashboard); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		dm.writeJSON(w, dashboard)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (dm *DashboardManager) handleDashboard(w http.ResponseWriter, r *http.Request) {
	// Extract dashboard ID from URL path
	// Implementation would parse the URL to get the dashboard ID
	dashboardID := "example" // Placeholder
	
	switch r.Method {
	case http.MethodGet:
		dashboard, err := dm.GetDashboard(dashboardID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		dm.writeJSON(w, dashboard)
	case http.MethodPut:
		var dashboard Dashboard
		if err := json.NewDecoder(r.Body).Decode(&dashboard); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		dashboard.ID = dashboardID
		if err := dm.UpdateDashboard(&dashboard); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		dm.writeJSON(w, dashboard)
	case http.MethodDelete:
		if err := dm.DeleteDashboard(dashboardID); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (dm *DashboardManager) handleWidget(w http.ResponseWriter, r *http.Request) {
	// Widget-specific endpoint handling
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "widget endpoint"})
}

func (dm *DashboardManager) handleData(w http.ResponseWriter, r *http.Request) {
	// Data endpoint handling
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "data endpoint"})
}

func (dm *DashboardManager) handleThemes(w http.ResponseWriter, r *http.Request) {
	themes := dm.getThemes()
	dm.writeJSON(w, themes)
}

func (dm *DashboardManager) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	dm.webSocketHub.HandleConnection(w, r)
}

func (dm *DashboardManager) writeJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(data); err != nil {
		dm.logger.WithError(err).Error("Failed to encode JSON response")
		http.Error(w, "Internal server error", http.StatusInternalServerError)
	}
}

func (dm *DashboardManager) loadDefaultThemes() {
	// Load built-in themes
	lightTheme := &Theme{
		ID:   "light",
		Name: "Light Theme",
		Colors: ThemeColors{
			Primary:   "#1976d2",
			Secondary: "#424242",
			Success:   "#4caf50",
			Warning:   "#ff9800",
			Error:     "#f44336",
			Info:      "#2196f3",
			Background: "#ffffff",
			Surface:   "#f5f5f5",
			Text:      "#212121",
			Chart:     []string{"#1976d2", "#4caf50", "#ff9800", "#f44336", "#9c27b0", "#00bcd4"},
		},
	}

	darkTheme := &Theme{
		ID:   "dark",
		Name: "Dark Theme",
		Colors: ThemeColors{
			Primary:   "#90caf9",
			Secondary: "#ce93d8",
			Success:   "#81c784",
			Warning:   "#ffb74d",
			Error:     "#e57373",
			Info:      "#64b5f6",
			Background: "#121212",
			Surface:   "#1e1e1e",
			Text:      "#ffffff",
			Chart:     []string{"#90caf9", "#81c784", "#ffb74d", "#e57373", "#ce93d8", "#4dd0e1"},
		},
	}

	dm.themes["light"] = lightTheme
	dm.themes["dark"] = darkTheme
}

func (dm *DashboardManager) registerDefaultDataProviders() {
	// Register built-in data providers
	dm.RegisterDataProvider("static", NewStaticDataProvider())
	dm.RegisterDataProvider("rest", NewRESTDataProvider())
	dm.RegisterDataProvider("websocket", NewWebSocketDataProvider())
}

func (dm *DashboardManager) getThemes() []*Theme {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	themes := make([]*Theme, 0, len(dm.themes))
	for _, theme := range dm.themes {
		themes = append(themes, theme)
	}

	return themes
}

func getDefaultDashboardConfig() *DashboardConfig {
	return &DashboardConfig{
		Enabled:               true,
		Port:                  8080,
		Host:                  "localhost",
		EnableRealTime:        true,
		UpdateInterval:        time.Second,
		MaxConnections:        1000,
		EnableCompression:     true,
		DefaultTheme:          "light",
		EnableCustomDashboards: true,
		MaxWidgetsPerDashboard: 50,
		CacheTimeout:          5 * time.Minute,
		EnableMetrics:         true,
		SessionTimeout:        24 * time.Hour,
	}
}