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
	w.Header().Set("Content-Type", "application/json")
	
	// Extract widget ID from URL path
	widgetID := dm.extractIDFromPath(r.URL.Path, "/api/v1/widgets/")
	
	switch r.Method {
	case http.MethodGet:
		if widgetID == "" {
			// List all widgets
			widgets := dm.widgetManager.GetAllWidgets()
			dm.writeJSON(w, map[string]interface{}{
				"widgets": widgets,
				"count":   len(widgets),
			})
			return
		}
		
		// Get specific widget
		widget, err := dm.widgetManager.GetWidget(widgetID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Widget not found: %v", err), http.StatusNotFound)
			return
		}
		
		// Render widget data
		data, err := dm.renderWidgetData(r.Context(), widget)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to render widget data: %v", err), http.StatusInternalServerError)
			return
		}
		
		response := map[string]interface{}{
			"widget": widget,
			"data":   data,
		}
		dm.writeJSON(w, response)
		
	case http.MethodPost:
		// Create new widget
		var widget Widget
		if err := json.NewDecoder(r.Body).Decode(&widget); err != nil {
			http.Error(w, fmt.Sprintf("Invalid widget data: %v", err), http.StatusBadRequest)
			return
		}
		
		// Generate ID if not provided
		if widget.ID == "" {
			widget.ID = dm.generateWidgetID()
		}
		
		// Set creation time
		widget.CreatedAt = time.Now()
		widget.UpdatedAt = time.Now()
		
		// Validate widget configuration
		if err := dm.validateWidget(&widget); err != nil {
			http.Error(w, fmt.Sprintf("Invalid widget configuration: %v", err), http.StatusBadRequest)
			return
		}
		
		// Create widget
		if err := dm.widgetManager.CreateWidget(&widget); err != nil {
			http.Error(w, fmt.Sprintf("Failed to create widget: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.WriteHeader(http.StatusCreated)
		dm.writeJSON(w, widget)
		
	case http.MethodPut:
		if widgetID == "" {
			http.Error(w, "Widget ID is required", http.StatusBadRequest)
			return
		}
		
		// Update widget
		var widget Widget
		if err := json.NewDecoder(r.Body).Decode(&widget); err != nil {
			http.Error(w, fmt.Sprintf("Invalid widget data: %v", err), http.StatusBadRequest)
			return
		}
		
		widget.ID = widgetID
		widget.UpdatedAt = time.Now()
		
		if err := dm.validateWidget(&widget); err != nil {
			http.Error(w, fmt.Sprintf("Invalid widget configuration: %v", err), http.StatusBadRequest)
			return
		}
		
		if err := dm.widgetManager.UpdateWidget(&widget); err != nil {
			http.Error(w, fmt.Sprintf("Failed to update widget: %v", err), http.StatusInternalServerError)
			return
		}
		
		dm.writeJSON(w, widget)
		
	case http.MethodDelete:
		if widgetID == "" {
			http.Error(w, "Widget ID is required", http.StatusBadRequest)
			return
		}
		
		if err := dm.widgetManager.DeleteWidget(widgetID); err != nil {
			http.Error(w, fmt.Sprintf("Failed to delete widget: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.WriteHeader(http.StatusNoContent)
		
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (dm *DashboardManager) handleData(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	// Parse query parameters
	query := r.URL.Query()
	
	// Extract data source and parameters
	dataSource := query.Get("source")
	widgetID := query.Get("widget_id")
	startTime := query.Get("start_time")
	endTime := query.Get("end_time")
	aggregation := query.Get("aggregation")
	granularity := query.Get("granularity")
	
	switch r.Method {
	case http.MethodGet:
		// Handle real-time data requests
		if query.Get("realtime") == "true" {
			dm.handleRealTimeData(w, r)
			return
		}
		
		// Handle historical data requests
		data, err := dm.fetchData(r.Context(), DataRequest{
			Source:      dataSource,
			WidgetID:    widgetID,
			StartTime:   dm.parseTimeParam(startTime),
			EndTime:     dm.parseTimeParam(endTime),
			Aggregation: aggregation,
			Granularity: granularity,
			Filters:     dm.parseFilters(query),
		})
		
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to fetch data: %v", err), http.StatusInternalServerError)
			return
		}
		
		response := map[string]interface{}{
			"data":      data,
			"timestamp": time.Now(),
			"count":     len(data),
			"metadata": map[string]interface{}{
				"source":      dataSource,
				"aggregation": aggregation,
				"granularity": granularity,
			},
		}
		
		dm.writeJSON(w, response)
		
	case http.MethodPost:
		// Handle data query with complex parameters
		var queryRequest DataQueryRequest
		if err := json.NewDecoder(r.Body).Decode(&queryRequest); err != nil {
			http.Error(w, fmt.Sprintf("Invalid query request: %v", err), http.StatusBadRequest)
			return
		}
		
		data, err := dm.executeDataQuery(r.Context(), &queryRequest)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to execute query: %v", err), http.StatusInternalServerError)
			return
		}
		
		response := map[string]interface{}{
			"data":           data,
			"query":          queryRequest,
			"execution_time": time.Now(),
		}
		
		dm.writeJSON(w, response)
		
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// DataRequest represents a data request
type DataRequest struct {
	Source      string                 `json:"source"`
	WidgetID    string                 `json:"widget_id"`
	StartTime   *time.Time             `json:"start_time"`
	EndTime     *time.Time             `json:"end_time"`
	Aggregation string                 `json:"aggregation"`
	Granularity string                 `json:"granularity"`
	Filters     map[string]interface{} `json:"filters"`
}

// DataQueryRequest represents a complex data query
type DataQueryRequest struct {
	Sources     []string               `json:"sources"`
	TimeRange   TimeRange              `json:"time_range"`
	Aggregation AggregationConfig      `json:"aggregation"`
	Filters     []FilterConfig         `json:"filters"`
	Grouping    []string               `json:"grouping"`
	Sorting     []SortConfig           `json:"sorting"`
	Limit       int                    `json:"limit"`
	Offset      int                    `json:"offset"`
	Format      string                 `json:"format"`
}

type TimeRange struct {
	Start *time.Time `json:"start"`
	End   *time.Time `json:"end"`
}

type AggregationConfig struct {
	Method   string        `json:"method"`
	Interval time.Duration `json:"interval"`
	Function string        `json:"function"`
}

type FilterConfig struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
}

type SortConfig struct {
	Field string `json:"field"`
	Order string `json:"order"` // "asc" or "desc"
}

func (dm *DashboardManager) handleRealTimeData(w http.ResponseWriter, r *http.Request) {
	// Upgrade connection to WebSocket for real-time data
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // In production, implement proper origin checking
		},
	}
	
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		dm.logger.WithError(err).Error("Failed to upgrade to WebSocket")
		return
	}
	defer conn.Close()
	
	// Register connection with WebSocket hub
	client := &WebSocketClient{
		ID:         dm.generateClientID(),
		Connection: conn,
		Send:       make(chan []byte, 256),
		Hub:        dm.webSocketHub,
	}
	
	dm.webSocketHub.Register <- client
	
	// Start goroutines for reading and writing
	go client.WritePump()
	go client.ReadPump()
	
	dm.logger.WithField("client_id", client.ID).Info("WebSocket client connected")
}

func (dm *DashboardManager) fetchData(ctx context.Context, request DataRequest) ([]interface{}, error) {
	// Get appropriate data provider
	provider, exists := dm.dataProviders[request.Source]
	if !exists {
		return nil, fmt.Errorf("unknown data source: %s", request.Source)
	}
	
	// Fetch data from provider
	data, err := provider.GetData(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch data from provider: %w", err)
	}
	
	// Apply filters and transformations
	filteredData := dm.applyDataFilters(data, request.Filters)
	
	// Apply aggregation if specified
	if request.Aggregation != "" {
		aggregatedData, err := dm.aggregateData(filteredData, request.Aggregation, request.Granularity)
		if err != nil {
			return nil, fmt.Errorf("failed to aggregate data: %w", err)
		}
		return aggregatedData, nil
	}
	
	return filteredData, nil
}

func (dm *DashboardManager) executeDataQuery(ctx context.Context, query *DataQueryRequest) (interface{}, error) {
	// Complex query execution logic
	var allData []interface{}
	
	// Fetch data from all sources
	for _, source := range query.Sources {
		provider, exists := dm.dataProviders[source]
		if !exists {
			dm.logger.Warnf("Unknown data source: %s", source)
			continue
		}
		
		request := DataRequest{
			Source:    source,
			StartTime: query.TimeRange.Start,
			EndTime:   query.TimeRange.End,
		}
		
		data, err := provider.GetData(ctx, request)
		if err != nil {
			dm.logger.WithError(err).Warnf("Failed to fetch data from source: %s", source)
			continue
		}
		
		allData = append(allData, data...)
	}
	
	// Apply filters
	for _, filter := range query.Filters {
		allData = dm.applyFilter(allData, filter)
	}
	
	// Apply grouping if specified
	if len(query.Grouping) > 0 {
		allData = dm.applyGrouping(allData, query.Grouping)
	}
	
	// Apply sorting
	if len(query.Sorting) > 0 {
		allData = dm.applySorting(allData, query.Sorting)
	}
	
	// Apply pagination
	if query.Limit > 0 {
		start := query.Offset
		end := start + query.Limit
		if end > len(allData) {
			end = len(allData)
		}
		if start < len(allData) {
			allData = allData[start:end]
		} else {
			allData = []interface{}{}
		}
	}
	
	return allData, nil
}

// Helper functions

func (dm *DashboardManager) extractIDFromPath(path, prefix string) string {
	if len(path) <= len(prefix) {
		return ""
	}
	return path[len(prefix):]
}

func (dm *DashboardManager) generateWidgetID() string {
	return fmt.Sprintf("widget_%d", time.Now().UnixNano())
}

func (dm *DashboardManager) generateClientID() string {
	return fmt.Sprintf("client_%d", time.Now().UnixNano())
}

func (dm *DashboardManager) validateWidget(widget *Widget) error {
	if widget.Name == "" {
		return fmt.Errorf("widget name is required")
	}
	if widget.Type == "" {
		return fmt.Errorf("widget type is required")
	}
	return nil
}

func (dm *DashboardManager) renderWidgetData(ctx context.Context, widget *Widget) (interface{}, error) {
	// Get data for the widget
	if widget.DataSource == "" {
		return nil, fmt.Errorf("widget has no data source configured")
	}
	
	provider, exists := dm.dataProviders[widget.DataSource]
	if !exists {
		return nil, fmt.Errorf("unknown data source: %s", widget.DataSource)
	}
	
	request := DataRequest{
		Source:   widget.DataSource,
		WidgetID: widget.ID,
		Filters:  widget.Configuration,
	}
	
	data, err := provider.GetData(ctx, request)
	if err != nil {
		return nil, err
	}
	
	// Transform data based on widget type
	return dm.transformDataForWidget(data, widget)
}

func (dm *DashboardManager) transformDataForWidget(data []interface{}, widget *Widget) (interface{}, error) {
	switch widget.Type {
	case "line_chart", "area_chart":
		return dm.transformForTimeSeriesChart(data, widget)
	case "bar_chart", "column_chart":
		return dm.transformForBarChart(data, widget)
	case "pie_chart", "donut_chart":
		return dm.transformForPieChart(data, widget)
	case "gauge", "metric":
		return dm.transformForMetric(data, widget)
	case "table":
		return dm.transformForTable(data, widget)
	case "heatmap":
		return dm.transformForHeatmap(data, widget)
	default:
		return data, nil
	}
}

func (dm *DashboardManager) transformForTimeSeriesChart(data []interface{}, widget *Widget) (interface{}, error) {
	// Transform data into time series format
	series := make([]map[string]interface{}, 0)
	
	for _, item := range data {
		if point, ok := item.(map[string]interface{}); ok {
			series = append(series, map[string]interface{}{
				"timestamp": point["timestamp"],
				"value":     point["value"],
				"series":    point["series"],
			})
		}
	}
	
	return map[string]interface{}{
		"type":   "time_series",
		"series": series,
		"config": widget.Configuration,
	}, nil
}

func (dm *DashboardManager) transformForBarChart(data []interface{}, widget *Widget) (interface{}, error) {
	// Transform data for bar chart
	categories := make([]string, 0)
	values := make([]float64, 0)
	
	for _, item := range data {
		if point, ok := item.(map[string]interface{}); ok {
			if cat, ok := point["category"].(string); ok {
				categories = append(categories, cat)
			}
			if val, ok := point["value"].(float64); ok {
				values = append(values, val)
			}
		}
	}
	
	return map[string]interface{}{
		"type":       "bar_chart",
		"categories": categories,
		"values":     values,
		"config":     widget.Configuration,
	}, nil
}

func (dm *DashboardManager) transformForPieChart(data []interface{}, widget *Widget) (interface{}, error) {
	// Transform data for pie chart
	segments := make([]map[string]interface{}, 0)
	
	for _, item := range data {
		if point, ok := item.(map[string]interface{}); ok {
			segments = append(segments, map[string]interface{}{
				"label": point["label"],
				"value": point["value"],
				"color": point["color"],
			})
		}
	}
	
	return map[string]interface{}{
		"type":     "pie_chart",
		"segments": segments,
		"config":   widget.Configuration,
	}, nil
}

func (dm *DashboardManager) transformForMetric(data []interface{}, widget *Widget) (interface{}, error) {
	// Transform data for metric display
	if len(data) == 0 {
		return map[string]interface{}{
			"type":  "metric",
			"value": 0,
			"config": widget.Configuration,
		}, nil
	}
	
	// Use the first value or calculate aggregate
	var value interface{}
	if point, ok := data[0].(map[string]interface{}); ok {
		value = point["value"]
	}
	
	return map[string]interface{}{
		"type":   "metric",
		"value":  value,
		"config": widget.Configuration,
	}, nil
}

func (dm *DashboardManager) transformForTable(data []interface{}, widget *Widget) (interface{}, error) {
	// Transform data for table display
	return map[string]interface{}{
		"type":    "table",
		"rows":    data,
		"columns": dm.extractTableColumns(data),
		"config":  widget.Configuration,
	}, nil
}

func (dm *DashboardManager) transformForHeatmap(data []interface{}, widget *Widget) (interface{}, error) {
	// Transform data for heatmap
	matrix := make([][]interface{}, 0)
	
	// Group data into matrix format
	// This is a simplified implementation
	for _, item := range data {
		if point, ok := item.(map[string]interface{}); ok {
			row := []interface{}{
				point["x"],
				point["y"],
				point["value"],
			}
			matrix = append(matrix, row)
		}
	}
	
	return map[string]interface{}{
		"type":   "heatmap",
		"matrix": matrix,
		"config": widget.Configuration,
	}, nil
}

func (dm *DashboardManager) extractTableColumns(data []interface{}) []string {
	if len(data) == 0 {
		return []string{}
	}
	
	if row, ok := data[0].(map[string]interface{}); ok {
		columns := make([]string, 0, len(row))
		for key := range row {
			columns = append(columns, key)
		}
		return columns
	}
	
	return []string{}
}

func (dm *DashboardManager) parseTimeParam(timeStr string) *time.Time {
	if timeStr == "" {
		return nil
	}
	
	if t, err := time.Parse(time.RFC3339, timeStr); err == nil {
		return &t
	}
	
	return nil
}

func (dm *DashboardManager) parseFilters(query map[string][]string) map[string]interface{} {
	filters := make(map[string]interface{})
	
	for key, values := range query {
		if len(values) == 1 {
			filters[key] = values[0]
		} else if len(values) > 1 {
			filters[key] = values
		}
	}
	
	return filters
}

func (dm *DashboardManager) applyDataFilters(data []interface{}, filters map[string]interface{}) []interface{} {
	if len(filters) == 0 {
		return data
	}
	
	filtered := make([]interface{}, 0)
	
	for _, item := range data {
		if dm.matchesFilters(item, filters) {
			filtered = append(filtered, item)
		}
	}
	
	return filtered
}

func (dm *DashboardManager) matchesFilters(item interface{}, filters map[string]interface{}) bool {
	point, ok := item.(map[string]interface{})
	if !ok {
		return false
	}
	
	for key, expectedValue := range filters {
		if actualValue, exists := point[key]; !exists || actualValue != expectedValue {
			return false
		}
	}
	
	return true
}

func (dm *DashboardManager) aggregateData(data []interface{}, aggregation, granularity string) ([]interface{}, error) {
	// Simple aggregation implementation
	// In a real implementation, this would be more sophisticated
	return data, nil
}

func (dm *DashboardManager) applyFilter(data []interface{}, filter FilterConfig) []interface{} {
	// Apply individual filter
	filtered := make([]interface{}, 0)
	
	for _, item := range data {
		if dm.evaluateFilter(item, filter) {
			filtered = append(filtered, item)
		}
	}
	
	return filtered
}

func (dm *DashboardManager) evaluateFilter(item interface{}, filter FilterConfig) bool {
	point, ok := item.(map[string]interface{})
	if !ok {
		return false
	}
	
	value, exists := point[filter.Field]
	if !exists {
		return false
	}
	
	switch filter.Operator {
	case "eq":
		return value == filter.Value
	case "ne":
		return value != filter.Value
	case "gt":
		return dm.compareValues(value, filter.Value) > 0
	case "gte":
		return dm.compareValues(value, filter.Value) >= 0
	case "lt":
		return dm.compareValues(value, filter.Value) < 0
	case "lte":
		return dm.compareValues(value, filter.Value) <= 0
	default:
		return false
	}
}

func (dm *DashboardManager) compareValues(a, b interface{}) int {
	// Simple comparison - in practice would need more sophisticated type handling
	if af, ok := a.(float64); ok {
		if bf, ok := b.(float64); ok {
			if af < bf {
				return -1
			} else if af > bf {
				return 1
			}
			return 0
		}
	}
	return 0
}

func (dm *DashboardManager) applyGrouping(data []interface{}, groupBy []string) []interface{} {
	// Group data by specified fields
	// Simplified implementation
	return data
}

func (dm *DashboardManager) applySorting(data []interface{}, sorting []SortConfig) []interface{} {
	// Sort data by specified fields
	// Simplified implementation
	return data
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