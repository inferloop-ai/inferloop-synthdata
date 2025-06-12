package dashboards

import (
	"encoding/json"
	"fmt"
	"time"
)

// GrafanaDashboard represents a Grafana dashboard configuration
type GrafanaDashboard struct {
	ID          int                    `json:"id"`
	UID         string                 `json:"uid"`
	Title       string                 `json:"title"`
	Tags        []string               `json:"tags"`
	Style       string                 `json:"style"`
	Timezone    string                 `json:"timezone"`
	Editable    bool                   `json:"editable"`
	HideControls bool                  `json:"hideControls"`
	Time        TimeConfig             `json:"time"`
	Timepicker  TimepickerConfig       `json:"timepicker"`
	Templating  TemplatingConfig       `json:"templating"`
	Annotations AnnotationsConfig      `json:"annotations"`
	Refresh     string                 `json:"refresh"`
	SchemaVersion int                  `json:"schemaVersion"`
	Version     int                    `json:"version"`
	Panels      []Panel                `json:"panels"`
}

// TimeConfig configures dashboard time range
type TimeConfig struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// TimepickerConfig configures the time picker
type TimepickerConfig struct {
	RefreshIntervals []string `json:"refresh_intervals"`
	TimeOptions      []string `json:"time_options"`
}

// TemplatingConfig configures dashboard variables
type TemplatingConfig struct {
	List []Variable `json:"list"`
}

// Variable represents a dashboard variable
type Variable struct {
	Name        string            `json:"name"`
	Type        string            `json:"type"`
	Label       string            `json:"label"`
	Query       string            `json:"query"`
	Datasource  string            `json:"datasource"`
	Refresh     int               `json:"refresh"`
	Options     []VariableOption  `json:"options"`
	Current     VariableOption    `json:"current"`
	Hide        int               `json:"hide"`
	IncludeAll  bool              `json:"includeAll"`
	Multi       bool              `json:"multi"`
	AllValue    string            `json:"allValue"`
	Tags        []string          `json:"tags"`
	TagsQuery   string            `json:"tagsQuery"`
	Definition  string            `json:"definition"`
}

// VariableOption represents a variable option
type VariableOption struct {
	Text     string `json:"text"`
	Value    string `json:"value"`
	Selected bool   `json:"selected"`
}

// AnnotationsConfig configures dashboard annotations
type AnnotationsConfig struct {
	List []Annotation `json:"list"`
}

// Annotation represents a dashboard annotation
type Annotation struct {
	Name        string            `json:"name"`
	Datasource  string            `json:"datasource"`
	Enable      bool              `json:"enable"`
	Hide        bool              `json:"hide"`
	IconColor   string            `json:"iconColor"`
	Query       string            `json:"query"`
	ShowLine    bool              `json:"showLine"`
	Step        string            `json:"step"`
	Tags        []string          `json:"tags"`
	Type        string            `json:"type"`
	TagKeys     string            `json:"tagKeys"`
	TitleFormat string            `json:"titleFormat"`
	TextFormat  string            `json:"textFormat"`
}

// Panel represents a dashboard panel
type Panel struct {
	ID              int                    `json:"id"`
	Title           string                 `json:"title"`
	Type            string                 `json:"type"`
	Datasource      string                 `json:"datasource"`
	GridPos         GridPos                `json:"gridPos"`
	Targets         []Target               `json:"targets"`
	XAxis           AxisConfig             `json:"xAxis,omitempty"`
	YAxes           []AxisConfig           `json:"yAxes,omitempty"`
	Legend          LegendConfig           `json:"legend,omitempty"`
	Tooltip         TooltipConfig          `json:"tooltip,omitempty"`
	Options         map[string]interface{} `json:"options,omitempty"`
	FieldConfig     FieldConfig            `json:"fieldConfig,omitempty"`
	Alert           *AlertConfig           `json:"alert,omitempty"`
	Transparent     bool                   `json:"transparent"`
	Description     string                 `json:"description"`
}

// GridPos defines panel position and size
type GridPos struct {
	H int `json:"h"`
	W int `json:"w"`
	X int `json:"x"`
	Y int `json:"y"`
}

// Target represents a query target
type Target struct {
	Expr           string            `json:"expr"`
	Format         string            `json:"format"`
	Interval       string            `json:"interval"`
	IntervalFactor int               `json:"intervalFactor"`
	LegendFormat   string            `json:"legendFormat"`
	RefID          string            `json:"refId"`
	Step           int               `json:"step"`
	Datasource     string            `json:"datasource"`
	Hide           bool              `json:"hide"`
	Exemplar       bool              `json:"exemplar"`
}

// AxisConfig configures chart axes
type AxisConfig struct {
	Show     bool   `json:"show"`
	Label    string `json:"label"`
	LogBase  int    `json:"logBase"`
	Max      string `json:"max"`
	Min      string `json:"min"`
	Unit     string `json:"unit"`
	Decimals int    `json:"decimals"`
}

// LegendConfig configures chart legend
type LegendConfig struct {
	Show         bool   `json:"show"`
	Values       bool   `json:"values"`
	Min          bool   `json:"min"`
	Max          bool   `json:"max"`
	Current      bool   `json:"current"`
	Total        bool   `json:"total"`
	Avg          bool   `json:"avg"`
	AlignAsTable bool   `json:"alignAsTable"`
	RightSide    bool   `json:"rightSide"`
	SideWidth    int    `json:"sideWidth"`
}

// TooltipConfig configures chart tooltip
type TooltipConfig struct {
	Shared    bool   `json:"shared"`
	Sort      int    `json:"sort"`
	ValueType string `json:"value_type"`
}

// FieldConfig configures field properties
type FieldConfig struct {
	Defaults  FieldDefaults    `json:"defaults"`
	Overrides []FieldOverride  `json:"overrides"`
}

// FieldDefaults defines default field configuration
type FieldDefaults struct {
	Color      ColorConfig      `json:"color"`
	Custom     CustomConfig     `json:"custom"`
	Mappings   []ValueMapping   `json:"mappings"`
	Thresholds ThresholdConfig  `json:"thresholds"`
	Unit       string           `json:"unit"`
	Min        *float64         `json:"min,omitempty"`
	Max        *float64         `json:"max,omitempty"`
	Decimals   *int             `json:"decimals,omitempty"`
}

// FieldOverride defines field-specific overrides
type FieldOverride struct {
	Matcher    FieldMatcher           `json:"matcher"`
	Properties []FieldProperty        `json:"properties"`
}

// FieldMatcher defines field matching criteria
type FieldMatcher struct {
	ID      string `json:"id"`
	Options string `json:"options"`
}

// FieldProperty defines a field property override
type FieldProperty struct {
	ID    string      `json:"id"`
	Value interface{} `json:"value"`
}

// ColorConfig defines color settings
type ColorConfig struct {
	Mode       string `json:"mode"`
	FixedColor string `json:"fixedColor,omitempty"`
	SeriesBy   string `json:"seriesBy,omitempty"`
}

// CustomConfig defines custom panel configuration
type CustomConfig struct {
	DrawStyle         string  `json:"drawStyle,omitempty"`
	LineInterpolation string  `json:"lineInterpolation,omitempty"`
	LineWidth         int     `json:"lineWidth,omitempty"`
	FillOpacity       int     `json:"fillOpacity,omitempty"`
	GradientMode      string  `json:"gradientMode,omitempty"`
	SpanNulls         bool    `json:"spanNulls,omitempty"`
	ShowPoints        string  `json:"showPoints,omitempty"`
	PointSize         int     `json:"pointSize,omitempty"`
	Stacking          Stacking `json:"stacking,omitempty"`
}

// Stacking defines stacking configuration
type Stacking struct {
	Group string `json:"group"`
	Mode  string `json:"mode"`
}

// ValueMapping defines value mappings
type ValueMapping struct {
	Type    string                 `json:"type"`
	Options map[string]interface{} `json:"options"`
}

// ThresholdConfig defines threshold configuration
type ThresholdConfig struct {
	Mode  string      `json:"mode"`
	Steps []Threshold `json:"steps"`
}

// Threshold defines a threshold step
type Threshold struct {
	Color string   `json:"color"`
	Value *float64 `json:"value"`
}

// AlertConfig defines panel alert configuration
type AlertConfig struct {
	Name         string           `json:"name"`
	Message      string           `json:"message"`
	Frequency    string           `json:"frequency"`
	Conditions   []AlertCondition `json:"conditions"`
	ExecutionErrorState string    `json:"executionErrorState"`
	NoDataState  string           `json:"noDataState"`
	For          string           `json:"for"`
}

// AlertCondition defines alert condition
type AlertCondition struct {
	Query      QueryCondition     `json:"query"`
	Reducer    ReducerCondition   `json:"reducer"`
	Evaluator  EvaluatorCondition `json:"evaluator"`
}

// QueryCondition defines query condition
type QueryCondition struct {
	QueryType string `json:"queryType"`
	RefID     string `json:"refId"`
}

// ReducerCondition defines reducer condition
type ReducerCondition struct {
	Type   string                 `json:"type"`
	Params []interface{}          `json:"params"`
}

// EvaluatorCondition defines evaluator condition
type EvaluatorCondition struct {
	Params []float64 `json:"params"`
	Type   string    `json:"type"`
}

// CreateTSIOTDashboard creates the main TSIOT monitoring dashboard
func CreateTSIOTDashboard() *GrafanaDashboard {
	dashboard := &GrafanaDashboard{
		UID:           "tsiot-main",
		Title:         "TSIOT - Time Series IoT Platform",
		Tags:          []string{"tsiot", "monitoring", "timeseries"},
		Style:         "dark",
		Timezone:      "browser",
		Editable:      true,
		HideControls:  false,
		SchemaVersion: 27,
		Version:       1,
		Time: TimeConfig{
			From: "now-1h",
			To:   "now",
		},
		Timepicker: TimepickerConfig{
			RefreshIntervals: []string{"5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"},
			TimeOptions:      []string{"5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"},
		},
		Refresh: "30s",
		Templating: TemplatingConfig{
			List: []Variable{
				{
					Name:       "instance",
					Type:       "query",
					Label:      "Instance",
					Query:      "label_values(tsiot_server_http_requests_total, instance)",
					Datasource: "prometheus",
					Refresh:    1,
					Multi:      true,
					IncludeAll: true,
					AllValue:   ".*",
				},
			},
		},
		Panels: createMainDashboardPanels(),
	}

	return dashboard
}

// createMainDashboardPanels creates panels for the main dashboard
func createMainDashboardPanels() []Panel {
	panels := []Panel{
		// System Overview Row
		{
			ID:      1,
			Title:   "System Overview",
			Type:    "row",
			GridPos: GridPos{H: 1, W: 24, X: 0, Y: 0},
		},
		
		// HTTP Requests
		{
			ID:         2,
			Title:      "HTTP Requests Rate",
			Type:       "stat",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 6, X: 0, Y: 1},
			Targets: []Target{
				{
					Expr:         "rate(tsiot_server_http_requests_total[$__rate_interval])",
					RefID:        "A",
					LegendFormat: "{{method}} {{path}}",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "reqps",
					Color: ColorConfig{
						Mode: "palette-classic",
					},
				},
			},
		},
		
		// Generation Requests
		{
			ID:         3,
			Title:      "Generation Requests",
			Type:       "timeseries",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 6, X: 6, Y: 1},
			Targets: []Target{
				{
					Expr:         "rate(tsiot_server_generation_requests_total[$__rate_interval])",
					RefID:        "A",
					LegendFormat: "{{generator}} - {{status}}",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "reqps",
					Custom: CustomConfig{
						DrawStyle:         "line",
						LineInterpolation: "linear",
						LineWidth:         1,
						FillOpacity:       10,
						SpanNulls:         false,
					},
				},
			},
		},
		
		// Active Generations
		{
			ID:         4,
			Title:      "Active Generations",
			Type:       "gauge",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 6, X: 12, Y: 1},
			Targets: []Target{
				{
					Expr:  "tsiot_server_generation_active",
					RefID: "A",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "short",
					Thresholds: ThresholdConfig{
						Mode: "absolute",
						Steps: []Threshold{
							{Color: "green", Value: nil},
							{Color: "yellow", Value: float64Ptr(50)},
							{Color: "red", Value: float64Ptr(100)},
						},
					},
				},
			},
		},
		
		// System Health
		{
			ID:         5,
			Title:      "System Health",
			Type:       "stat",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 6, X: 18, Y: 1},
			Targets: []Target{
				{
					Expr:  "tsiot_server_health_status",
					RefID: "A",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Mappings: []ValueMapping{
						{
							Type: "value",
							Options: map[string]interface{}{
								"0": map[string]interface{}{"text": "Unhealthy", "color": "red"},
								"1": map[string]interface{}{"text": "Healthy", "color": "green"},
								"2": map[string]interface{}{"text": "Degraded", "color": "yellow"},
							},
						},
					},
				},
			},
		},
		
		// Performance Row
		{
			ID:      6,
			Title:   "Performance Metrics",
			Type:    "row",
			GridPos: GridPos{H: 1, W: 24, X: 0, Y: 9},
		},
		
		// Request Duration
		{
			ID:         7,
			Title:      "Request Duration",
			Type:       "timeseries",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 12, X: 0, Y: 10},
			Targets: []Target{
				{
					Expr:         "histogram_quantile(0.50, rate(tsiot_server_http_request_duration_seconds_bucket[$__rate_interval]))",
					RefID:        "A",
					LegendFormat: "50th percentile",
				},
				{
					Expr:         "histogram_quantile(0.95, rate(tsiot_server_http_request_duration_seconds_bucket[$__rate_interval]))",
					RefID:        "B",
					LegendFormat: "95th percentile",
				},
				{
					Expr:         "histogram_quantile(0.99, rate(tsiot_server_http_request_duration_seconds_bucket[$__rate_interval]))",
					RefID:        "C",
					LegendFormat: "99th percentile",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "s",
				},
			},
		},
		
		// Generation Duration
		{
			ID:         8,
			Title:      "Generation Duration",
			Type:       "timeseries",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 12, X: 12, Y: 10},
			Targets: []Target{
				{
					Expr:         "histogram_quantile(0.95, rate(tsiot_server_generation_duration_seconds_bucket[$__rate_interval]))",
					RefID:        "A",
					LegendFormat: "{{generator}} - 95th percentile",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "s",
				},
			},
		},
		
		// Quality Metrics Row
		{
			ID:      9,
			Title:   "Quality Metrics",
			Type:    "row",
			GridPos: GridPos{H: 1, W: 24, X: 0, Y: 18},
		},
		
		// Validation Quality Score
		{
			ID:         10,
			Title:      "Validation Quality Score",
			Type:       "timeseries",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 12, X: 0, Y: 19},
			Targets: []Target{
				{
					Expr:         "tsiot_server_validation_quality_score",
					RefID:        "A",
					LegendFormat: "{{validator}}",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "percentunit",
					Min:  float64Ptr(0),
					Max:  float64Ptr(1),
				},
			},
		},
		
		// Error Rate
		{
			ID:         11,
			Title:      "Error Rate",
			Type:       "timeseries",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 12, X: 12, Y: 19},
			Targets: []Target{
				{
					Expr:         "rate(tsiot_server_errors_total[$__rate_interval])",
					RefID:        "A",
					LegendFormat: "{{component}} - {{type}}",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "reqps",
				},
			},
		},
	}

	return panels
}

// CreatePerformanceDashboard creates a performance-focused dashboard
func CreatePerformanceDashboard() *GrafanaDashboard {
	dashboard := &GrafanaDashboard{
		UID:           "tsiot-performance",
		Title:         "TSIOT - Performance Dashboard",
		Tags:          []string{"tsiot", "performance", "monitoring"},
		Style:         "dark",
		Timezone:      "browser",
		Editable:      true,
		SchemaVersion: 27,
		Version:       1,
		Time: TimeConfig{
			From: "now-6h",
			To:   "now",
		},
		Refresh: "30s",
		Panels:  createPerformancePanels(),
	}

	return dashboard
}

// createPerformancePanels creates panels for the performance dashboard
func createPerformancePanels() []Panel {
	return []Panel{
		// CPU Usage
		{
			ID:         1,
			Title:      "CPU Usage",
			Type:       "timeseries",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 12, X: 0, Y: 0},
			Targets: []Target{
				{
					Expr:         "tsiot_system_cpu_usage_percent",
					RefID:        "A",
					LegendFormat: "CPU Usage %",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "percent",
					Max:  float64Ptr(100),
				},
			},
		},
		
		// Memory Usage
		{
			ID:         2,
			Title:      "Memory Usage",
			Type:       "timeseries",
			Datasource: "prometheus",
			GridPos:    GridPos{H: 8, W: 12, X: 12, Y: 0},
			Targets: []Target{
				{
					Expr:         "tsiot_system_memory_usage_bytes",
					RefID:        "A",
					LegendFormat: "Memory Usage",
				},
			},
			FieldConfig: FieldConfig{
				Defaults: FieldDefaults{
					Unit: "bytes",
				},
			},
		},
	}
}

// ToJSON converts the dashboard to JSON
func (d *GrafanaDashboard) ToJSON() ([]byte, error) {
	return json.MarshalIndent(d, "", "  ")
}

// Helper function to create float64 pointer
func float64Ptr(f float64) *float64 {
	return &f
}

// CreateDashboardFromTemplate creates a dashboard from a template
func CreateDashboardFromTemplate(templateName string, config map[string]interface{}) (*GrafanaDashboard, error) {
	switch templateName {
	case "main":
		return CreateTSIOTDashboard(), nil
	case "performance":
		return CreatePerformanceDashboard(), nil
	default:
		return nil, fmt.Errorf("unknown dashboard template: %s", templateName)
	}
}