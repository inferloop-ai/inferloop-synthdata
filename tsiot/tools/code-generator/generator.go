package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v2"
)

type Config struct {
	Version    string                    `yaml:"version"`
	Settings   Settings                  `yaml:"settings"`
	Generators map[string]GeneratorSpec  `yaml:"generators"`
	Variables  map[string]interface{}    `yaml:"variables"`
	TypeMappings TypeMappings            `yaml:"type_mappings"`
	Validation ValidationConfig          `yaml:"validation"`
	Rules      Rules                     `yaml:"rules"`
	Plugins    map[string]PluginConfig   `yaml:"plugins"`
	Hooks      HooksConfig               `yaml:"hooks"`
	CustomTemplates []CustomTemplate     `yaml:"custom_templates"`
}

type Settings struct {
	OutputDirectory   string `yaml:"output_directory"`
	OverwriteExisting bool   `yaml:"overwrite_existing"`
	FormatCode        bool   `yaml:"format_code"`
	AddComments       bool   `yaml:"add_comments"`
	LicenseHeader     string `yaml:"license_header"`
}

type GeneratorSpec struct {
	Enabled   bool                       `yaml:"enabled"`
	Templates []TemplateSpec             `yaml:"templates"`
	Options   map[string]interface{}     `yaml:"options"`
}

type TemplateSpec struct {
	Name          string `yaml:"name"`
	Template      string `yaml:"template"`
	OutputPattern string `yaml:"output_pattern"`
}

type TypeMappings struct {
	ProtoToGo map[string]string `yaml:"proto_to_go"`
	GoToSQL   map[string]string `yaml:"go_to_sql"`
}

type ValidationConfig struct {
	ModelFields []FieldValidation `yaml:"model_fields"`
}

type FieldValidation struct {
	Name  string   `yaml:"name"`
	Rules []string `yaml:"rules"`
}

type Rules struct {
	Naming NamingRules `yaml:"naming"`
	Style  StyleRules  `yaml:"style"`
}

type NamingRules struct {
	Go    GoNaming   `yaml:"go"`
	Files FileNaming `yaml:"files"`
}

type GoNaming struct {
	InterfaceSuffix string `yaml:"interface_suffix"`
	TestSuffix      string `yaml:"test_suffix"`
	MockPrefix      string `yaml:"mock_prefix"`
	PrivatePrefix   string `yaml:"private_prefix"`
}

type FileNaming struct {
	UseSnakeCase     bool   `yaml:"use_snake_case"`
	TestSuffix       string `yaml:"test_suffix"`
	InterfaceSuffix  string `yaml:"interface_suffix"`
}

type StyleRules struct {
	MaxLineLength int  `yaml:"max_line_length"`
	IndentSize    int  `yaml:"indent_size"`
	UseTabs       bool `yaml:"use_tabs"`
	GroupImports  bool `yaml:"group_imports"`
	SortImports   bool `yaml:"sort_imports"`
}

type PluginConfig struct {
	Version string                 `yaml:"version"`
	Options map[string]interface{} `yaml:"options"`
}

type HooksConfig struct {
	PreGenerate  []Hook `yaml:"pre_generate"`
	PostGenerate []Hook `yaml:"post_generate"`
}

type Hook struct {
	Name    string `yaml:"name"`
	Command string `yaml:"command"`
}

type CustomTemplate struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	Template    string `yaml:"template"`
}

type Generator struct {
	config   *Config
	logger   *logrus.Logger
	funcMap  template.FuncMap
	vars     map[string]interface{}
}

type TemplateData struct {
	// Common fields
	Name        string
	Package     string
	Imports     []string
	Comments    string
	Description string
	
	// Type-specific fields
	Fields      []Field
	Methods     []Method
	Operations  []Operation
	Services    []Service
	Resources   []Resource
	
	// Metadata
	Generated   time.Time
	Version     string
	Generator   string
}

type Field struct {
	Name        string
	Type        string
	Tags        string
	Comments    string
	Validation  []string
	JsonTag     string
	DbTag       string
}

type Method struct {
	Name     string
	Params   string
	Returns  string
	Body     string
	Comments string
}

type Operation struct {
	Name   string
	Type   string
	Path   string
	Method string
	Params []Parameter
}

type Parameter struct {
	Name     string
	Type     string
	Required bool
	Location string // path, query, body, header
}

type Service struct {
	Name    string
	Package string
	Methods []Method
}

type Resource struct {
	Name       string
	Operations []Operation
	Model      string
	Package    string
}

func main() {
	var (
		configFile     = flag.String("config", "config.yaml", "Configuration file path")
		generatorType  = flag.String("type", "", "Generator type")
		name           = flag.String("name", "", "Component name")
		resource       = flag.String("resource", "", "Resource name for API generation")
		service        = flag.String("service", "", "Service name for protocol generation")
		outputDir      = flag.String("output", "", "Output directory")
		templateFile   = flag.String("template", "", "Custom template file")
		vars           = flag.String("vars", "{}", "Additional template variables (JSON)")
		dryRun         = flag.Bool("dry-run", false, "Show what would be generated")
		overwrite      = flag.Bool("overwrite", false, "Overwrite existing files")
		verbose        = flag.Bool("verbose", false, "Enable verbose logging")
		validate       = flag.Bool("validate", false, "Validate template syntax")
		listTypes      = flag.Bool("list-types", false, "List available generator types")
	)
	flag.Parse()

	// Setup logging
	logger := logrus.New()
	if *verbose {
		logger.SetLevel(logrus.DebugLevel)
	}

	// Load configuration
	config, err := loadConfig(*configFile)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Override settings from command line
	if *outputDir != "" {
		config.Settings.OutputDirectory = *outputDir
	}
	if *overwrite {
		config.Settings.OverwriteExisting = true
	}

	generator := NewGenerator(config, logger)

	// Handle special commands
	if *listTypes {
		generator.listGeneratorTypes()
		return
	}

	if *validate && *templateFile != "" {
		if err := generator.validateTemplate(*templateFile); err != nil {
			log.Fatalf("Template validation failed: %v", err)
		}
		fmt.Println("Template is valid")
		return
	}

	// Parse additional variables
	additionalVars := make(map[string]interface{})
	if err := json.Unmarshal([]byte(*vars), &additionalVars); err != nil {
		log.Fatalf("Failed to parse vars: %v", err)
	}

	// Merge additional variables
	for k, v := range additionalVars {
		generator.vars[k] = v
	}

	// Set component-specific variables
	if *name != "" {
		generator.vars["Name"] = *name
	}
	if *resource != "" {
		generator.vars["Resource"] = *resource
	}
	if *service != "" {
		generator.vars["Service"] = *service
	}

	// Run pre-generation hooks
	if err := generator.runHooks(config.Hooks.PreGenerate); err != nil {
		log.Fatalf("Pre-generation hooks failed: %v", err)
	}

	// Generate code
	if *templateFile != "" {
		// Custom template generation
		err = generator.generateFromTemplate(*templateFile, generator.vars, *dryRun)
	} else if *generatorType != "" {
		// Type-specific generation
		err = generator.generateByType(*generatorType, generator.vars, *dryRun)
	} else {
		// Generate all enabled generators
		err = generator.generateAll(*dryRun)
	}

	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Run post-generation hooks
	if !*dryRun {
		if err := generator.runHooks(config.Hooks.PostGenerate); err != nil {
			logger.WithError(err).Warn("Post-generation hooks failed")
		}
	}

	logger.Info("Code generation completed successfully")
}

func NewGenerator(config *Config, logger *logrus.Logger) *Generator {
	g := &Generator{
		config: config,
		logger: logger,
		vars:   make(map[string]interface{}),
	}

	// Copy config variables
	for k, v := range config.Variables {
		g.vars[k] = v
	}

	// Add metadata
	g.vars["Generated"] = time.Now()
	g.vars["Version"] = config.Version
	g.vars["Generator"] = "TSIoT Code Generator"

	// Setup template functions
	g.funcMap = template.FuncMap{
		"camelCase":   camelCase,
		"snakeCase":   snakeCase,
		"kebabCase":   kebabCase,
		"pascalCase":  pascalCase,
		"pluralize":   pluralize,
		"singularize": singularize,
		"lower":       strings.ToLower,
		"upper":       strings.ToUpper,
		"title":       strings.Title,
		"contains":    strings.Contains,
		"hasPrefix":   strings.HasPrefix,
		"hasSuffix":   strings.HasSuffix,
		"replace":     strings.ReplaceAll,
		"split":       strings.Split,
		"join":        strings.Join,
		"quote":       fmt.Sprintf("%q", ""),
		"add":         add,
		"sub":         sub,
		"mul":         mul,
		"div":         div,
		"mod":         mod,
		"eq":          eq,
		"ne":          ne,
		"lt":          lt,
		"le":          le,
		"gt":          gt,
		"ge":          ge,
		"and":         and,
		"or":          or,
		"not":         not,
	}

	return g
}

func (g *Generator) generateAll(dryRun bool) error {
	for generatorType, spec := range g.config.Generators {
		if !spec.Enabled {
			g.logger.WithField("type", generatorType).Debug("Generator disabled, skipping")
			continue
		}

		g.logger.WithField("type", generatorType).Info("Running generator")
		
		if err := g.generateByType(generatorType, g.vars, dryRun); err != nil {
			return fmt.Errorf("generator %s failed: %w", generatorType, err)
		}
	}
	return nil
}

func (g *Generator) generateByType(generatorType string, vars map[string]interface{}, dryRun bool) error {
	spec, exists := g.config.Generators[generatorType]
	if !exists {
		return fmt.Errorf("unknown generator type: %s", generatorType)
	}

	if !spec.Enabled {
		return fmt.Errorf("generator %s is disabled", generatorType)
	}

	// Merge generator-specific options
	generatorVars := make(map[string]interface{})
	for k, v := range vars {
		generatorVars[k] = v
	}
	for k, v := range spec.Options {
		generatorVars[k] = v
	}

	// Generate from each template
	for _, templateSpec := range spec.Templates {
		if err := g.generateFromTemplateSpec(templateSpec, generatorVars, dryRun); err != nil {
			return fmt.Errorf("template %s failed: %w", templateSpec.Name, err)
		}
	}

	return nil
}

func (g *Generator) generateFromTemplateSpec(spec TemplateSpec, vars map[string]interface{}, dryRun bool) error {
	// Read template
	templateContent, err := ioutil.ReadFile(spec.Template)
	if err != nil {
		return fmt.Errorf("failed to read template: %w", err)
	}

	// Parse template
	tmpl, err := template.New(spec.Name).Funcs(g.funcMap).Parse(string(templateContent))
	if err != nil {
		return fmt.Errorf("failed to parse template: %w", err)
	}

	// Prepare template data
	data := g.prepareTemplateData(vars)

	// Execute template
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	// Generate output path
	outputPath, err := g.generateOutputPath(spec.OutputPattern, data)
	if err != nil {
		return fmt.Errorf("failed to generate output path: %w", err)
	}

	// Format code if it's a Go file
	content := buf.Bytes()
	if strings.HasSuffix(outputPath, ".go") && g.config.Settings.FormatCode {
		formatted, err := format.Source(content)
		if err != nil {
			g.logger.WithError(err).Warn("Failed to format Go code, using unformatted")
		} else {
			content = formatted
		}
	}

	if dryRun {
		g.logger.WithFields(logrus.Fields{
			"template": spec.Name,
			"output":   outputPath,
			"size":     len(content),
		}).Info("DRY RUN - Would generate file")
		return nil
	}

	// Write file
	return g.writeFile(outputPath, content)
}

func (g *Generator) generateFromTemplate(templateFile string, vars map[string]interface{}, dryRun bool) error {
	// Read template
	templateContent, err := ioutil.ReadFile(templateFile)
	if err != nil {
		return fmt.Errorf("failed to read template: %w", err)
	}

	// Parse template
	tmpl, err := template.New(filepath.Base(templateFile)).Funcs(g.funcMap).Parse(string(templateContent))
	if err != nil {
		return fmt.Errorf("failed to parse template: %w", err)
	}

	// Prepare template data
	data := g.prepareTemplateData(vars)

	// Execute template
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	content := buf.Bytes()

	if dryRun {
		fmt.Printf("Generated content from %s:\n%s\n", templateFile, string(content))
		return nil
	}

	// Write to stdout if no output specified
	fmt.Print(string(content))
	return nil
}

func (g *Generator) prepareTemplateData(vars map[string]interface{}) TemplateData {
	data := TemplateData{
		Generated: time.Now(),
		Version:   g.config.Version,
		Generator: "TSIoT Code Generator",
	}

	// Extract common fields from vars
	if name, ok := vars["Name"].(string); ok {
		data.Name = name
	}
	if pkg, ok := vars["Package"].(string); ok {
		data.Package = pkg
	}
	if desc, ok := vars["Description"].(string); ok {
		data.Description = desc
	}

	// Extract imports
	if imports, ok := vars["Imports"].([]interface{}); ok {
		for _, imp := range imports {
			if impStr, ok := imp.(string); ok {
				data.Imports = append(data.Imports, impStr)
			}
		}
	}

	// Extract fields
	if fields, ok := vars["Fields"].([]interface{}); ok {
		for _, field := range fields {
			if fieldMap, ok := field.(map[string]interface{}); ok {
				f := Field{}
				if name, ok := fieldMap["name"].(string); ok {
					f.Name = name
				}
				if typ, ok := fieldMap["type"].(string); ok {
					f.Type = typ
				}
				if tags, ok := fieldMap["tags"].([]interface{}); ok {
					var tagStrs []string
					for _, tag := range tags {
						if tagStr, ok := tag.(string); ok {
							tagStrs = append(tagStrs, tagStr)
						}
					}
					f.Tags = strings.Join(tagStrs, " ")
				}
				data.Fields = append(data.Fields, f)
			}
		}
	}

	// Extract methods
	if methods, ok := vars["Methods"].([]interface{}); ok {
		for _, method := range methods {
			if methodMap, ok := method.(map[string]interface{}); ok {
				m := Method{}
				if name, ok := methodMap["name"].(string); ok {
					m.Name = name
				}
				if params, ok := methodMap["params"].(string); ok {
					m.Params = params
				}
				if returns, ok := methodMap["returns"].(string); ok {
					m.Returns = returns
				}
				if body, ok := methodMap["body"].(string); ok {
					m.Body = body
				}
				data.Methods = append(data.Methods, m)
			}
		}
	}

	return data
}

func (g *Generator) generateOutputPath(pattern string, data TemplateData) (string, error) {
	// Parse output pattern as template
	tmpl, err := template.New("output_path").Funcs(g.funcMap).Parse(pattern)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", err
	}

	outputPath := buf.String()
	
	// Ensure output directory exists
	fullPath := filepath.Join(g.config.Settings.OutputDirectory, outputPath)
	dir := filepath.Dir(fullPath)
	
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	return fullPath, nil
}

func (g *Generator) writeFile(path string, content []byte) error {
	// Check if file exists and overwrite is disabled
	if !g.config.Settings.OverwriteExisting {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("file %s already exists and overwrite is disabled", path)
		}
	}

	// Add license header if configured
	if g.config.Settings.LicenseHeader != "" && strings.HasSuffix(path, ".go") {
		content = append([]byte(g.config.Settings.LicenseHeader+"\n\n"), content...)
	}

	g.logger.WithFields(logrus.Fields{
		"path": path,
		"size": len(content),
	}).Info("Writing file")

	return ioutil.WriteFile(path, content, 0644)
}

func (g *Generator) validateTemplate(templateFile string) error {
	templateContent, err := ioutil.ReadFile(templateFile)
	if err != nil {
		return fmt.Errorf("failed to read template: %w", err)
	}

	_, err = template.New(filepath.Base(templateFile)).Funcs(g.funcMap).Parse(string(templateContent))
	return err
}

func (g *Generator) listGeneratorTypes() {
	fmt.Println("Available generator types:")
	for generatorType, spec := range g.config.Generators {
		status := "disabled"
		if spec.Enabled {
			status = "enabled"
		}
		fmt.Printf("  %-15s (%s) - %d templates\n", generatorType, status, len(spec.Templates))
	}
}

func (g *Generator) runHooks(hooks []Hook) error {
	for _, hook := range hooks {
		g.logger.WithField("hook", hook.Name).Info("Running hook")

		// Replace variables in command
		command := g.replaceVariables(hook.Command)
		
		// Execute command
		cmd := exec.Command("sh", "-c", command)
		output, err := cmd.CombinedOutput()
		
		if err != nil {
			g.logger.WithFields(logrus.Fields{
				"hook":   hook.Name,
				"output": string(output),
			}).WithError(err).Error("Hook failed")
			return fmt.Errorf("hook %s failed: %w", hook.Name, err)
		}

		if len(output) > 0 {
			g.logger.WithField("output", string(output)).Debug("Hook output")
		}
	}
	return nil
}

func (g *Generator) replaceVariables(text string) string {
	// Simple variable replacement
	text = strings.ReplaceAll(text, "{{.OutputDirectory}}", g.config.Settings.OutputDirectory)
	
	// Add more variable replacements as needed
	for k, v := range g.vars {
		if str, ok := v.(string); ok {
			text = strings.ReplaceAll(text, fmt.Sprintf("{{.%s}}", k), str)
		}
	}
	
	return text
}

// Helper functions

func loadConfig(path string) (*Config, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	if strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
		err = yaml.Unmarshal(data, &config)
	} else {
		err = json.Unmarshal(data, &config)
	}

	return &config, err
}

// String transformation functions

func camelCase(s string) string {
	words := strings.FieldsFunc(s, func(c rune) bool {
		return c == '_' || c == '-' || c == ' '
	})
	
	if len(words) == 0 {
		return s
	}
	
	result := strings.ToLower(words[0])
	for i := 1; i < len(words); i++ {
		result += strings.Title(strings.ToLower(words[i]))
	}
	return result
}

func snakeCase(s string) string {
	var result []rune
	for i, r := range s {
		if i > 0 && ('A' <= r && r <= 'Z') {
			result = append(result, '_')
		}
		result = append(result, ('a'-'A')|r)
	}
	return string(result)
}

func kebabCase(s string) string {
	return strings.ReplaceAll(snakeCase(s), "_", "-")
}

func pascalCase(s string) string {
	camel := camelCase(s)
	if len(camel) > 0 {
		return strings.ToUpper(camel[:1]) + camel[1:]
	}
	return camel
}

func pluralize(s string) string {
	// Simple pluralization rules
	if strings.HasSuffix(s, "y") {
		return s[:len(s)-1] + "ies"
	}
	if strings.HasSuffix(s, "s") || strings.HasSuffix(s, "x") || strings.HasSuffix(s, "z") {
		return s + "es"
	}
	return s + "s"
}

func singularize(s string) string {
	// Simple singularization rules
	if strings.HasSuffix(s, "ies") {
		return s[:len(s)-3] + "y"
	}
	if strings.HasSuffix(s, "es") {
		return s[:len(s)-2]
	}
	if strings.HasSuffix(s, "s") {
		return s[:len(s)-1]
	}
	return s
}

// Math functions for templates

func add(a, b int) int { return a + b }
func sub(a, b int) int { return a - b }
func mul(a, b int) int { return a * b }
func div(a, b int) int { 
	if b == 0 { return 0 }
	return a / b 
}
func mod(a, b int) int { 
	if b == 0 { return 0 }
	return a % b 
}

// Comparison functions

func eq(a, b interface{}) bool { return a == b }
func ne(a, b interface{}) bool { return a != b }
func lt(a, b int) bool { return a < b }
func le(a, b int) bool { return a <= b }
func gt(a, b int) bool { return a > b }
func ge(a, b int) bool { return a >= b }

// Logical functions

func and(a, b bool) bool { return a && b }
func or(a, b bool) bool { return a || b }
func not(a bool) bool { return !a }