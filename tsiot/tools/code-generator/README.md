# Code Generator Tool

This tool provides automated code generation capabilities for the TSIoT platform, helping to generate boilerplate code, maintain consistency, and accelerate development.

## Overview

The code generator supports creation of:
- Data models and validators
- Protocol handlers (gRPC, HTTP, MQTT, Kafka)
- Synthetic data generators
- Privacy transformation modules
- API handlers and routers
- Storage repositories
- Tests and benchmarks
- Documentation

## Features

- **Template-based Generation**: Flexible Jinja2-style templates
- **Multi-format Support**: Go, Protocol Buffers, SQL, Markdown
- **Custom Plugins**: Extensible plugin architecture
- **Configuration-driven**: YAML/JSON configuration files
- **Code Formatting**: Automatic gofmt and goimports
- **Documentation**: Auto-generated API docs and README files

## Quick Start

### Basic Usage

```bash
# Generate from configuration file
./generator -config config.yaml

# Generate specific component
./generator -type model -name TimeSeries

# Generate API handlers
./generator -type api -resource sensors

# Generate gRPC service
./generator -type protocol -service DataIngestion
```

### Command Line Options

```bash
Flags:
  -config string
        Configuration file path
  -type string
        Generator type: model, protocol, api, storage, test, docs
  -name string
        Component name to generate
  -resource string
        Resource name for API generation
  -service string
        Service name for protocol generation
  -output string
        Output directory (default: "./generated")
  -template string
        Custom template file
  -vars string
        Additional template variables (JSON)
  -dry-run
        Show what would be generated without creating files
  -overwrite
        Overwrite existing files
  -verbose
        Enable verbose logging
```

## Configuration

### Basic Configuration

```yaml
version: "1.0"

settings:
  output_directory: "./generated"
  overwrite_existing: false
  format_code: true

generators:
  model:
    enabled: true
    templates:
      - name: "model"
        template: "templates/model.go.tmpl"
        output_pattern: "pkg/models/{{.Name}}.go"

variables:
  project_name: "TSIoT"
  base_package: "github.com/inferloop/tsiot"
```

### Advanced Configuration

See `config.yaml` for comprehensive configuration options including:
- Multiple generator types
- Template customization
- Type mappings
- Validation rules
- Plugin configuration
- Pre/post-generation hooks

## Generator Types

### 1. Model Generator

Generates data models with validation and serialization:

```bash
# Generate TimeSeries model
./generator -type model -name TimeSeries -vars '{
  "fields": [
    {"name": "ID", "type": "string", "tags": ["required", "uuid"]},
    {"name": "Name", "type": "string", "tags": ["required", "min=3"]},
    {"name": "Points", "type": "[]DataPoint", "tags": ["required"]}
  ]
}'
```

Generated output:
```go
type TimeSeries struct {
    ID     string      `json:"id" db:"id" validate:"required,uuid"`
    Name   string      `json:"name" db:"name" validate:"required,min=3"`
    Points []DataPoint `json:"points" db:"-" validate:"required"`
}
```

### 2. Protocol Generator

Generates gRPC services, HTTP handlers, MQTT subscribers:

```bash
# Generate gRPC service
./generator -type protocol -service DataIngestion -vars '{
  "methods": [
    {"name": "IngestData", "input": "IngestRequest", "output": "IngestResponse"},
    {"name": "QueryData", "input": "QueryRequest", "output": "QueryResponse"}
  ]
}'

# Generate MQTT handler
./generator -type protocol -name SensorData -vars '{
  "topics": ["sensors/+/temperature", "sensors/+/humidity"],
  "qos": 1
}'
```

### 3. Data Generator

Creates synthetic data generators:

```bash
# Generate pattern-based generator
./generator -type data_generator -name SeasonalPattern -vars '{
  "patterns": ["sine", "trend", "noise"],
  "parameters": {
    "amplitude": "float64",
    "frequency": "float64",
    "noise_level": "float64"
  }
}'
```

### 4. Privacy Module Generator

Generates privacy transformation modules:

```bash
# Generate k-anonymity transformer
./generator -type privacy -method k_anonymity -vars '{
  "quasi_identifiers": ["age", "zipcode"],
  "sensitive_attributes": ["income"]
}'

# Generate differential privacy module
./generator -type privacy -method differential_privacy -vars '{
  "epsilon": 1.0,
  "delta": 1e-5,
  "mechanisms": ["laplace", "gaussian"]
}'
```

### 5. API Generator

Creates REST API handlers and routers:

```bash
# Generate CRUD API for sensors
./generator -type api -resource sensors -vars '{
  "operations": ["create", "read", "update", "delete", "list"],
  "middleware": ["auth", "rate_limit", "logging"]
}'
```

### 6. Storage Generator

Generates repository patterns for different backends:

```bash
# Generate PostgreSQL repository
./generator -type storage -backend postgres -entity TimeSeries

# Generate MongoDB repository
./generator -type storage -backend mongodb -entity TimeSeries
```

### 7. Test Generator

Creates unit tests, integration tests, and benchmarks:

```bash
# Generate unit tests
./generator -type test -name TimeSeriesService -vars '{
  "test_cases": [
    {"name": "TestCreateTimeSeries", "setup": "mock_db"},
    {"name": "TestQueryTimeSeries", "setup": "test_data"}
  ]
}'

# Generate benchmark tests
./generator -type test -name DataIngestion -template benchmark -vars '{
  "benchmarks": ["BenchmarkSingleInsert", "BenchmarkBatchInsert"]
}'
```

## Templates

### Template Structure

Templates use Go's text/template syntax with additional functions:

```go
package {{.Package}}

import (
    {{range .Imports}}
    "{{.}}"
    {{end}}
)

{{if .Comments}}
// {{.Name}} represents {{.Description}}
{{end}}
type {{.Name}} struct {
    {{range .Fields}}
    {{.Name}} {{.Type}} `{{.Tags}}`
    {{end}}
}

{{range .Methods}}
func ({{$.Receiver}} *{{$.Name}}) {{.Name}}({{.Params}}) {{.Returns}} {
    {{.Body}}
}
{{end}}
```

### Custom Templates

Create custom templates for specific needs:

```yaml
custom_templates:
  - name: "event_handler"
    template: |
      type {{.Name}}Handler struct {
          eventBus EventBus
          logger   *logrus.Logger
      }
      
      func (h *{{.Name}}Handler) Handle(event Event) error {
          // Custom event handling logic
          return nil
      }
```

### Template Functions

Available template functions:

- `camelCase`: Convert to camelCase
- `snakeCase`: Convert to snake_case
- `kebabCase`: Convert to kebab-case
- `pluralize`: Pluralize word
- `singularize`: Singularize word
- `lower`: Convert to lowercase
- `upper`: Convert to uppercase
- `title`: Convert to title case

## Examples

### Example 1: Generate Complete Service

```bash
# Generate model
./generator -type model -name SensorReading -vars '{
  "fields": [
    {"name": "ID", "type": "string", "tags": ["json:\"id\"", "db:\"id\""]},
    {"name": "SensorID", "type": "string", "tags": ["json:\"sensor_id\""]},
    {"name": "Value", "type": "float64", "tags": ["json:\"value\""]},
    {"name": "Timestamp", "type": "time.Time", "tags": ["json:\"timestamp\""]}
  ]
}'

# Generate repository
./generator -type storage -backend postgres -entity SensorReading

# Generate API handlers
./generator -type api -resource sensor-readings

# Generate tests
./generator -type test -name SensorReadingService
```

### Example 2: Protocol Buffers and gRPC

```bash
# Create proto file first
cat > sensor_service.proto << EOF
syntax = "proto3";

package sensor;

service SensorService {
  rpc CreateReading(CreateReadingRequest) returns (CreateReadingResponse);
  rpc GetReadings(GetReadingsRequest) returns (GetReadingsResponse);
}

message CreateReadingRequest {
  string sensor_id = 1;
  double value = 2;
  int64 timestamp = 3;
}
EOF

# Generate gRPC service
./generator -type protocol -service SensorService -proto sensor_service.proto
```

### Example 3: Batch Generation

```yaml
# batch_config.yaml
batch_generation:
  - type: model
    items:
      - name: "Sensor"
      - name: "Reading"
      - name: "Alert"
  
  - type: api
    items:
      - resource: "sensors"
      - resource: "readings"
      - resource: "alerts"
  
  - type: test
    items:
      - name: "SensorService"
      - name: "ReadingService"
      - name: "AlertService"
```

```bash
./generator -config batch_config.yaml
```

## Integration

### IDE Integration

#### VS Code Extension

Install the TSIoT Code Generator extension:

```json
{
  "tsiot.generator.configPath": "./tools/code-generator/config.yaml",
  "tsiot.generator.autoFormat": true,
  "tsiot.generator.showPreview": true
}
```

#### IntelliJ Plugin

Configure the generator as an external tool:

```
Program: ./tools/code-generator/generator
Arguments: -type $GENERATOR_TYPE$ -name $NAME$
Working Directory: $ProjectFileDir$
```

### CI/CD Integration

```yaml
# .github/workflows/codegen.yml
name: Code Generation
on:
  push:
    paths:
      - 'schemas/**'
      - 'templates/**'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Generate Code
        run: |
          ./tools/code-generator/generator -config config.yaml
          git diff --exit-code || (echo "Generated code is out of date" && exit 1)
```

### Git Hooks

```bash
#!/bin/sh
# pre-commit hook
./tools/code-generator/generator -config config.yaml -dry-run
if [ $? -ne 0 ]; then
    echo "Code generation would create changes. Run generator and commit changes."
    exit 1
fi
```

## Advanced Features

### Plugin Development

Create custom plugins:

```go
package main

import (
    "github.com/inferloop/tsiot/tools/code-generator/plugin"
)

type MyPlugin struct{}

func (p *MyPlugin) Name() string {
    return "my-plugin"
}

func (p *MyPlugin) Generate(config plugin.Config) error {
    // Custom generation logic
    return nil
}

func main() {
    plugin.Register(&MyPlugin{})
}
```

### Template Inheritance

Use template inheritance for code reuse:

```yaml
templates:
  base:
    name: "service_base"
    template: |
      type {{.Name}}Service struct {
          repo Repository
          logger *logrus.Logger
      }
  
  derived:
    name: "crud_service"
    extends: "service_base"
    template: |
      {{template "service_base" .}}
      
      func (s *{{.Name}}Service) Create(ctx context.Context, entity {{.Entity}}) error {
          return s.repo.Create(ctx, entity)
      }
```

### Code Analysis

Analyze existing code to generate templates:

```bash
# Analyze existing code patterns
./generator analyze -path ./internal/services -pattern "*.go" -output patterns.yaml

# Generate based on analysis
./generator -config patterns.yaml -type service -name NewService
```

## Best Practices

1. **Template Organization**
   - Keep templates modular and reusable
   - Use consistent naming conventions
   - Document template variables
   - Version control templates

2. **Configuration Management**
   - Use environment-specific configs
   - Validate configurations
   - Document configuration options
   - Use inheritance for common settings

3. **Code Quality**
   - Always format generated code
   - Include comprehensive tests
   - Add meaningful comments
   - Follow project conventions

4. **Version Control**
   - Include generation metadata
   - Use .gitignore for generated files
   - Document generation process
   - Tag generated code clearly

## Troubleshooting

### Common Issues

1. **Template Syntax Errors**
   ```bash
   # Validate template syntax
   ./generator -template templates/model.go.tmpl -validate
   ```

2. **Missing Variables**
   ```bash
   # Check required variables
   ./generator -template templates/service.go.tmpl -vars-check
   ```

3. **Output Path Issues**
   ```bash
   # Dry run to check output paths
   ./generator -config config.yaml -dry-run
   ```

### Debug Mode

```bash
# Enable debug logging
./generator -config config.yaml -verbose -debug

# Show template execution
./generator -template templates/model.go.tmpl -debug-template
```

## Resources

- [Go Templates Documentation](https://golang.org/pkg/text/template/)
- [Protocol Buffers Guide](https://developers.google.com/protocol-buffers)
- [Code Generation Best Practices](https://blog.golang.org/generate)
- [TSIoT Architecture Guide](https://inferloop.com/docs/architecture)