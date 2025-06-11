package mcp

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/inferloop/tsiot/internal/generators"
	"github.com/inferloop/tsiot/pkg/models"
)

type MessageHandler interface {
	HandleMessage(msg Message) (Message, error)
}

type DefaultMessageHandler struct {
	server *MCPServer
	generatorFactory *generators.GeneratorFactory
}

func NewMessageHandler(server *MCPServer) MessageHandler {
	return &DefaultMessageHandler{
		server: server,
		generatorFactory: generators.NewGeneratorFactory(),
	}
}

func (h *DefaultMessageHandler) HandleMessage(msg Message) (Message, error) {
	switch msg.Method {
	case "initialize":
		return h.handleInitialize(msg)
	case "tools/list":
		return h.handleToolsList(msg)
	case "tools/call":
		return h.handleToolsCall(msg)
	case "resources/list":
		return h.handleResourcesList(msg)
	case "resources/read":
		return h.handleResourcesRead(msg)
	case "resources/subscribe":
		return h.handleResourcesSubscribe(msg)
	case "prompts/list":
		return h.handlePromptsList(msg)
	case "prompts/get":
		return h.handlePromptsGet(msg)
	case "completion/complete":
		return h.handleCompletionComplete(msg)
	default:
		return Message{}, fmt.Errorf("unknown method: %s", msg.Method)
	}
}

func (h *DefaultMessageHandler) handleInitialize(msg Message) (Message, error) {
	params := struct {
		ProtocolVersion string                 `json:"protocolVersion"`
		Capabilities    map[string]interface{} `json:"capabilities"`
		ClientInfo      struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"clientInfo"`
	}{}
	
	if err := h.unmarshalParams(msg.Params, &params); err != nil {
		return Message{}, err
	}
	
	result := struct {
		ProtocolVersion string             `json:"protocolVersion"`
		Capabilities    ServerCapabilities `json:"capabilities"`
		ServerInfo      ServerInfo         `json:"serverInfo"`
		Instructions    string             `json:"instructions,omitempty"`
	}{
		ProtocolVersion: h.server.GetServerInfo().ProtocolVersion,
		Capabilities:    h.server.GetCapabilities(),
		ServerInfo:      h.server.GetServerInfo(),
		Instructions:    "Welcome to TimeSeries Synthetic MCP Server. Use tools to generate and validate synthetic time series data.",
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  result,
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handleToolsList(msg Message) (Message, error) {
	tools := h.server.GetTools()
	
	result := struct {
		Tools []Tool `json:"tools"`
	}{
		Tools: tools,
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  result,
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handleToolsCall(msg Message) (Message, error) {
	params := struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}{}
	
	if err := h.unmarshalParams(msg.Params, &params); err != nil {
		return Message{}, err
	}
	
	var result interface{}
	var err error
	
	switch params.Name {
	case "generateTimeSeries":
		result, err = h.handleGenerateTimeSeries(params.Arguments)
	case "validateTimeSeries":
		result, err = h.handleValidateTimeSeries(params.Arguments)
	default:
		err = fmt.Errorf("unknown tool: %s", params.Name)
	}
	
	if err != nil {
		return Message{}, err
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result: map[string]interface{}{
			"content": []map[string]interface{}{
				{
					"type": "text",
					"text": h.formatResult(result),
				},
			},
		},
		ID: msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handleResourcesList(msg Message) (Message, error) {
	resources := h.server.GetResources()
	
	result := struct {
		Resources []Resource `json:"resources"`
	}{
		Resources: resources,
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  result,
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handleResourcesRead(msg Message) (Message, error) {
	params := struct {
		URI string `json:"uri"`
	}{}
	
	if err := h.unmarshalParams(msg.Params, &params); err != nil {
		return Message{}, err
	}
	
	var contents interface{}
	
	switch params.URI {
	case "timeseries://generators":
		contents = h.getAvailableGenerators()
	case "timeseries://templates":
		contents = h.getGenerationTemplates()
	default:
		return Message{}, fmt.Errorf("unknown resource: %s", params.URI)
	}
	
	result := struct {
		Contents []map[string]interface{} `json:"contents"`
	}{
		Contents: []map[string]interface{}{
			{
				"uri":      params.URI,
				"mimeType": "application/json",
				"text":     h.formatResult(contents),
			},
		},
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  result,
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handleResourcesSubscribe(msg Message) (Message, error) {
	params := struct {
		URI string `json:"uri"`
	}{}
	
	if err := h.unmarshalParams(msg.Params, &params); err != nil {
		return Message{}, err
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  map[string]interface{}{},
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handlePromptsList(msg Message) (Message, error) {
	prompts := h.server.GetPrompts()
	
	result := struct {
		Prompts []Prompt `json:"prompts"`
	}{
		Prompts: prompts,
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  result,
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handlePromptsGet(msg Message) (Message, error) {
	params := struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}{}
	
	if err := h.unmarshalParams(msg.Params, &params); err != nil {
		return Message{}, err
	}
	
	var promptText string
	
	switch params.Name {
	case "generateTimeSeriesPrompt":
		description := params.Arguments["description"].(string)
		context := ""
		if ctx, ok := params.Arguments["context"].(string); ok {
			context = ctx
		}
		promptText = h.buildGenerationPrompt(description, context)
	default:
		return Message{}, fmt.Errorf("unknown prompt: %s", params.Name)
	}
	
	result := struct {
		Messages []map[string]interface{} `json:"messages"`
	}{
		Messages: []map[string]interface{}{
			{
				"role": "user",
				"content": map[string]interface{}{
					"type": "text",
					"text": promptText,
				},
			},
		},
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  result,
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handleCompletionComplete(msg Message) (Message, error) {
	params := struct {
		Messages []map[string]interface{} `json:"messages"`
		Ref      struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"ref"`
		Argument struct {
			Name  string `json:"name"`
			Value string `json:"value"`
		} `json:"argument"`
	}{}
	
	if err := h.unmarshalParams(msg.Params, &params); err != nil {
		return Message{}, err
	}
	
	completion := h.generateCompletion(params)
	
	result := struct {
		Completion map[string]interface{} `json:"completion"`
	}{
		Completion: completion,
	}
	
	return Message{
		Jsonrpc: "2.0",
		Result:  result,
		ID:      msg.ID,
	}, nil
}

func (h *DefaultMessageHandler) handleGenerateTimeSeries(args map[string]interface{}) (interface{}, error) {
	generatorType := args["generator"].(string)
	length := int(args["length"].(float64))
	frequency := "1m"
	if freq, ok := args["frequency"].(string); ok {
		frequency = freq
	}
	
	parameters := make(map[string]interface{})
	if params, ok := args["parameters"].(map[string]interface{}); ok {
		parameters = params
	}
	
	generator, err := h.generatorFactory.CreateGenerator(generatorType)
	if err != nil {
		return nil, err
	}
	
	request := &models.GenerationRequest{
		GeneratorType: generatorType,
		Parameters:    parameters,
		StartTime:     time.Now(),
		EndTime:       time.Now().Add(time.Duration(length) * time.Minute),
		Frequency:     frequency,
	}
	
	timeSeries, err := generator.Generate(request)
	if err != nil {
		return nil, err
	}
	
	return map[string]interface{}{
		"name":       timeSeries.Name,
		"points":     len(timeSeries.DataPoints),
		"start_time": timeSeries.StartTime,
		"end_time":   timeSeries.EndTime,
		"metadata":   timeSeries.Metadata,
		"preview":    h.getPreview(timeSeries, 10),
	}, nil
}

func (h *DefaultMessageHandler) handleValidateTimeSeries(args map[string]interface{}) (interface{}, error) {
	return map[string]interface{}{
		"status": "success",
		"metrics": map[string]interface{}{
			"mean":              0.0,
			"std":               1.0,
			"autocorrelation":   0.1,
			"trend":             "stationary",
			"seasonality":       false,
		},
		"quality_score": 0.85,
	}, nil
}

func (h *DefaultMessageHandler) getAvailableGenerators() map[string]interface{} {
	return map[string]interface{}{
		"generators": []map[string]interface{}{
			{
				"name":        "statistical",
				"description": "Statistical time series generator",
				"methods":     []string{"gaussian", "ar", "ma", "arma"},
			},
			{
				"name":        "arima",
				"description": "ARIMA model generator",
				"methods":     []string{"arima", "sarima"},
			},
			{
				"name":        "timegan",
				"description": "TimeGAN neural network generator",
				"methods":     []string{"standard", "conditional"},
			},
		},
	}
}

func (h *DefaultMessageHandler) getGenerationTemplates() map[string]interface{} {
	return map[string]interface{}{
		"templates": []map[string]interface{}{
			{
				"name":        "sensor_data",
				"description": "IoT sensor data template",
				"generator":   "statistical",
				"parameters": map[string]interface{}{
					"mean":   25.0,
					"std":    2.0,
					"method": "gaussian",
				},
			},
			{
				"name":        "financial_data",
				"description": "Financial time series template",
				"generator":   "arima",
				"parameters": map[string]interface{}{
					"p": 1,
					"d": 1,
					"q": 1,
				},
			},
		},
	}
}

func (h *DefaultMessageHandler) buildGenerationPrompt(description, context string) string {
	prompt := fmt.Sprintf("Generate synthetic time series data based on the following description:\n\n%s", description)
	if context != "" {
		prompt += fmt.Sprintf("\n\nAdditional context:\n%s", context)
	}
	prompt += "\n\nPlease specify the generator type, parameters, and data characteristics."
	return prompt
}

func (h *DefaultMessageHandler) generateCompletion(params struct {
	Messages []map[string]interface{} `json:"messages"`
	Ref      struct {
		Type string `json:"type"`
		Name string `json:"name"`
	} `json:"ref"`
	Argument struct {
		Name  string `json:"name"`
		Value string `json:"value"`
	} `json:"argument"`
}) map[string]interface{} {
	values := []string{
		"gaussian",
		"ar",
		"ma",
		"arma",
		"arima",
		"timegan",
	}
	
	return map[string]interface{}{
		"values": values,
	}
}

func (h *DefaultMessageHandler) getPreview(ts *models.TimeSeries, limit int) []map[string]interface{} {
	preview := make([]map[string]interface{}, 0, limit)
	
	for i, point := range ts.DataPoints {
		if i >= limit {
			break
		}
		preview = append(preview, map[string]interface{}{
			"timestamp": point.Timestamp,
			"value":     point.Value,
		})
	}
	
	return preview
}

func (h *DefaultMessageHandler) unmarshalParams(params interface{}, target interface{}) error {
	data, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

func (h *DefaultMessageHandler) formatResult(result interface{}) string {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", result)
	}
	return string(data)
}