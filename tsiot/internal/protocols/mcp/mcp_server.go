package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/inferloop/tsiot/pkg/models"
)

type ServerCapabilities struct {
	Tools     *ToolsCapability     `json:"tools,omitempty"`
	Resources *ResourcesCapability `json:"resources,omitempty"`
	Prompts   *PromptsCapability   `json:"prompts,omitempty"`
}

type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type ResourcesCapability struct {
	Subscribe   bool `json:"subscribe,omitempty"`
	ListChanged bool `json:"listChanged,omitempty"`
}

type PromptsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type ServerInfo struct {
	Name         string `json:"name"`
	Version      string `json:"version"`
	ProtocolVersion string `json:"protocolVersion"`
}

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MimeType    string `json:"mimeType"`
}

type Prompt struct {
	Name        string                   `json:"name"`
	Description string                   `json:"description"`
	Arguments   []PromptArgument         `json:"arguments,omitempty"`
}

type PromptArgument struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
}

type MCPServer struct {
	info         ServerInfo
	capabilities ServerCapabilities
	transport    Transport
	handler      MessageHandler
	
	tools     map[string]Tool
	resources map[string]Resource
	prompts   map[string]Prompt
	
	resourceManager *ResourceManager
	sessions        map[string]*Session
	mu              sync.RWMutex
	
	ctx       context.Context
	cancel    context.CancelFunc
}

type Session struct {
	ID        string
	ClientID  string
	CreatedAt time.Time
	LastSeen  time.Time
}

func NewMCPServer(transport Transport) *MCPServer {
	ctx, cancel := context.WithCancel(context.Background())
	
	server := &MCPServer{
		info: ServerInfo{
			Name:            "TimeSeries Synthetic MCP Server",
			Version:         "1.0.0",
			ProtocolVersion: "2024-11-05",
		},
		capabilities: ServerCapabilities{
			Tools: &ToolsCapability{
				ListChanged: true,
			},
			Resources: &ResourcesCapability{
				Subscribe:   true,
				ListChanged: true,
			},
			Prompts: &PromptsCapability{
				ListChanged: true,
			},
		},
		transport:       transport,
		tools:           make(map[string]Tool),
		resources:       make(map[string]Resource),
		prompts:         make(map[string]Prompt),
		resourceManager: NewResourceManager(),
		sessions:        make(map[string]*Session),
		ctx:             ctx,
		cancel:          cancel,
	}
	
	server.handler = NewMessageHandler(server)
	server.registerBuiltinTools()
	server.registerBuiltinResources()
	server.registerBuiltinPrompts()
	
	return server
}

func (s *MCPServer) Start() error {
	if err := s.transport.Start(); err != nil {
		return fmt.Errorf("failed to start transport: %w", err)
	}
	
	go s.handleMessages()
	
	return nil
}

func (s *MCPServer) Stop() error {
	s.cancel()
	return s.transport.Stop()
}

func (s *MCPServer) handleMessages() {
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
			msg, err := s.transport.Receive()
			if err != nil {
				continue
			}
			
			response, err := s.handler.HandleMessage(msg)
			if err != nil {
				errorResp := s.createErrorResponse(msg, err)
				s.transport.Send(errorResp)
				continue
			}
			
			if response != nil {
				s.transport.Send(response)
			}
		}
	}
}

func (s *MCPServer) registerBuiltinTools() {
	s.RegisterTool(Tool{
		Name:        "generateTimeSeries",
		Description: "Generate synthetic time series data",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"generator": map[string]interface{}{
					"type":        "string",
					"description": "Generator type (statistical, arima, timegan)",
				},
				"length": map[string]interface{}{
					"type":        "integer",
					"description": "Number of data points to generate",
				},
				"frequency": map[string]interface{}{
					"type":        "string",
					"description": "Data frequency (1s, 1m, 1h, 1d)",
				},
				"parameters": map[string]interface{}{
					"type":        "object",
					"description": "Generator-specific parameters",
				},
			},
			"required": []string{"generator", "length"},
		},
	})
	
	s.RegisterTool(Tool{
		Name:        "validateTimeSeries",
		Description: "Validate synthetic time series data quality",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"data": map[string]interface{}{
					"type":        "array",
					"description": "Time series data points",
				},
				"metrics": map[string]interface{}{
					"type":        "array",
					"description": "Validation metrics to compute",
				},
			},
			"required": []string{"data"},
		},
	})
}

func (s *MCPServer) registerBuiltinResources() {
	s.RegisterResource(Resource{
		URI:         "timeseries://generators",
		Name:        "Available Generators",
		Description: "List of available time series generators",
		MimeType:    "application/json",
	})
	
	s.RegisterResource(Resource{
		URI:         "timeseries://templates",
		Name:        "Generation Templates",
		Description: "Pre-configured generation templates",
		MimeType:    "application/json",
	})
}

func (s *MCPServer) registerBuiltinPrompts() {
	s.RegisterPrompt(Prompt{
		Name:        "generateTimeSeriesPrompt",
		Description: "Generate time series data based on natural language description",
		Arguments: []PromptArgument{
			{
				Name:        "description",
				Description: "Natural language description of the time series to generate",
				Required:    true,
			},
			{
				Name:        "context",
				Description: "Additional context or constraints",
				Required:    false,
			},
		},
	})
}

func (s *MCPServer) RegisterTool(tool Tool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[tool.Name] = tool
}

func (s *MCPServer) RegisterResource(resource Resource) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resources[resource.URI] = resource
}

func (s *MCPServer) RegisterPrompt(prompt Prompt) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prompts[prompt.Name] = prompt
}

func (s *MCPServer) GetTools() []Tool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	tools := make([]Tool, 0, len(s.tools))
	for _, tool := range s.tools {
		tools = append(tools, tool)
	}
	return tools
}

func (s *MCPServer) GetResources() []Resource {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	resources := make([]Resource, 0, len(s.resources))
	for _, resource := range s.resources {
		resources = append(resources, resource)
	}
	return resources
}

func (s *MCPServer) GetPrompts() []Prompt {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	prompts := make([]Prompt, 0, len(s.prompts))
	for _, prompt := range s.prompts {
		prompts = append(prompts, prompt)
	}
	return prompts
}

func (s *MCPServer) createErrorResponse(request Message, err error) Message {
	return Message{
		Jsonrpc: "2.0",
		Error: &Error{
			Code:    -32603,
			Message: err.Error(),
		},
		ID: request.ID,
	}
}

func (s *MCPServer) CreateSession(clientID string) *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	session := &Session{
		ID:        uuid.New().String(),
		ClientID:  clientID,
		CreatedAt: time.Now(),
		LastSeen:  time.Now(),
	}
	
	s.sessions[session.ID] = session
	return session
}

func (s *MCPServer) GetServerInfo() ServerInfo {
	return s.info
}

func (s *MCPServer) GetCapabilities() ServerCapabilities {
	return s.capabilities
}

func (s *MCPServer) GetResourceManager() *ResourceManager {
	return s.resourceManager
}