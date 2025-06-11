package mcp

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPServerInitialization(t *testing.T) {
	transport := NewMockTransport()
	server := NewMCPServer(transport)
	
	assert.NotNil(t, server)
	assert.Equal(t, "TimeSeries Synthetic MCP Server", server.info.Name)
	assert.Equal(t, "2024-11-05", server.info.ProtocolVersion)
	assert.NotNil(t, server.capabilities.Tools)
	assert.True(t, server.capabilities.Tools.ListChanged)
}

func TestMCPServerToolsRegistration(t *testing.T) {
	transport := NewMockTransport()
	server := NewMCPServer(transport)
	
	tools := server.GetTools()
	assert.Len(t, tools, 2)
	
	toolNames := make([]string, 0)
	for _, tool := range tools {
		toolNames = append(toolNames, tool.Name)
	}
	
	assert.Contains(t, toolNames, "generateTimeSeries")
	assert.Contains(t, toolNames, "validateTimeSeries")
}

func TestMCPServerResourcesRegistration(t *testing.T) {
	transport := NewMockTransport()
	server := NewMCPServer(transport)
	
	resources := server.GetResources()
	assert.Len(t, resources, 2)
	
	resourceURIs := make([]string, 0)
	for _, resource := range resources {
		resourceURIs = append(resourceURIs, resource.URI)
	}
	
	assert.Contains(t, resourceURIs, "timeseries://generators")
	assert.Contains(t, resourceURIs, "timeseries://templates")
}

func TestMCPClientInitialization(t *testing.T) {
	transport := NewMockTransport()
	client := NewMCPClient(transport)
	
	transport.SetResponse(Message{
		Jsonrpc: "2.0",
		Result: InitializeResult{
			ProtocolVersion: "2024-11-05",
			ServerInfo: ServerInfo{
				Name:    "Test Server",
				Version: "1.0.0",
			},
			Capabilities: ServerCapabilities{
				Tools: &ToolsCapability{ListChanged: true},
			},
		},
		ID: 1,
	})
	
	err := client.Connect()
	require.NoError(t, err)
	
	assert.Equal(t, "Test Server", client.GetServerInfo().Name)
	assert.NotNil(t, client.GetCapabilities().Tools)
}

func TestMessageHandlerInitialize(t *testing.T) {
	transport := NewMockTransport()
	server := NewMCPServer(transport)
	handler := NewMessageHandler(server)
	
	request := Message{
		Jsonrpc: "2.0",
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "Test Client",
				"version": "1.0.0",
			},
		},
		ID: 1,
	}
	
	response, err := handler.HandleMessage(request)
	require.NoError(t, err)
	
	var result InitializeResult
	data, _ := json.Marshal(response.Result)
	json.Unmarshal(data, &result)
	
	assert.Equal(t, "2024-11-05", result.ProtocolVersion)
	assert.Equal(t, "TimeSeries Synthetic MCP Server", result.ServerInfo.Name)
}

func TestMessageHandlerToolsList(t *testing.T) {
	transport := NewMockTransport()
	server := NewMCPServer(transport)
	handler := NewMessageHandler(server)
	
	request := Message{
		Jsonrpc: "2.0",
		Method:  "tools/list",
		ID:      1,
	}
	
	response, err := handler.HandleMessage(request)
	require.NoError(t, err)
	
	var result struct {
		Tools []Tool `json:"tools"`
	}
	data, _ := json.Marshal(response.Result)
	json.Unmarshal(data, &result)
	
	assert.Len(t, result.Tools, 2)
}

func TestMessageHandlerToolsCall(t *testing.T) {
	transport := NewMockTransport()
	server := NewMCPServer(transport)
	handler := NewMessageHandler(server)
	
	request := Message{
		Jsonrpc: "2.0",
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name": "generateTimeSeries",
			"arguments": map[string]interface{}{
				"generator": "statistical",
				"length":    100,
				"frequency": "1m",
				"parameters": map[string]interface{}{
					"mean": 0.0,
					"std":  1.0,
				},
			},
		},
		ID: 1,
	}
	
	response, err := handler.HandleMessage(request)
	require.NoError(t, err)
	assert.NotNil(t, response.Result)
}

func TestResourceManager(t *testing.T) {
	rm := NewResourceManager()
	
	t.Run("GetResource", func(t *testing.T) {
		data, err := rm.GetResource("timeseries://generators")
		require.NoError(t, err)
		assert.NotNil(t, data)
		
		generators, ok := data.(map[string]interface{})
		assert.True(t, ok)
		assert.Contains(t, generators, "generators")
	})
	
	t.Run("Caching", func(t *testing.T) {
		// First call
		data1, err := rm.GetResource("timeseries://templates")
		require.NoError(t, err)
		
		// Second call should be cached
		data2, err := rm.GetResource("timeseries://templates")
		require.NoError(t, err)
		
		assert.Equal(t, data1, data2)
	})
	
	t.Run("InvalidResource", func(t *testing.T) {
		_, err := rm.GetResource("timeseries://invalid")
		assert.Error(t, err)
	})
}

func TestTransports(t *testing.T) {
	t.Run("StdioTransport", func(t *testing.T) {
		transport := NewStdioTransport()
		assert.NotNil(t, transport)
		
		err := transport.Start()
		assert.NoError(t, err)
		
		err = transport.Stop()
		assert.NoError(t, err)
	})
	
	t.Run("HTTPTransport", func(t *testing.T) {
		transport := NewHTTPTransport(":0")
		assert.NotNil(t, transport)
		
		err := transport.Start()
		assert.NoError(t, err)
		
		time.Sleep(100 * time.Millisecond)
		
		err = transport.Stop()
		assert.NoError(t, err)
	})
}

// Mock Transport for testing
type MockTransport struct {
	incoming chan Message
	outgoing chan Message
	response Message
}

func NewMockTransport() *MockTransport {
	return &MockTransport{
		incoming: make(chan Message, 10),
		outgoing: make(chan Message, 10),
	}
}

func (t *MockTransport) Start() error {
	return nil
}

func (t *MockTransport) Stop() error {
	close(t.incoming)
	close(t.outgoing)
	return nil
}

func (t *MockTransport) Send(msg Message) error {
	t.outgoing <- msg
	return nil
}

func (t *MockTransport) Receive() (Message, error) {
	if t.response.Jsonrpc != "" {
		msg := t.response
		t.response = Message{}
		return msg, nil
	}
	
	select {
	case msg := <-t.incoming:
		return msg, nil
	case <-time.After(100 * time.Millisecond):
		return Message{}, nil
	}
}

func (t *MockTransport) SetResponse(msg Message) {
	t.response = msg
}