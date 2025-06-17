package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

type MCPClient struct {
	transport    Transport
	serverInfo   *ServerInfo
	capabilities *ServerCapabilities
	
	requestID    atomic.Int64
	pending      map[interface{}]chan Message
	mu           sync.RWMutex
	
	ctx          context.Context
	cancel       context.CancelFunc
}

func NewMCPClient(transport Transport) *MCPClient {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &MCPClient{
		transport: transport,
		pending:   make(map[interface{}]chan Message),
		ctx:       ctx,
		cancel:    cancel,
	}
}

func (c *MCPClient) Connect() error {
	if err := c.transport.Start(); err != nil {
		return fmt.Errorf("failed to start transport: %w", err)
	}
	
	go c.handleResponses()
	
	initResult, err := c.Initialize()
	if err != nil {
		return fmt.Errorf("failed to initialize: %w", err)
	}
	
	c.serverInfo = &initResult.ServerInfo
	c.capabilities = &initResult.Capabilities
	
	return nil
}

func (c *MCPClient) Disconnect() error {
	c.cancel()
	return c.transport.Stop()
}

func (c *MCPClient) Initialize() (*InitializeResult, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "TimeSeries MCP Client",
				"version": "1.0.0",
			},
		},
		ID: c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result InitializeResult
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return &result, nil
}

type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo         `json:"serverInfo"`
	Instructions    string             `json:"instructions,omitempty"`
}

func (c *MCPClient) ListTools() ([]Tool, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "tools/list",
		ID:      c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result struct {
		Tools []Tool `json:"tools"`
	}
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return result.Tools, nil
}

func (c *MCPClient) CallTool(name string, arguments map[string]interface{}) (*ToolResult, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name":      name,
			"arguments": arguments,
		},
		ID: c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result ToolResult
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return &result, nil
}

type ToolResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

func (c *MCPClient) ListResources() ([]Resource, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "resources/list",
		ID:      c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result struct {
		Resources []Resource `json:"resources"`
	}
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return result.Resources, nil
}

func (c *MCPClient) ReadResource(uri string) (*ResourceContent, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "resources/read",
		Params: map[string]interface{}{
			"uri": uri,
		},
		ID: c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result ResourceContent
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return &result, nil
}

type ResourceContent struct {
	Contents []struct {
		URI      string `json:"uri"`
		MimeType string `json:"mimeType"`
		Text     string `json:"text"`
	} `json:"contents"`
}

func (c *MCPClient) SubscribeToResource(uri string) error {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "resources/subscribe",
		Params: map[string]interface{}{
			"uri": uri,
		},
		ID: c.nextRequestID(),
	}
	
	_, err := c.sendRequest(request)
	return err
}

func (c *MCPClient) ListPrompts() ([]Prompt, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "prompts/list",
		ID:      c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result struct {
		Prompts []Prompt `json:"prompts"`
	}
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return result.Prompts, nil
}

func (c *MCPClient) GetPrompt(name string, arguments map[string]interface{}) (*PromptResult, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "prompts/get",
		Params: map[string]interface{}{
			"name":      name,
			"arguments": arguments,
		},
		ID: c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result PromptResult
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return &result, nil
}

type PromptResult struct {
	Messages []struct {
		Role    string                 `json:"role"`
		Content map[string]interface{} `json:"content"`
	} `json:"messages"`
}

func (c *MCPClient) Complete(messages []map[string]interface{}, ref map[string]string, argument map[string]string) (*CompletionResult, error) {
	request := Message{
		Jsonrpc: "2.0",
		Method:  "completion/complete",
		Params: map[string]interface{}{
			"messages": messages,
			"ref":      ref,
			"argument": argument,
		},
		ID: c.nextRequestID(),
	}
	
	response, err := c.sendRequest(request)
	if err != nil {
		return nil, err
	}
	
	var result CompletionResult
	if err := c.unmarshalResult(response.Result, &result); err != nil {
		return nil, err
	}
	
	return &result, nil
}

type CompletionResult struct {
	Completion map[string]interface{} `json:"completion"`
}

func (c *MCPClient) sendRequest(request Message) (Message, error) {
	responseChan := make(chan Message, 1)
	
	c.mu.Lock()
	c.pending[request.ID] = responseChan
	c.mu.Unlock()
	
	defer func() {
		c.mu.Lock()
		delete(c.pending, request.ID)
		c.mu.Unlock()
	}()
	
	if err := c.transport.Send(request); err != nil {
		return Message{}, err
	}
	
	select {
	case response := <-responseChan:
		if response.Error != nil {
			return Message{}, fmt.Errorf("server error: %s", response.Error.Message)
		}
		return response, nil
	case <-time.After(30 * time.Second):
		return Message{}, fmt.Errorf("request timeout")
	case <-c.ctx.Done():
		return Message{}, fmt.Errorf("client closed")
	}
}

func (c *MCPClient) handleResponses() {
	for {
		select {
		case <-c.ctx.Done():
			return
		default:
			msg, err := c.transport.Receive()
			if err != nil {
				continue
			}
			
			if msg.ID != nil {
				c.mu.RLock()
				responseChan, ok := c.pending[msg.ID]
				c.mu.RUnlock()
				
				if ok {
					select {
					case responseChan <- msg:
					default:
					}
				}
			} else {
				// Handle notifications
				c.handleNotification(msg)
			}
		}
	}
}

func (c *MCPClient) handleNotification(msg Message) {
	switch msg.Method {
	case "notifications/tools/list_changed":
		// Handle tool list change notification
	case "notifications/resources/list_changed":
		// Handle resource list change notification
	case "notifications/prompts/list_changed":
		// Handle prompt list change notification
	}
}

func (c *MCPClient) nextRequestID() interface{} {
	return c.requestID.Add(1)
}

func (c *MCPClient) unmarshalResult(result interface{}, target interface{}) error {
	data, err := json.Marshal(result)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

func (c *MCPClient) GetServerInfo() *ServerInfo {
	return c.serverInfo
}

func (c *MCPClient) GetCapabilities() *ServerCapabilities {
	return c.capabilities
}