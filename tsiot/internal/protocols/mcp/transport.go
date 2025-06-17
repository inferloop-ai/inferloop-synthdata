package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
)

type Transport interface {
	Start() error
	Stop() error
	Send(msg Message) error
	Receive() (Message, error)
}

type Message struct {
	Jsonrpc string                 `json:"jsonrpc"`
	Method  string                 `json:"method,omitempty"`
	Params  interface{}            `json:"params,omitempty"`
	Result  interface{}            `json:"result,omitempty"`
	Error   *Error                 `json:"error,omitempty"`
	ID      interface{}            `json:"id,omitempty"`
}

type Error struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

type StdioTransport struct {
	reader  *bufio.Reader
	writer  *bufio.Writer
	encoder *json.Encoder
	decoder *json.Decoder
	mu      sync.Mutex
	ctx     context.Context
	cancel  context.CancelFunc
}

func NewStdioTransport() *StdioTransport {
	ctx, cancel := context.WithCancel(context.Background())
	return &StdioTransport{
		reader:  bufio.NewReader(os.Stdin),
		writer:  bufio.NewWriter(os.Stdout),
		encoder: json.NewEncoder(os.Stdout),
		decoder: json.NewDecoder(os.Stdin),
		ctx:     ctx,
		cancel:  cancel,
	}
}

func (t *StdioTransport) Start() error {
	return nil
}

func (t *StdioTransport) Stop() error {
	t.cancel()
	return nil
}

func (t *StdioTransport) Send(msg Message) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	if err := t.encoder.Encode(msg); err != nil {
		return fmt.Errorf("failed to encode message: %w", err)
	}
	
	return t.writer.Flush()
}

func (t *StdioTransport) Receive() (Message, error) {
	var msg Message
	
	if err := t.decoder.Decode(&msg); err != nil {
		if err == io.EOF {
			return msg, fmt.Errorf("connection closed")
		}
		return msg, fmt.Errorf("failed to decode message: %w", err)
	}
	
	return msg, nil
}

type HTTPTransport struct {
	server   *http.Server
	client   *http.Client
	messages chan Message
	errors   chan error
	mu       sync.Mutex
	ctx      context.Context
	cancel   context.CancelFunc
}

func NewHTTPTransport(addr string) *HTTPTransport {
	ctx, cancel := context.WithCancel(context.Background())
	
	transport := &HTTPTransport{
		client:   &http.Client{},
		messages: make(chan Message, 100),
		errors:   make(chan error, 10),
		ctx:      ctx,
		cancel:   cancel,
	}
	
	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", transport.handleHTTP)
	
	transport.server = &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	
	return transport
}

func (t *HTTPTransport) Start() error {
	go func() {
		if err := t.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			t.errors <- err
		}
	}()
	
	return nil
}

func (t *HTTPTransport) Stop() error {
	t.cancel()
	close(t.messages)
	close(t.errors)
	return t.server.Shutdown(t.ctx)
}

func (t *HTTPTransport) Send(msg Message) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	select {
	case <-t.ctx.Done():
		return fmt.Errorf("transport closed")
	default:
		t.messages <- msg
		return nil
	}
}

func (t *HTTPTransport) Receive() (Message, error) {
	select {
	case <-t.ctx.Done():
		return Message{}, fmt.Errorf("transport closed")
	case err := <-t.errors:
		return Message{}, err
	case msg := <-t.messages:
		return msg, nil
	}
}

func (t *HTTPTransport) handleHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var msg Message
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	t.messages <- msg
	
	select {
	case response := <-t.messages:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	case <-r.Context().Done():
		return
	}
}

type SSETransport struct {
	server   *http.Server
	clients  map[string]chan Message
	mu       sync.RWMutex
	messages chan Message
	ctx      context.Context
	cancel   context.CancelFunc
}

func NewSSETransport(addr string) *SSETransport {
	ctx, cancel := context.WithCancel(context.Background())
	
	transport := &SSETransport{
		clients:  make(map[string]chan Message),
		messages: make(chan Message, 100),
		ctx:      ctx,
		cancel:   cancel,
	}
	
	mux := http.NewServeMux()
	mux.HandleFunc("/mcp/sse", transport.handleSSE)
	mux.HandleFunc("/mcp/send", transport.handleSend)
	
	transport.server = &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	
	return transport
}

func (t *SSETransport) Start() error {
	go func() {
		if err := t.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			return
		}
	}()
	
	go t.broadcast()
	
	return nil
}

func (t *SSETransport) Stop() error {
	t.cancel()
	
	t.mu.Lock()
	for _, ch := range t.clients {
		close(ch)
	}
	t.mu.Unlock()
	
	close(t.messages)
	return t.server.Shutdown(t.ctx)
}

func (t *SSETransport) Send(msg Message) error {
	select {
	case <-t.ctx.Done():
		return fmt.Errorf("transport closed")
	case t.messages <- msg:
		return nil
	}
}

func (t *SSETransport) Receive() (Message, error) {
	select {
	case <-t.ctx.Done():
		return Message{}, fmt.Errorf("transport closed")
	case msg := <-t.messages:
		return msg, nil
	}
}

func (t *SSETransport) handleSSE(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	
	clientID := r.URL.Query().Get("client_id")
	if clientID == "" {
		clientID = fmt.Sprintf("client-%d", len(t.clients))
	}
	
	messageChan := make(chan Message, 10)
	
	t.mu.Lock()
	t.clients[clientID] = messageChan
	t.mu.Unlock()
	
	defer func() {
		t.mu.Lock()
		delete(t.clients, clientID)
		t.mu.Unlock()
		close(messageChan)
	}()
	
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}
	
	for {
		select {
		case <-r.Context().Done():
			return
		case msg := <-messageChan:
			data, err := json.Marshal(msg)
			if err != nil {
				continue
			}
			
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

func (t *SSETransport) handleSend(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var msg Message
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	t.messages <- msg
	w.WriteHeader(http.StatusOK)
}

func (t *SSETransport) broadcast() {
	for {
		select {
		case <-t.ctx.Done():
			return
		case msg := <-t.messages:
			t.mu.RLock()
			for _, client := range t.clients {
				select {
				case client <- msg:
				default:
				}
			}
			t.mu.RUnlock()
		}
	}
}