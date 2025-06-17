package errors

import (
	"fmt"
	"net"
	"time"
)

// Network-specific error definitions
var (
	// Connection errors
	ErrConnectionRefused     = NewNetworkError("CONNECTION_REFUSED", "connection refused")
	ErrConnectionReset       = NewNetworkError("CONNECTION_RESET", "connection reset by peer")
	ErrConnectionAborted     = NewNetworkError("CONNECTION_ABORTED", "connection aborted")
	ErrConnectionLost        = NewNetworkError("CONNECTION_LOST", "connection lost")
	ErrConnectionUnavailable = NewNetworkError("CONNECTION_UNAVAILABLE", "connection unavailable")
	
	// Timeout errors
	ErrDialTimeout          = NewNetworkError("DIAL_TIMEOUT", "dial timeout")
	ErrReadTimeout          = NewNetworkError("READ_TIMEOUT", "read timeout")
	ErrWriteTimeout         = NewNetworkError("WRITE_TIMEOUT", "write timeout")
	ErrKeepAliveTimeout     = NewNetworkError("KEEPALIVE_TIMEOUT", "keep alive timeout")
	ErrIdleTimeout          = NewNetworkError("IDLE_TIMEOUT", "idle timeout")
	
	// DNS errors
	ErrDNSResolutionFailed  = NewNetworkError("DNS_RESOLUTION_FAILED", "DNS resolution failed")
	ErrDNSTimeout           = NewNetworkError("DNS_TIMEOUT", "DNS timeout")
	ErrDNSNoSuchHost        = NewNetworkError("DNS_NO_SUCH_HOST", "no such host")
	ErrDNSTemporaryFailure  = NewNetworkError("DNS_TEMPORARY_FAILURE", "DNS temporary failure")
	
	// TLS/SSL errors
	ErrTLSHandshakeFailed   = NewNetworkError("TLS_HANDSHAKE_FAILED", "TLS handshake failed")
	ErrTLSCertificateInvalid = NewNetworkError("TLS_CERTIFICATE_INVALID", "TLS certificate invalid")
	ErrTLSCertificateExpired = NewNetworkError("TLS_CERTIFICATE_EXPIRED", "TLS certificate expired")
	ErrTLSVersionMismatch   = NewNetworkError("TLS_VERSION_MISMATCH", "TLS version mismatch")
	ErrTLSCipherSuiteUnsupported = NewNetworkError("TLS_CIPHER_SUITE_UNSUPPORTED", "TLS cipher suite unsupported")
	
	// HTTP-specific errors
	ErrHTTPRequestFailed    = NewNetworkError("HTTP_REQUEST_FAILED", "HTTP request failed")
	ErrHTTPResponseInvalid  = NewNetworkError("HTTP_RESPONSE_INVALID", "HTTP response invalid")
	ErrHTTPRedirectLoop     = NewNetworkError("HTTP_REDIRECT_LOOP", "HTTP redirect loop")
	ErrHTTPUpgradeRequired  = NewNetworkError("HTTP_UPGRADE_REQUIRED", "HTTP upgrade required")
	
	// WebSocket errors
	ErrWebSocketUpgradeFailed = NewNetworkError("WEBSOCKET_UPGRADE_FAILED", "WebSocket upgrade failed")
	ErrWebSocketConnectionClosed = NewNetworkError("WEBSOCKET_CONNECTION_CLOSED", "WebSocket connection closed")
	ErrWebSocketProtocolError = NewNetworkError("WEBSOCKET_PROTOCOL_ERROR", "WebSocket protocol error")
	ErrWebSocketMessageTooLarge = NewNetworkError("WEBSOCKET_MESSAGE_TOO_LARGE", "WebSocket message too large")
	
	// gRPC errors
	ErrGRPCConnectionFailed = NewNetworkError("GRPC_CONNECTION_FAILED", "gRPC connection failed")
	ErrGRPCStreamClosed     = NewNetworkError("GRPC_STREAM_CLOSED", "gRPC stream closed")
	ErrGRPCDeadlineExceeded = NewNetworkError("GRPC_DEADLINE_EXCEEDED", "gRPC deadline exceeded")
	ErrGRPCUnavailable      = NewNetworkError("GRPC_UNAVAILABLE", "gRPC service unavailable")
	
	// Network infrastructure errors
	ErrNetworkUnreachable   = NewNetworkError("NETWORK_UNREACHABLE", "network unreachable")
	ErrHostUnreachable      = NewNetworkError("HOST_UNREACHABLE", "host unreachable")
	ErrPortUnreachable      = NewNetworkError("PORT_UNREACHABLE", "port unreachable")
	ErrNetworkDown          = NewNetworkError("NETWORK_DOWN", "network is down")
	ErrRouteNotFound        = NewNetworkError("ROUTE_NOT_FOUND", "route not found")
	
	// Bandwidth and capacity errors
	ErrBandwidthLimitExceeded = NewNetworkError("BANDWIDTH_LIMIT_EXCEEDED", "bandwidth limit exceeded")
	ErrNetworkCongestion    = NewNetworkError("NETWORK_CONGESTION", "network congestion")
	ErrBufferOverflow       = NewNetworkError("BUFFER_OVERFLOW", "network buffer overflow")
	ErrQueueFull            = NewNetworkError("QUEUE_FULL", "network queue full")
	
	// Protocol-specific errors
	ErrTCPConnectionReset   = NewNetworkError("TCP_CONNECTION_RESET", "TCP connection reset")
	ErrUDPPortUnreachable   = NewNetworkError("UDP_PORT_UNREACHABLE", "UDP port unreachable")
	ErrICMPError            = NewNetworkError("ICMP_ERROR", "ICMP error")
	ErrIPFragmentationNeeded = NewNetworkError("IP_FRAGMENTATION_NEEDED", "IP fragmentation needed")
)

// NetworkError represents a network-specific error with additional context
type NetworkError struct {
	*AppError
	RemoteAddr    string        `json:"remote_addr,omitempty"`
	LocalAddr     string        `json:"local_addr,omitempty"`
	Protocol      string        `json:"protocol,omitempty"`
	Operation     string        `json:"operation,omitempty"`
	Duration      time.Duration `json:"duration,omitempty"`
	BytesRead     int64         `json:"bytes_read,omitempty"`
	BytesWritten  int64         `json:"bytes_written,omitempty"`
	RetryAttempt  int           `json:"retry_attempt,omitempty"`
	Recoverable   bool          `json:"recoverable"`
}

// NewNetworkError creates a new network error
func NewNetworkError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeNetwork,
		Code:       code,
		Message:    message,
		Retryable:  true,
		HTTPStatus: 503,
	}
}

// WrapNetworkError wraps a network error with additional context
func WrapNetworkError(err error, operation string) *NetworkError {
	if err == nil {
		return nil
	}
	
	netErr := &NetworkError{
		AppError:  WrapError(err, ErrorTypeNetwork, "NETWORK_ERROR", "Network operation failed"),
		Operation: operation,
		Recoverable: isRecoverableNetworkError(err),
	}
	
	// Extract network-specific information from the underlying error
	if netOpErr, ok := err.(*net.OpError); ok {
		netErr.Operation = netOpErr.Op
		if netOpErr.Addr != nil {
			netErr.RemoteAddr = netOpErr.Addr.String()
		}
		if netOpErr.Source != nil {
			netErr.LocalAddr = netOpErr.Source.String()
		}
	}
	
	if dnsErr, ok := err.(*net.DNSError); ok {
		netErr.AppError.Code = "DNS_ERROR"
		netErr.AppError.Message = fmt.Sprintf("DNS error for %s: %s", dnsErr.Name, dnsErr.Err)
		netErr.Recoverable = dnsErr.Temporary()
	}
	
	return netErr
}

// NewConnectionError creates a connection-specific error
func NewConnectionError(remoteAddr, operation string, err error) *NetworkError {
	return &NetworkError{
		AppError: &AppError{
			Type:       ErrorTypeNetwork,
			Code:       "CONNECTION_ERROR",
			Message:    fmt.Sprintf("Connection error during %s", operation),
			Cause:      err,
			Retryable:  true,
			HTTPStatus: 503,
		},
		RemoteAddr:  remoteAddr,
		Operation:   operation,
		Protocol:    "tcp",
		Recoverable: true,
	}
}

// NewTimeoutError creates a timeout-specific error
func NewTimeoutError(operation string, duration time.Duration) *NetworkError {
	return &NetworkError{
		AppError: &AppError{
			Type:       ErrorTypeNetwork,
			Code:       "TIMEOUT_ERROR",
			Message:    fmt.Sprintf("Timeout during %s after %v", operation, duration),
			Retryable:  true,
			HTTPStatus: 408,
		},
		Operation:   operation,
		Duration:    duration,
		Recoverable: true,
	}
}

// NewTLSError creates a TLS-specific error
func NewTLSError(operation string, err error) *NetworkError {
	return &NetworkError{
		AppError: &AppError{
			Type:       ErrorTypeNetwork,
			Code:       "TLS_ERROR",
			Message:    fmt.Sprintf("TLS error during %s", operation),
			Cause:      err,
			Retryable:  false, // TLS errors are usually not retryable
			HTTPStatus: 495,   // SSL Certificate Error
		},
		Operation:   operation,
		Protocol:    "tls",
		Recoverable: false,
	}
}

// NewHTTPError creates an HTTP-specific error
func NewHTTPError(statusCode int, method, url string) *NetworkError {
	return &NetworkError{
		AppError: &AppError{
			Type:       ErrorTypeNetwork,
			Code:       "HTTP_ERROR",
			Message:    fmt.Sprintf("HTTP %d error for %s %s", statusCode, method, url),
			Retryable:  isRetryableHTTPStatus(statusCode),
			HTTPStatus: statusCode,
		},
		Operation:   fmt.Sprintf("%s %s", method, url),
		Protocol:    "http",
		Recoverable: isRetryableHTTPStatus(statusCode),
	}
}

// NewWebSocketError creates a WebSocket-specific error
func NewWebSocketError(operation string, err error) *NetworkError {
	return &NetworkError{
		AppError: &AppError{
			Type:       ErrorTypeNetwork,
			Code:       "WEBSOCKET_ERROR",
			Message:    fmt.Sprintf("WebSocket error during %s", operation),
			Cause:      err,
			Retryable:  true,
			HTTPStatus: 503,
		},
		Operation:   operation,
		Protocol:    "websocket",
		Recoverable: true,
	}
}

// NewGRPCError creates a gRPC-specific error
func NewGRPCError(operation string, err error) *NetworkError {
	return &NetworkError{
		AppError: &AppError{
			Type:       ErrorTypeNetwork,
			Code:       "GRPC_ERROR",
			Message:    fmt.Sprintf("gRPC error during %s", operation),
			Cause:      err,
			Retryable:  true,
			HTTPStatus: 503,
		},
		Operation:   operation,
		Protocol:    "grpc",
		Recoverable: true,
	}
}

// WithRemoteAddr adds remote address information to the network error
func (ne *NetworkError) WithRemoteAddr(addr string) *NetworkError {
	ne.RemoteAddr = addr
	return ne
}

// WithLocalAddr adds local address information to the network error
func (ne *NetworkError) WithLocalAddr(addr string) *NetworkError {
	ne.LocalAddr = addr
	return ne
}

// WithProtocol adds protocol information to the network error
func (ne *NetworkError) WithProtocol(protocol string) *NetworkError {
	ne.Protocol = protocol
	return ne
}

// WithDuration adds duration information to the network error
func (ne *NetworkError) WithDuration(duration time.Duration) *NetworkError {
	ne.Duration = duration
	return ne
}

// WithBytesTransferred adds bytes transferred information to the network error
func (ne *NetworkError) WithBytesTransferred(read, written int64) *NetworkError {
	ne.BytesRead = read
	ne.BytesWritten = written
	return ne
}

// WithRetryAttempt adds retry attempt information to the network error
func (ne *NetworkError) WithRetryAttempt(attempt int) *NetworkError {
	ne.RetryAttempt = attempt
	return ne
}

// IsRecoverable checks if the network error is recoverable
func (ne *NetworkError) IsRecoverable() bool {
	return ne.Recoverable
}

// ShouldRetry checks if the operation should be retried based on the error
func (ne *NetworkError) ShouldRetry() bool {
	return ne.Retryable && ne.Recoverable
}

// isRecoverableNetworkError determines if a network error is recoverable
func isRecoverableNetworkError(err error) bool {
	if err == nil {
		return false
	}
	
	// Check for specific error types
	if netErr, ok := err.(net.Error); ok {
		return netErr.Temporary()
	}
	
	if opErr, ok := err.(*net.OpError); ok {
		// Connection refused, reset, or timeout are usually recoverable
		return opErr.Temporary() || 
			   isConnectionRefused(opErr) ||
			   isConnectionReset(opErr) ||
			   isTimeout(opErr)
	}
	
	if dnsErr, ok := err.(*net.DNSError); ok {
		return dnsErr.Temporary()
	}
	
	return false
}

// isRetryableHTTPStatus determines if an HTTP status code is retryable
func isRetryableHTTPStatus(statusCode int) bool {
	retryableStatuses := map[int]bool{
		408: true, // Request Timeout
		429: true, // Too Many Requests
		500: true, // Internal Server Error
		502: true, // Bad Gateway
		503: true, // Service Unavailable
		504: true, // Gateway Timeout
		507: true, // Insufficient Storage
		509: true, // Bandwidth Limit Exceeded
	}
	
	return retryableStatuses[statusCode]
}

// isConnectionRefused checks if the error is a connection refused error
func isConnectionRefused(err error) bool {
	if opErr, ok := err.(*net.OpError); ok {
		if syscallErr, ok := opErr.Err.(*net.AddrError); ok {
			return syscallErr.Err == "connection refused"
		}
	}
	return false
}

// isConnectionReset checks if the error is a connection reset error
func isConnectionReset(err error) bool {
	if opErr, ok := err.(*net.OpError); ok {
		if syscallErr, ok := opErr.Err.(*net.AddrError); ok {
			return syscallErr.Err == "connection reset by peer"
		}
	}
	return false
}

// isTimeout checks if the error is a timeout error
func isTimeout(err error) bool {
	if netErr, ok := err.(net.Error); ok {
		return netErr.Timeout()
	}
	return false
}

// GetRetryDelay calculates the appropriate retry delay based on the error and attempt number
func (ne *NetworkError) GetRetryDelay(attempt int) time.Duration {
	if !ne.ShouldRetry() {
		return 0
	}
	
	// Base delay with exponential backoff
	baseDelay := time.Second
	maxDelay := 30 * time.Second
	
	// Calculate exponential backoff: baseDelay * 2^attempt
	delay := time.Duration(float64(baseDelay) * (1 << uint(attempt)))
	
	if delay > maxDelay {
		delay = maxDelay
	}
	
	return delay
}

// NetworkErrorSummary provides a summary of network error statistics
type NetworkErrorSummary struct {
	TotalErrors       int                    `json:"total_errors"`
	ErrorsByType      map[string]int         `json:"errors_by_type"`
	ErrorsByProtocol  map[string]int         `json:"errors_by_protocol"`
	RecoverableErrors int                    `json:"recoverable_errors"`
	RetryableErrors   int                    `json:"retryable_errors"`
	AverageRetries    float64                `json:"average_retries"`
	TopErrors         []string               `json:"top_errors"`
}

// AnalyzeNetworkErrors analyzes a collection of network errors and returns a summary
func AnalyzeNetworkErrors(errors []*NetworkError) *NetworkErrorSummary {
	summary := &NetworkErrorSummary{
		ErrorsByType:     make(map[string]int),
		ErrorsByProtocol: make(map[string]int),
	}
	
	totalRetries := 0
	errorCounts := make(map[string]int)
	
	for _, err := range errors {
		summary.TotalErrors++
		
		// Count by error code
		summary.ErrorsByType[err.Code]++
		errorCounts[err.Code]++
		
		// Count by protocol
		if err.Protocol != "" {
			summary.ErrorsByProtocol[err.Protocol]++
		}
		
		// Count recoverable and retryable errors
		if err.IsRecoverable() {
			summary.RecoverableErrors++
		}
		
		if err.ShouldRetry() {
			summary.RetryableErrors++
		}
		
		totalRetries += err.RetryAttempt
	}
	
	// Calculate average retries
	if summary.TotalErrors > 0 {
		summary.AverageRetries = float64(totalRetries) / float64(summary.TotalErrors)
	}
	
	// Find top errors (up to 5)
	type errorCount struct {
		code  string
		count int
	}
	
	var errorList []errorCount
	for code, count := range errorCounts {
		errorList = append(errorList, errorCount{code, count})
	}
	
	// Sort by count (descending)
	for i := 0; i < len(errorList)-1; i++ {
		for j := i + 1; j < len(errorList); j++ {
			if errorList[j].count > errorList[i].count {
				errorList[i], errorList[j] = errorList[j], errorList[i]
			}
		}
	}
	
	// Take top 5
	maxTop := 5
	if len(errorList) < maxTop {
		maxTop = len(errorList)
	}
	
	for i := 0; i < maxTop; i++ {
		summary.TopErrors = append(summary.TopErrors, errorList[i].code)
	}
	
	return summary
}