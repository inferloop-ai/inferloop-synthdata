package middleware

import (
	"encoding/json"
	"net/http"
	"time"
)

// writeJSON writes a JSON response to the http.ResponseWriter
func writeJSON(w http.ResponseWriter, data interface{}) error {
	w.Header().Set("Content-Type", "application/json")
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

// generateRequestID generates a unique request ID
func generateRequestID() string {
	// Simple timestamp-based ID - in production, use a proper UUID library
	return time.Now().Format("20060102150405.000000")
}

// sanitizeString removes potentially dangerous characters from strings
func sanitizeString(input string) string {
	// Basic sanitization - remove null bytes and control characters
	result := make([]byte, 0, len(input))
	for _, b := range []byte(input) {
		if b >= 32 && b < 127 { // Printable ASCII characters
			result = append(result, b)
		}
	}
	return string(result)
}

// parseContentType extracts the media type from a Content-Type header value
func parseContentType(contentType string) string {
	for i, char := range contentType {
		if char == ' ' || char == ';' {
			return contentType[:i]
		}
	}
	return contentType
}

// isJSONContentType checks if the content type indicates JSON
func isJSONContentType(contentType string) bool {
	mediaType := parseContentType(contentType)
	return mediaType == "application/json" || 
		   mediaType == "application/json; charset=utf-8" ||
		   mediaType == "text/json"
}

// ErrorResponse represents a standard error response format
type ErrorResponse struct {
	Error     ErrorDetail `json:"error"`
	Timestamp string      `json:"timestamp"`
	RequestID string      `json:"request_id,omitempty"`
}

// ErrorDetail contains error information
type ErrorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

// NewErrorResponse creates a standardized error response
func NewErrorResponse(code, message, details, requestID string) *ErrorResponse {
	return &ErrorResponse{
		Error: ErrorDetail{
			Code:    code,
			Message: message,
			Details: details,
		},
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		RequestID: requestID,
	}
}

// WriteErrorResponse writes a standardized error response
func WriteErrorResponse(w http.ResponseWriter, statusCode int, code, message, details, requestID string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	
	response := NewErrorResponse(code, message, details, requestID)
	if err := writeJSON(w, response); err != nil {
		// Fallback to simple text response if JSON encoding fails
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte(message))
	}
}