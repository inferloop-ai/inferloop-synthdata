package constants

// HTTP status codes used throughout the TSIOT system
// Based on RFC 7231 and common REST API practices

// 1xx Informational responses
const (
	StatusContinue                      = 100
	StatusSwitchingProtocols           = 101
	StatusProcessing                   = 102 // WebDAV
	StatusEarlyHints                   = 103
)

// 2xx Success responses
const (
	StatusOK                           = 200
	StatusCreated                      = 201
	StatusAccepted                     = 202
	StatusNonAuthoritativeInfo         = 203
	StatusNoContent                    = 204
	StatusResetContent                 = 205
	StatusPartialContent               = 206
	StatusMultiStatus                  = 207 // WebDAV
	StatusAlreadyReported              = 208 // WebDAV
	StatusIMUsed                       = 226
)

// 3xx Redirection responses
const (
	StatusMultipleChoices              = 300
	StatusMovedPermanently             = 301
	StatusFound                        = 302
	StatusSeeOther                     = 303
	StatusNotModified                  = 304
	StatusUseProxy                     = 305
	StatusTemporaryRedirect            = 307
	StatusPermanentRedirect            = 308
)

// 4xx Client error responses
const (
	StatusBadRequest                   = 400
	StatusUnauthorized                 = 401
	StatusPaymentRequired              = 402
	StatusForbidden                    = 403
	StatusNotFound                     = 404
	StatusMethodNotAllowed             = 405
	StatusNotAcceptable                = 406
	StatusProxyAuthRequired            = 407
	StatusRequestTimeout               = 408
	StatusConflict                     = 409
	StatusGone                         = 410
	StatusLengthRequired               = 411
	StatusPreconditionFailed           = 412
	StatusRequestEntityTooLarge        = 413
	StatusRequestURITooLong            = 414
	StatusUnsupportedMediaType         = 415
	StatusRequestedRangeNotSatisfiable = 416
	StatusExpectationFailed            = 417
	StatusTeapot                       = 418 // RFC 2324
	StatusMisdirectedRequest           = 421
	StatusUnprocessableEntity          = 422
	StatusLocked                       = 423 // WebDAV
	StatusFailedDependency             = 424 // WebDAV
	StatusTooEarly                     = 425
	StatusUpgradeRequired              = 426
	StatusPreconditionRequired         = 428
	StatusTooManyRequests              = 429
	StatusRequestHeaderFieldsTooLarge  = 431
	StatusUnavailableForLegalReasons   = 451
)

// 5xx Server error responses
const (
	StatusInternalServerError          = 500
	StatusNotImplemented               = 501
	StatusBadGateway                   = 502
	StatusServiceUnavailable           = 503
	StatusGatewayTimeout               = 504
	StatusHTTPVersionNotSupported      = 505
	StatusVariantAlsoNegotiates        = 506
	StatusInsufficientStorage          = 507 // WebDAV
	StatusLoopDetected                 = 508 // WebDAV
	StatusNotExtended                  = 510
	StatusNetworkAuthenticationRequired = 511
)

// Custom application-specific status codes (using unofficial ranges)
const (
	// 7xx Data processing errors
	StatusDataProcessingError          = 700
	StatusDataValidationError          = 701
	StatusDataTransformationError      = 702
	StatusDataCorruptionError          = 703
	StatusDataSizeExceeded             = 704
	StatusDataFormatUnsupported        = 705
	StatusDataQualityInsufficient      = 706
	StatusDataPrivacyViolation         = 707
	StatusDataRetentionExceeded        = 708
	StatusDataIntegrityError           = 709
	
	// 8xx Generation and modeling errors
	StatusGenerationError              = 800
	StatusModelNotFound                = 801
	StatusModelLoadError               = 802
	StatusModelTrainingError           = 803
	StatusInsufficientTrainingData     = 804
	StatusModelValidationError         = 805
	StatusModelConvergenceError        = 806
	StatusModelCapacityExceeded        = 807
	StatusModelVersionMismatch         = 808
	StatusModelCorrupted               = 809
	
	// 9xx System and infrastructure errors
	StatusStorageError                 = 900
	StatusStorageConnectionError       = 901
	StatusStorageCapacityExceeded      = 902
	StatusStoragePermissionError       = 903
	StatusNetworkConnectivityError     = 904
	StatusResourceExhausted            = 905
	StatusConcurrencyLimitExceeded     = 906
	StatusSystemMaintenance            = 907
	StatusConfigurationError           = 908
	StatusDependencyError              = 909
)

// StatusText returns a text for the HTTP status code
var StatusText = map[int]string{
	// 1xx
	StatusContinue:                     "Continue",
	StatusSwitchingProtocols:          "Switching Protocols",
	StatusProcessing:                  "Processing",
	StatusEarlyHints:                  "Early Hints",
	
	// 2xx
	StatusOK:                          "OK",
	StatusCreated:                     "Created",
	StatusAccepted:                    "Accepted",
	StatusNonAuthoritativeInfo:        "Non-Authoritative Information",
	StatusNoContent:                   "No Content",
	StatusResetContent:                "Reset Content",
	StatusPartialContent:              "Partial Content",
	StatusMultiStatus:                 "Multi-Status",
	StatusAlreadyReported:             "Already Reported",
	StatusIMUsed:                      "IM Used",
	
	// 3xx
	StatusMultipleChoices:             "Multiple Choices",
	StatusMovedPermanently:            "Moved Permanently",
	StatusFound:                       "Found",
	StatusSeeOther:                    "See Other",
	StatusNotModified:                 "Not Modified",
	StatusUseProxy:                    "Use Proxy",
	StatusTemporaryRedirect:           "Temporary Redirect",
	StatusPermanentRedirect:           "Permanent Redirect",
	
	// 4xx
	StatusBadRequest:                  "Bad Request",
	StatusUnauthorized:                "Unauthorized",
	StatusPaymentRequired:             "Payment Required",
	StatusForbidden:                   "Forbidden",
	StatusNotFound:                    "Not Found",
	StatusMethodNotAllowed:            "Method Not Allowed",
	StatusNotAcceptable:               "Not Acceptable",
	StatusProxyAuthRequired:           "Proxy Authentication Required",
	StatusRequestTimeout:              "Request Timeout",
	StatusConflict:                    "Conflict",
	StatusGone:                        "Gone",
	StatusLengthRequired:              "Length Required",
	StatusPreconditionFailed:          "Precondition Failed",
	StatusRequestEntityTooLarge:       "Request Entity Too Large",
	StatusRequestURITooLong:           "Request URI Too Long",
	StatusUnsupportedMediaType:        "Unsupported Media Type",
	StatusRequestedRangeNotSatisfiable: "Requested Range Not Satisfiable",
	StatusExpectationFailed:           "Expectation Failed",
	StatusTeapot:                      "I'm a teapot",
	StatusMisdirectedRequest:          "Misdirected Request",
	StatusUnprocessableEntity:         "Unprocessable Entity",
	StatusLocked:                      "Locked",
	StatusFailedDependency:            "Failed Dependency",
	StatusTooEarly:                    "Too Early",
	StatusUpgradeRequired:             "Upgrade Required",
	StatusPreconditionRequired:        "Precondition Required",
	StatusTooManyRequests:             "Too Many Requests",
	StatusRequestHeaderFieldsTooLarge: "Request Header Fields Too Large",
	StatusUnavailableForLegalReasons:  "Unavailable For Legal Reasons",
	
	// 5xx
	StatusInternalServerError:         "Internal Server Error",
	StatusNotImplemented:              "Not Implemented",
	StatusBadGateway:                  "Bad Gateway",
	StatusServiceUnavailable:          "Service Unavailable",
	StatusGatewayTimeout:              "Gateway Timeout",
	StatusHTTPVersionNotSupported:     "HTTP Version Not Supported",
	StatusVariantAlsoNegotiates:       "Variant Also Negotiates",
	StatusInsufficientStorage:         "Insufficient Storage",
	StatusLoopDetected:                "Loop Detected",
	StatusNotExtended:                 "Not Extended",
	StatusNetworkAuthenticationRequired: "Network Authentication Required",
	
	// 7xx - Data errors
	StatusDataProcessingError:         "Data Processing Error",
	StatusDataValidationError:         "Data Validation Error",
	StatusDataTransformationError:     "Data Transformation Error",
	StatusDataCorruptionError:         "Data Corruption Error",
	StatusDataSizeExceeded:            "Data Size Exceeded",
	StatusDataFormatUnsupported:       "Data Format Unsupported",
	StatusDataQualityInsufficient:     "Data Quality Insufficient",
	StatusDataPrivacyViolation:        "Data Privacy Violation",
	StatusDataRetentionExceeded:       "Data Retention Exceeded",
	StatusDataIntegrityError:          "Data Integrity Error",
	
	// 8xx - Generation errors
	StatusGenerationError:             "Generation Error",
	StatusModelNotFound:               "Model Not Found",
	StatusModelLoadError:              "Model Load Error",
	StatusModelTrainingError:          "Model Training Error",
	StatusInsufficientTrainingData:    "Insufficient Training Data",
	StatusModelValidationError:        "Model Validation Error",
	StatusModelConvergenceError:       "Model Convergence Error",
	StatusModelCapacityExceeded:       "Model Capacity Exceeded",
	StatusModelVersionMismatch:        "Model Version Mismatch",
	StatusModelCorrupted:              "Model Corrupted",
	
	// 9xx - System errors
	StatusStorageError:                "Storage Error",
	StatusStorageConnectionError:      "Storage Connection Error",
	StatusStorageCapacityExceeded:     "Storage Capacity Exceeded",
	StatusStoragePermissionError:      "Storage Permission Error",
	StatusNetworkConnectivityError:    "Network Connectivity Error",
	StatusResourceExhausted:           "Resource Exhausted",
	StatusConcurrencyLimitExceeded:    "Concurrency Limit Exceeded",
	StatusSystemMaintenance:           "System Maintenance",
	StatusConfigurationError:          "Configuration Error",
	StatusDependencyError:             "Dependency Error",
}

// GetStatusText returns the status text for the given status code
func GetStatusText(code int) string {
	if text, exists := StatusText[code]; exists {
		return text
	}
	return "Unknown Status"
}

// IsInformational checks if the status code is informational (1xx)
func IsInformational(code int) bool {
	return code >= 100 && code < 200
}

// IsSuccess checks if the status code indicates success (2xx)
func IsSuccess(code int) bool {
	return code >= 200 && code < 300
}

// IsRedirection checks if the status code indicates redirection (3xx)
func IsRedirection(code int) bool {
	return code >= 300 && code < 400
}

// IsClientError checks if the status code indicates client error (4xx)
func IsClientError(code int) bool {
	return code >= 400 && code < 500
}

// IsServerError checks if the status code indicates server error (5xx)
func IsServerError(code int) bool {
	return code >= 500 && code < 600
}

// IsDataError checks if the status code indicates data processing error (7xx)
func IsDataError(code int) bool {
	return code >= 700 && code < 800
}

// IsGenerationError checks if the status code indicates generation/modeling error (8xx)
func IsGenerationError(code int) bool {
	return code >= 800 && code < 900
}

// IsSystemError checks if the status code indicates system/infrastructure error (9xx)
func IsSystemError(code int) bool {
	return code >= 900 && code < 1000
}

// IsError checks if the status code indicates any type of error (4xx, 5xx, 7xx, 8xx, 9xx)
func IsError(code int) bool {
	return IsClientError(code) || IsServerError(code) || IsDataError(code) || 
		   IsGenerationError(code) || IsSystemError(code)
}

// IsRetryable checks if the status code indicates a retryable error
func IsRetryable(code int) bool {
	retryableCodes := map[int]bool{
		StatusRequestTimeout:              true,
		StatusTooManyRequests:            true,
		StatusInternalServerError:        true,
		StatusBadGateway:                 true,
		StatusServiceUnavailable:         true,
		StatusGatewayTimeout:             true,
		StatusNetworkConnectivityError:   true,
		StatusResourceExhausted:          true,
		StatusConcurrencyLimitExceeded:   true,
		StatusSystemMaintenance:          true,
		StatusStorageConnectionError:     true,
		StatusDependencyError:            true,
	}
	
	return retryableCodes[code]
}

// GetErrorCategory returns the category of error for the status code
func GetErrorCategory(code int) string {
	switch {
	case IsInformational(code):
		return "informational"
	case IsSuccess(code):
		return "success"
	case IsRedirection(code):
		return "redirection"
	case IsClientError(code):
		return "client_error"
	case IsServerError(code):
		return "server_error"
	case IsDataError(code):
		return "data_error"
	case IsGenerationError(code):
		return "generation_error"
	case IsSystemError(code):
		return "system_error"
	default:
		return "unknown"
	}
}

// Common status code groups for easy reference
var (
	SuccessCodes = []int{
		StatusOK, StatusCreated, StatusAccepted, StatusNoContent,
	}
	
	ClientErrorCodes = []int{
		StatusBadRequest, StatusUnauthorized, StatusForbidden, 
		StatusNotFound, StatusMethodNotAllowed, StatusConflict,
		StatusUnprocessableEntity, StatusTooManyRequests,
	}
	
	ServerErrorCodes = []int{
		StatusInternalServerError, StatusNotImplemented, StatusBadGateway,
		StatusServiceUnavailable, StatusGatewayTimeout,
	}
	
	DataErrorCodes = []int{
		StatusDataProcessingError, StatusDataValidationError,
		StatusDataTransformationError, StatusDataQualityInsufficient,
		StatusDataPrivacyViolation,
	}
	
	GenerationErrorCodes = []int{
		StatusGenerationError, StatusModelNotFound, StatusModelLoadError,
		StatusModelTrainingError, StatusInsufficientTrainingData,
		StatusModelValidationError,
	}
	
	SystemErrorCodes = []int{
		StatusStorageError, StatusNetworkConnectivityError,
		StatusResourceExhausted, StatusConfigurationError,
		StatusDependencyError,
	}
)