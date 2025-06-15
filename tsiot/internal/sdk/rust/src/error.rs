//! Error types and handling for the TSIOT Rust SDK.

use std::fmt;
use thiserror::Error;

/// Result type alias for SDK operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for the TSIOT SDK
#[derive(Error, Debug)]
pub struct Error {
    /// The kind of error that occurred
    pub kind: ErrorKind,
    /// Additional context about the error
    pub context: Option<String>,
    /// The underlying source error, if any
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

/// Different kinds of errors that can occur
#[derive(Error, Debug)]
pub enum ErrorKind {
    /// Network-related errors
    #[error("Network error")]
    Network {
        /// HTTP status code, if available
        status_code: Option<u16>,
        /// Response body, if available
        response_body: Option<String>,
    },

    /// Authentication and authorization errors
    #[error("Authentication error")]
    Authentication {
        /// Specific authentication error message
        message: String,
    },

    /// Rate limiting errors
    #[error("Rate limit exceeded")]
    RateLimit {
        /// Number of seconds to wait before retrying
        retry_after: Option<u64>,
    },

    /// Request timeout errors
    #[error("Request timeout")]
    Timeout {
        /// Timeout duration in seconds
        timeout_seconds: u64,
    },

    /// Data validation errors
    #[error("Validation error")]
    Validation {
        /// Field that failed validation
        field: Option<String>,
        /// Value that failed validation
        value: Option<String>,
        /// Validation error message
        message: String,
    },

    /// Data generation errors
    #[error("Generation error")]
    Generation {
        /// Type of generator that failed
        generator_type: Option<String>,
        /// Generation error message
        message: String,
    },

    /// Analytics operation errors
    #[error("Analytics error")]
    Analytics {
        /// Type of analysis that failed
        analysis_type: Option<String>,
        /// Analytics error message
        message: String,
    },

    /// Serialization/deserialization errors
    #[error("Serialization error")]
    Serialization {
        /// Serialization error message
        message: String,
    },

    /// Configuration errors
    #[error("Configuration error")]
    Configuration {
        /// Configuration error message
        message: String,
    },

    /// Internal SDK errors
    #[error("Internal error")]
    Internal {
        /// Internal error message
        message: String,
    },
}

impl Error {
    /// Create a new error with the given kind
    pub fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            context: None,
            source: None,
        }
    }

    /// Create a new error with context
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Create a new error with a source error
    pub fn with_source<E: std::error::Error + Send + Sync + 'static>(mut self, source: E) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Create a network error
    pub fn network(status_code: Option<u16>, response_body: Option<String>) -> Self {
        Self::new(ErrorKind::Network {
            status_code,
            response_body,
        })
    }

    /// Create an authentication error
    pub fn authentication<S: Into<String>>(message: S) -> Self {
        Self::new(ErrorKind::Authentication {
            message: message.into(),
        })
    }

    /// Create a rate limit error
    pub fn rate_limit(retry_after: Option<u64>) -> Self {
        Self::new(ErrorKind::RateLimit { retry_after })
    }

    /// Create a timeout error
    pub fn timeout(timeout_seconds: u64) -> Self {
        Self::new(ErrorKind::Timeout { timeout_seconds })
    }

    /// Create a validation error
    pub fn validation<S: Into<String>>(
        message: S,
        field: Option<String>,
        value: Option<String>,
    ) -> Self {
        Self::new(ErrorKind::Validation {
            field,
            value,
            message: message.into(),
        })
    }

    /// Create a generation error
    pub fn generation<S: Into<String>>(message: S, generator_type: Option<String>) -> Self {
        Self::new(ErrorKind::Generation {
            generator_type,
            message: message.into(),
        })
    }

    /// Create an analytics error
    pub fn analytics<S: Into<String>>(message: S, analysis_type: Option<String>) -> Self {
        Self::new(ErrorKind::Analytics {
            analysis_type,
            message: message.into(),
        })
    }

    /// Create a serialization error
    pub fn serialization<S: Into<String>>(message: S) -> Self {
        Self::new(ErrorKind::Serialization {
            message: message.into(),
        })
    }

    /// Create a configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::new(ErrorKind::Configuration {
            message: message.into(),
        })
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::new(ErrorKind::Internal {
            message: message.into(),
        })
    }

    /// Check if this is a retryable error
    pub fn is_retryable(&self) -> bool {
        match &self.kind {
            ErrorKind::Network { status_code, .. } => {
                match status_code {
                    Some(500..=599) => true, // Server errors are retryable
                    Some(429) => true,       // Rate limit is retryable
                    Some(408) => true,       // Request timeout is retryable
                    _ => false,
                }
            }
            ErrorKind::RateLimit { .. } => true,
            ErrorKind::Timeout { .. } => true,
            _ => false,
        }
    }

    /// Get the retry delay for retryable errors
    pub fn retry_delay(&self) -> Option<std::time::Duration> {
        match &self.kind {
            ErrorKind::RateLimit { retry_after } => {
                retry_after.map(|seconds| std::time::Duration::from_secs(*seconds))
            }
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(context) = &self.context {
            write!(f, "{}: {}", context, self.kind)?;
        } else {
            write!(f, "{}", self.kind)?;
        }

        if let Some(source) = &self.source {
            write!(f, " (caused by: {})", source)?;
        }

        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

// Conversion from common error types
impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::timeout(30) // Default timeout
        } else if err.is_connect() {
            Self::network(None, None).with_source(err)
        } else if let Some(status) = err.status() {
            Self::network(Some(status.as_u16()), None).with_source(err)
        } else {
            Self::network(None, None).with_source(err)
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::serialization(err.to_string()).with_source(err)
    }
}

impl From<url::ParseError> for Error {
    fn from(err: url::ParseError) -> Self {
        Self::configuration(format!("Invalid URL: {}", err)).with_source(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::internal(err.to_string()).with_source(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::validation("Invalid value", Some("field".to_string()), Some("value".to_string()));
        assert!(matches!(err.kind, ErrorKind::Validation { .. }));
        assert!(err.to_string().contains("Validation error"));
    }

    #[test]
    fn test_error_with_context() {
        let err = Error::internal("Something went wrong")
            .with_context("During operation X");
        assert!(err.to_string().contains("During operation X"));
    }

    #[test]
    fn test_retryable_errors() {
        let network_500 = Error::network(Some(500), None);
        assert!(network_500.is_retryable());

        let network_400 = Error::network(Some(400), None);
        assert!(!network_400.is_retryable());

        let rate_limit = Error::rate_limit(Some(60));
        assert!(rate_limit.is_retryable());
        assert_eq!(rate_limit.retry_delay(), Some(std::time::Duration::from_secs(60)));
    }
}