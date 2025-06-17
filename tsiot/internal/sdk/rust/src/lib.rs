//! # TSIOT Rust SDK
//!
//! A Rust client library for interacting with the Time Series IoT Synthetic Data (TSIOT) service.
//! This SDK provides comprehensive functionality for generating, validating, and analyzing time series data.
//!
//! ## Features
//!
//! - **Time series data generation** with multiple algorithms (ARIMA, LSTM, GRU, TimeGAN, Statistical)
//! - **Data validation and quality assessment** with comprehensive metrics
//! - **Analytics and statistical analysis** including trend, seasonality, and anomaly detection
//! - **Async/await support** for high-performance applications
//! - **Multiple data formats** (JSON, CSV, Parquet)
//! - **Comprehensive error handling** with detailed error types
//! - **Authentication support** (API key, JWT)
//! - **Retry logic and timeout handling**
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use tsiot_sdk::{Client, generators::ARIMAParams, Result};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create client
//!     let client = Client::new("http://localhost:8080")
//!         .with_api_key("your-api-key")?;
//!
//!     // Generate ARIMA time series
//!     let params = ARIMAParams {
//!         length: 1000,
//!         ar_params: vec![0.5, -0.3],
//!         ma_params: vec![0.2],
//!         ..Default::default()
//!     };
//!
//!     let time_series = client.generate_arima(params).await?;
//!     println!("Generated {} data points", time_series.len());
//!
//!     // Validate the data
//!     let validation = client.validate(&time_series, None).await?;
//!     println!("Quality score: {}", validation.quality_score);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Authentication
//!
//! The SDK supports multiple authentication methods:
//!
//! ```rust,no_run
//! use tsiot_sdk::Client;
//!
//! // API Key authentication
//! let client = Client::new("http://localhost:8080")
//!     .with_api_key("your-api-key")?;
//!
//! // JWT Token authentication
//! let client = Client::new("http://localhost:8080")
//!     .with_jwt_token("your-jwt-token")?;
//!
//! // No authentication (for public endpoints)
//! let client = Client::new("http://localhost:8080");
//! ```
//!
//! ## Error Handling
//!
//! All SDK operations return a `Result<T, Error>` where `Error` provides detailed
//! information about what went wrong:
//!
//! ```rust,no_run
//! use tsiot_sdk::{Client, Error, ErrorKind};
//!
//! match client.health_check().await {
//!     Ok(health) => println!("Service is healthy: {:?}", health),
//!     Err(Error { kind: ErrorKind::Network { status_code: Some(503), .. }, .. }) => {
//!         println!("Service temporarily unavailable");
//!     }
//!     Err(err) => {
//!         println!("Unexpected error: {}", err);
//!     }
//! }
//! ```

pub mod client;
pub mod error;
pub mod generators;
pub mod timeseries;
pub mod types;
pub mod validators;

// Re-export commonly used types
pub use client::Client;
pub use error::{Error, ErrorKind, Result};
pub use timeseries::{DataPoint, TimeSeries, TimeSeriesMetadata};

// Re-export generator types
pub use generators::{
    ARIMAGenerator, ARIMAParams, GRUGenerator, GRUParams, Generator, GeneratorParams,
    LSTMGenerator, LSTMParams, StatisticalGenerator, StatisticalParams, TimeGANGenerator,
    TimeGANParams,
};

// Re-export validation types
pub use validators::{ValidationLevel, ValidationResult, Validator};

// Re-export common types
pub use types::{DataFormat, Frequency};

/// SDK version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// SDK name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Default configuration values
pub mod defaults {
    use std::time::Duration;

    /// Default request timeout
    pub const TIMEOUT: Duration = Duration::from_secs(30);

    /// Default number of retries
    pub const RETRIES: usize = 3;

    /// Default batch size for bulk operations
    pub const BATCH_SIZE: usize = 1000;

    /// Default base delay for exponential backoff
    pub const BASE_DELAY: Duration = Duration::from_millis(1000);

    /// Default maximum delay for exponential backoff
    pub const MAX_DELAY: Duration = Duration::from_secs(30);
}

/// Initialize the SDK with logging
///
/// This function sets up environment-based logging for the SDK.
/// Call this once at the start of your application.
///
/// # Example
///
/// ```rust
/// tsiot_sdk::init_logging();
/// ```
pub fn init_logging() {
    env_logger::init();
}

/// Get the SDK version
pub fn version() -> &'static str {
    VERSION
}

/// Get the SDK name
pub fn name() -> &'static str {
    NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_name() {
        assert_eq!(name(), "tsiot-sdk");
    }
}