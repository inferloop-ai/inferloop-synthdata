//! TSIOT Rust SDK Client
//!
//! This module provides the main client for interacting with the TSIOT service.
//! It includes comprehensive error handling, authentication, retry logic, and support
//! for both sync and async operations.

use crate::error::{Error, ErrorKind, Result};
use crate::timeseries::TimeSeries;
use crate::types::DataFormat;
use reqwest::{Client as HttpClient, RequestBuilder, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use url::Url;

/// Configuration for the TSIOT client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Base URL of the TSIOT service
    pub base_url: Url,
    /// API key for authentication
    pub api_key: Option<String>,
    /// JWT token for authentication
    pub jwt_token: Option<String>,
    /// Request timeout duration
    pub timeout: Duration,
    /// Number of retry attempts
    pub max_retries: usize,
    /// Whether to verify SSL certificates
    pub verify_ssl: bool,
    /// Custom user agent string
    pub user_agent: String,
    /// Connect timeout duration
    pub connect_timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".parse().unwrap(),
            api_key: None,
            jwt_token: None,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            verify_ssl: true,
            user_agent: "tsiot-rust-sdk/1.0.0".to_string(),
            connect_timeout: Duration::from_secs(10),
        }
    }
}

impl ClientConfig {
    /// Create a new client configuration with the given base URL
    pub fn new<S: AsRef<str>>(base_url: S) -> Result<Self> {
        let base_url = Url::parse(base_url.as_ref())?;
        Ok(Self {
            base_url,
            ..Default::default()
        })
    }

    /// Set the API key for authentication
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the JWT token for authentication
    pub fn with_jwt_token<S: Into<String>>(mut self, jwt_token: S) -> Self {
        self.jwt_token = Some(jwt_token.into());
        self
    }

    /// Set the request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the number of retry attempts
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set SSL verification
    pub fn with_ssl_verification(mut self, verify_ssl: bool) -> Self {
        self.verify_ssl = verify_ssl;
        self
    }

    /// Set custom user agent
    pub fn with_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.user_agent = user_agent.into();
        self
    }
}

/// Request for generating time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Type of generator to use
    #[serde(rename = "type")]
    pub generator_type: String,
    /// Number of data points to generate
    pub length: usize,
    /// Generator-specific parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    /// Optional metadata for the generated series
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GenerateRequest {
    /// Create a new generation request
    pub fn new<S: Into<String>>(generator_type: S, length: usize) -> Self {
        Self {
            generator_type: generator_type.into(),
            length,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a parameter to the request
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Add metadata to the request
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Request for validating time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateRequest {
    /// Time series data to validate
    pub time_series: serde_json::Value,
    /// Types of validation to perform
    #[serde(default)]
    pub validation_types: Vec<String>,
}

impl ValidateRequest {
    /// Create a new validation request
    pub fn new(time_series: &TimeSeries) -> Result<Self> {
        let time_series_json = serde_json::to_value(time_series)?;
        Ok(Self {
            time_series: time_series_json,
            validation_types: vec!["quality".to_string(), "statistical".to_string()],
        })
    }

    /// Set validation types
    pub fn with_validation_types(mut self, types: Vec<String>) -> Self {
        self.validation_types = types;
        self
    }
}

/// Request for analyzing time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeRequest {
    /// Time series data to analyze
    pub time_series: serde_json::Value,
    /// Types of analysis to perform
    #[serde(default)]
    pub analysis_types: Vec<String>,
    /// Analysis parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

impl AnalyzeRequest {
    /// Create a new analysis request
    pub fn new(time_series: &TimeSeries) -> Result<Self> {
        let time_series_json = serde_json::to_value(time_series)?;
        Ok(Self {
            time_series: time_series_json,
            analysis_types: vec!["basic".to_string(), "trend".to_string()],
            parameters: HashMap::new(),
        })
    }

    /// Set analysis types
    pub fn with_analysis_types(mut self, types: Vec<String>) -> Self {
        self.analysis_types = types;
        self
    }

    /// Add a parameter
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

/// Export request for time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    /// Time series data to export
    pub time_series: serde_json::Value,
    /// Export format
    pub format: DataFormat,
    /// Whether to compress the output
    #[serde(default)]
    pub compression: bool,
}

impl ExportRequest {
    /// Create a new export request
    pub fn new(time_series: &TimeSeries, format: DataFormat) -> Result<Self> {
        let time_series_json = serde_json::to_value(time_series)?;
        Ok(Self {
            time_series: time_series_json,
            format,
            compression: false,
        })
    }

    /// Enable compression
    pub fn with_compression(mut self, compression: bool) -> Self {
        self.compression = compression;
        self
    }
}

/// Main TSIOT client
#[derive(Debug)]
pub struct TSIOTClient {
    config: ClientConfig,
    http_client: HttpClient,
}

impl TSIOTClient {
    /// Create a new TSIOT client with the given configuration
    pub fn new(config: ClientConfig) -> Result<Self> {
        let mut builder = HttpClient::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .user_agent(&config.user_agent);

        if !config.verify_ssl {
            builder = builder.danger_accept_invalid_certs(true);
        }

        let http_client = builder.build()?;

        let client = Self {
            config,
            http_client,
        };

        log::info!("Initialized TSIOT client for {}", client.config.base_url);
        Ok(client)
    }

    /// Create a client with default configuration for the given URL
    pub fn from_url<S: AsRef<str>>(url: S) -> Result<Self> {
        let config = ClientConfig::new(url)?;
        Self::new(config)
    }

    /// Create a client with API key authentication
    pub fn with_api_key<U, K>(url: U, api_key: K) -> Result<Self>
    where
        U: AsRef<str>,
        K: Into<String>,
    {
        let config = ClientConfig::new(url)?.with_api_key(api_key);
        Self::new(config)
    }

    /// Create a client with JWT token authentication
    pub fn with_jwt_token<U, T>(url: U, jwt_token: T) -> Result<Self>
    where
        U: AsRef<str>,
        T: Into<String>,
    {
        let config = ClientConfig::new(url)?.with_jwt_token(jwt_token);
        Self::new(config)
    }

    /// Make an HTTP request with retry logic
    async fn make_request(&self, mut request_builder: RequestBuilder) -> Result<Response> {
        // Add authentication headers
        if let Some(ref api_key) = self.config.api_key {
            request_builder = request_builder.header("X-API-Key", api_key);
        } else if let Some(ref jwt_token) = self.config.jwt_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", jwt_token));
        }

        // Add standard headers
        request_builder = request_builder
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            let request = request_builder
                .try_clone()
                .ok_or_else(|| Error::internal("Failed to clone request"))?;

            match self.http_client.execute(request).await {
                Ok(response) => {
                    if self.is_retryable_status(response.status().as_u16()) && attempt < self.config.max_retries {
                        let delay = self.calculate_retry_delay(attempt);
                        log::warn!("Request failed with status {}, retrying in {:?}", response.status(), delay);
                        sleep(delay).await;
                        continue;
                    }
                    return self.handle_response(response).await;
                }
                Err(err) => {
                    last_error = Some(err.into());
                    if attempt < self.config.max_retries {
                        let delay = self.calculate_retry_delay(attempt);
                        log::warn!("Request failed: {:?}, retrying in {:?}", last_error, delay);
                        sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::internal("Max retries exceeded")))
    }

    /// Handle HTTP response and convert to appropriate result
    async fn handle_response(&self, response: Response) -> Result<Response> {
        let status = response.status().as_u16();

        match status {
            200..=299 => Ok(response),
            400 => {
                let body = response.text().await.unwrap_or_default();
                Err(Error::validation(body, None, None))
            }
            401 => Err(Error::authentication("Authentication failed")),
            403 => Err(Error::authentication("Access forbidden")),
            404 => Err(Error::network(Some(404), Some("Endpoint not found".to_string()))),
            429 => {
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|h| h.to_str().ok())
                    .and_then(|s| s.parse().ok());
                Err(Error::rate_limit(retry_after))
            }
            500..=599 => {
                let body = response.text().await.unwrap_or_default();
                Err(Error::network(Some(status), Some(body)))
            }
            _ => {
                let body = response.text().await.unwrap_or_default();
                Err(Error::network(Some(status), Some(format!("Unexpected status: {}", body))))
            }
        }
    }

    /// Check if HTTP status code is retryable
    fn is_retryable_status(&self, status: u16) -> bool {
        matches!(status, 429 | 500..=599)
    }

    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, attempt: usize) -> Duration {
        let base_delay = Duration::from_millis(1000);
        let max_delay = Duration::from_secs(30);
        let delay = base_delay * (2_u32.pow(attempt as u32));
        std::cmp::min(delay, max_delay)
    }

    /// Build URL for endpoint
    fn build_url(&self, endpoint: &str) -> Result<Url> {
        self.config
            .base_url
            .join(endpoint)
            .map_err(|e| Error::configuration(format!("Invalid endpoint: {}", e)))
    }

    /// Check service health
    pub async fn health_check(&self) -> Result<serde_json::Value> {
        let url = self.build_url("/health")?;
        let request = self.http_client.get(url);
        let response = self.make_request(request).await?;
        let json = response.json().await?;
        Ok(json)
    }

    /// Get service information
    pub async fn get_info(&self) -> Result<serde_json::Value> {
        let url = self.build_url("/api/v1/info")?;
        let request = self.http_client.get(url);
        let response = self.make_request(request).await?;
        let json = response.json().await?;
        Ok(json)
    }

    /// Generate synthetic time series data
    pub async fn generate(&self, request: &GenerateRequest) -> Result<TimeSeries> {
        let url = self.build_url("/api/v1/generate")?;
        let request_builder = self.http_client.post(url).json(request);
        let response = self.make_request(request_builder).await?;
        let time_series: TimeSeries = response.json().await?;
        Ok(time_series)
    }

    /// Validate time series data
    pub async fn validate(&self, request: &ValidateRequest) -> Result<serde_json::Value> {
        let url = self.build_url("/api/v1/validate")?;
        let request_builder = self.http_client.post(url).json(request);
        let response = self.make_request(request_builder).await?;
        let validation_result = response.json().await?;
        Ok(validation_result)
    }

    /// Analyze time series data
    pub async fn analyze(&self, request: &AnalyzeRequest) -> Result<serde_json::Value> {
        let url = self.build_url("/api/v1/analyze")?;
        let request_builder = self.http_client.post(url).json(request);
        let response = self.make_request(request_builder).await?;
        let analysis_result = response.json().await?;
        Ok(analysis_result)
    }

    /// List available generators
    pub async fn list_generators(&self) -> Result<Vec<serde_json::Value>> {
        let url = self.build_url("/api/v1/generators")?;
        let request = self.http_client.get(url);
        let response = self.make_request(request).await?;
        let generators = response.json().await?;
        Ok(generators)
    }

    /// List available validators
    pub async fn list_validators(&self) -> Result<Vec<serde_json::Value>> {
        let url = self.build_url("/api/v1/validators")?;
        let request = self.http_client.get(url);
        let response = self.make_request(request).await?;
        let validators = response.json().await?;
        Ok(validators)
    }

    /// Export time series data
    pub async fn export_data(&self, request: &ExportRequest) -> Result<Vec<u8>> {
        let url = self.build_url("/api/v1/export")?;
        let request_builder = self.http_client.post(url).json(request);
        let response = self.make_request(request_builder).await?;

        if request.compression {
            // Handle compressed data
            let json: serde_json::Value = response.json().await?;
            if let Some(data_str) = json.get("data").and_then(|v| v.as_str()) {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(data_str)
                    .map_err(|e| Error::serialization(format!("Failed to decode base64: {}", e)))
            } else {
                Err(Error::serialization("Invalid compressed data format"))
            }
        } else {
            // Handle uncompressed data
            let bytes = response.bytes().await?;
            Ok(bytes.to_vec())
        }
    }

    /// Get service metrics
    pub async fn get_metrics(&self) -> Result<serde_json::Value> {
        let url = self.build_url("/metrics")?;
        let request = self.http_client.get(url);
        let response = self.make_request(request).await?;
        let metrics = response.json().await?;
        Ok(metrics)
    }

    /// Generate multiple time series in batch
    pub async fn batch_generate(&self, requests: &[GenerateRequest]) -> Result<Vec<TimeSeries>> {
        let url = self.build_url("/api/v1/batch/generate")?;
        let batch_request = serde_json::json!({ "requests": requests });
        let request_builder = self.http_client.post(url).json(&batch_request);
        let response = self.make_request(request_builder).await?;
        
        let batch_response: serde_json::Value = response.json().await?;
        if let Some(results) = batch_response.get("results").and_then(|v| v.as_array()) {
            let mut time_series_list = Vec::new();
            for result in results {
                let ts: TimeSeries = serde_json::from_value(result.clone())?;
                time_series_list.push(ts);
            }
            Ok(time_series_list)
        } else {
            Err(Error::serialization("Invalid batch response format"))
        }
    }

    /// Generate multiple time series concurrently
    pub async fn concurrent_generate(
        &self,
        requests: &[GenerateRequest],
        max_concurrent: usize,
    ) -> Result<Vec<TimeSeries>> {
        use futures::stream::{self, StreamExt};

        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        
        let futures = requests.iter().map(|request| {
            let client = self;
            let sem = semaphore.clone();
            let req = request.clone();
            
            async move {
                let _permit = sem.acquire().await.map_err(|e| Error::internal(e.to_string()))?;
                client.generate(&req).await
            }
        });

        let results: Result<Vec<TimeSeries>> = stream::iter(futures)
            .buffered(max_concurrent)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        results
    }
}

/// Convenience function to create a TSIOT client
pub fn create_client(config: ClientConfig) -> Result<TSIOTClient> {
    TSIOTClient::new(config)
}

/// Convenience function to create a client with URL and API key
pub fn create_client_with_api_key<U, K>(url: U, api_key: K) -> Result<TSIOTClient>
where
    U: AsRef<str>,
    K: Into<String>,
{
    TSIOTClient::with_api_key(url, api_key)
}

/// Convenience function to create a client with URL and JWT token
pub fn create_client_with_jwt<U, T>(url: U, jwt_token: T) -> Result<TSIOTClient>
where
    U: AsRef<str>,
    T: Into<String>,
{
    TSIOTClient::with_jwt_token(url, jwt_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_creation() {
        let config = ClientConfig::new("http://localhost:8080").unwrap();
        assert_eq!(config.base_url.as_str(), "http://localhost:8080/");
        assert_eq!(config.max_retries, 3);
        assert!(config.verify_ssl);
    }

    #[test]
    fn test_client_config_with_api_key() {
        let config = ClientConfig::new("http://localhost:8080")
            .unwrap()
            .with_api_key("test-key");
        assert_eq!(config.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_generate_request_creation() {
        let request = GenerateRequest::new("arima", 1000)
            .with_parameter("ar_params", vec![0.5, -0.3])
            .with_metadata("name", "Test Series");
        
        assert_eq!(request.generator_type, "arima");
        assert_eq!(request.length, 1000);
        assert!(request.parameters.contains_key("ar_params"));
        assert!(request.metadata.contains_key("name"));
    }

    #[tokio::test]
    async fn test_client_creation() {
        let config = ClientConfig::new("http://localhost:8080").unwrap();
        let client = TSIOTClient::new(config);
        assert!(client.is_ok());
    }
}