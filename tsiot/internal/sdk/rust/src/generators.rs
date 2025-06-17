//! Time series data generators for the TSIOT Rust SDK.
//!
//! This module provides various generator implementations for creating synthetic time series data
//! using different algorithms including ARIMA, LSTM, GRU, TimeGAN, and statistical methods.

use crate::error::{Error, Result};
use crate::timeseries::{DataPoint, TimeSeries, TimeSeriesMetadata};
use crate::types::{Distribution, TrendType};
use chrono::{DateTime, Duration, Utc};
use rand::prelude::*;
use rand_distr::{Normal, Uniform, Exponential, Poisson};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base trait for all time series generators
pub trait Generator {
    /// Generate time series data
    fn generate(
        &self,
        length: usize,
        start_time: Option<DateTime<Utc>>,
        metadata: Option<TimeSeriesMetadata>,
    ) -> Result<TimeSeries>;

    /// Get the generator type name
    fn generator_type(&self) -> &'static str;

    /// Get generator parameters as JSON
    fn parameters(&self) -> Result<serde_json::Value>;
}

/// ARIMA (AutoRegressive Integrated Moving Average) generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMAGenerator {
    /// Autoregressive parameters
    pub ar_params: Vec<f64>,
    /// Moving average parameters  
    pub ma_params: Vec<f64>,
    /// Degree of differencing
    pub differencing: usize,
    /// Noise level
    pub noise: f64,
    /// Trend type
    pub trend: TrendType,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for ARIMAGenerator {
    fn default() -> Self {
        Self {
            ar_params: vec![0.5],
            ma_params: vec![0.3],
            differencing: 0,
            noise: 1.0,
            trend: TrendType::None,
            seed: None,
        }
    }
}

impl ARIMAGenerator {
    /// Create a new ARIMA generator
    pub fn new(ar_params: Vec<f64>, ma_params: Vec<f64>) -> Self {
        Self {
            ar_params,
            ma_params,
            ..Default::default()
        }
    }

    /// Set differencing degree
    pub fn with_differencing(mut self, differencing: usize) -> Self {
        self.differencing = differencing;
        self
    }

    /// Set noise level
    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise = noise;
        self
    }

    /// Set trend type
    pub fn with_trend(mut self, trend: TrendType) -> Self {
        self.trend = trend;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate trend value for given time point
    fn get_trend_value(&self, t: usize, length: usize) -> f64 {
        let normalized_t = t as f64 / length as f64;
        match self.trend {
            TrendType::None => 0.0,
            TrendType::Linear => normalized_t,
            TrendType::Quadratic => normalized_t * normalized_t,
            TrendType::Exponential => normalized_t.exp() - 1.0,
            TrendType::Logarithmic => (normalized_t + 1.0).ln(),
        }
    }
}

impl Generator for ARIMAGenerator {
    fn generate(
        &self,
        length: usize,
        start_time: Option<DateTime<Utc>>,
        metadata: Option<TimeSeriesMetadata>,
    ) -> Result<TimeSeries> {
        if length == 0 {
            return Err(Error::validation("length must be positive", Some("length".to_string()), Some(length.to_string())));
        }

        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let normal = Normal::new(0.0, self.noise).map_err(|e| Error::generation(e.to_string(), Some("arima".to_string())))?;

        let max_lag = std::cmp::max(self.ar_params.len(), self.ma_params.len());
        let mut values = Vec::with_capacity(length);
        let mut errors = Vec::with_capacity(length);

        // Initialize with small random values
        for _ in 0..max_lag {
            values.push(rng.sample(normal) * 0.1);
            errors.push(rng.sample(normal) * 0.1);
        }

        // Generate ARIMA process
        for t in max_lag..length {
            let mut value = 0.0;
            let error = rng.sample(normal);

            // Autoregressive component
            for (i, &ar_param) in self.ar_params.iter().enumerate() {
                if t > i {
                    value += ar_param * values[t - i - 1];
                }
            }

            // Moving average component
            for (i, &ma_param) in self.ma_params.iter().enumerate() {
                if t > i {
                    value += ma_param * errors[t - i - 1];
                }
            }

            // Add current error
            value += error;

            // Add trend
            value += self.get_trend_value(t, length);

            values.push(value);
            errors.push(error);
        }

        // Apply differencing (integrate)
        if self.differencing > 0 {
            for _ in 0..self.differencing {
                for i in 1..values.len() {
                    values[i] += values[i - 1];
                }
            }
        }

        // Create data points
        let start = start_time.unwrap_or_else(Utc::now);
        let data_points: Vec<DataPoint> = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let timestamp = start + Duration::seconds(i as i64);
                DataPoint::new(timestamp, value)
            })
            .collect();

        let ts_metadata = metadata.unwrap_or_else(|| {
            TimeSeriesMetadata::new("arima-generated")
                .with_name("ARIMA Generated Series")
                .with_description(format!(
                    "ARIMA({},{},{}) generated series",
                    self.ar_params.len(),
                    self.differencing,
                    self.ma_params.len()
                ))
        });

        Ok(TimeSeries::with_data_points(ts_metadata.series_id.clone(), data_points).with_metadata(ts_metadata))
    }

    fn generator_type(&self) -> &'static str {
        "arima"
    }

    fn parameters(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(Into::into)
    }
}

/// LSTM (Long Short-Term Memory) generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMGenerator {
    /// Hidden layer size
    pub hidden_size: usize,
    /// Sequence length for input
    pub sequence_length: usize,
    /// Number of LSTM layers
    pub num_layers: usize,
    /// Pattern to learn and repeat
    pub pattern: Vec<f64>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for LSTMGenerator {
    fn default() -> Self {
        Self {
            hidden_size: 50,
            sequence_length: 10,
            num_layers: 2,
            pattern: Self::generate_sine_pattern(),
            seed: None,
        }
    }
}

impl LSTMGenerator {
    /// Create a new LSTM generator
    pub fn new(hidden_size: usize, sequence_length: usize) -> Self {
        Self {
            hidden_size,
            sequence_length,
            ..Default::default()
        }
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set pattern to learn
    pub fn with_pattern(mut self, pattern: Vec<f64>) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate default sine wave pattern
    fn generate_sine_pattern() -> Vec<f64> {
        (0..100)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
            .collect()
    }
}

impl Generator for LSTMGenerator {
    fn generate(
        &self,
        length: usize,
        start_time: Option<DateTime<Utc>>,
        metadata: Option<TimeSeriesMetadata>,
    ) -> Result<TimeSeries> {
        if length == 0 {
            return Err(Error::validation("length must be positive", Some("length".to_string()), Some(length.to_string())));
        }

        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let normal = Normal::new(0.0, 0.1).map_err(|e| Error::generation(e.to_string(), Some("lstm".to_string())))?;
        let mut values = Vec::with_capacity(length);

        // Simple pattern-based generation (simulating LSTM output)
        for i in 0..length {
            let mut value = 0.0;

            // Use pattern with some variation
            let pattern_index = i % self.pattern.len();
            value = self.pattern[pattern_index];

            // Add dependency on previous values (simulating LSTM memory)
            if i > 0 {
                value += 0.1 * values[i - 1];
            }
            if i > 1 {
                value += 0.05 * values[i - 2];
            }

            // Add noise
            value += rng.sample(normal);

            values.push(value);
        }

        // Create data points
        let start = start_time.unwrap_or_else(Utc::now);
        let data_points: Vec<DataPoint> = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let timestamp = start + Duration::seconds(i as i64);
                DataPoint::new(timestamp, value)
            })
            .collect();

        let ts_metadata = metadata.unwrap_or_else(|| {
            TimeSeriesMetadata::new("lstm-generated")
                .with_name("LSTM Generated Series")
                .with_description(format!("LSTM generated series with {} hidden units", self.hidden_size))
        });

        Ok(TimeSeries::with_data_points(ts_metadata.series_id.clone(), data_points).with_metadata(ts_metadata))
    }

    fn generator_type(&self) -> &'static str {
        "lstm"
    }

    fn parameters(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(Into::into)
    }
}

/// GRU (Gated Recurrent Unit) generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRUGenerator {
    /// Hidden layer size
    pub hidden_size: usize,
    /// Sequence length for input
    pub sequence_length: usize,
    /// Seasonal pattern
    pub seasonality: Vec<f64>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for GRUGenerator {
    fn default() -> Self {
        Self {
            hidden_size: 32,
            sequence_length: 8,
            seasonality: vec![1.0, 0.8, 0.6, 0.4, 0.6, 0.8],
            seed: None,
        }
    }
}

impl GRUGenerator {
    /// Create a new GRU generator
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            ..Default::default()
        }
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, sequence_length: usize) -> Self {
        self.sequence_length = sequence_length;
        self
    }

    /// Set seasonality pattern
    pub fn with_seasonality(mut self, seasonality: Vec<f64>) -> Self {
        self.seasonality = seasonality;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Tanh activation function
    fn tanh(x: f64) -> f64 {
        x.tanh()
    }
}

impl Generator for GRUGenerator {
    fn generate(
        &self,
        length: usize,
        start_time: Option<DateTime<Utc>>,
        metadata: Option<TimeSeriesMetadata>,
    ) -> Result<TimeSeries> {
        if length == 0 {
            return Err(Error::validation("length must be positive", Some("length".to_string()), Some(length.to_string())));
        }

        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let normal = Normal::new(0.0, 0.1).map_err(|e| Error::generation(e.to_string(), Some("gru".to_string())))?;
        let mut values = Vec::with_capacity(length);
        let mut hidden_state = 0.0;

        for i in 0..length {
            // Get seasonal input
            let seasonal_value = self.seasonality[i % self.seasonality.len()];
            let input = seasonal_value + rng.sample(normal);

            // Simplified GRU computation
            let update_gate = Self::sigmoid(input + hidden_state);
            let reset_gate = Self::sigmoid(input * 0.5 + hidden_state * 0.5);
            let candidate_state = Self::tanh(input + reset_gate * hidden_state);

            // Update hidden state
            hidden_state = (1.0 - update_gate) * hidden_state + update_gate * candidate_state;

            values.push(hidden_state);
        }

        // Create data points
        let start = start_time.unwrap_or_else(Utc::now);
        let data_points: Vec<DataPoint> = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let timestamp = start + Duration::seconds(i as i64);
                DataPoint::new(timestamp, value)
            })
            .collect();

        let ts_metadata = metadata.unwrap_or_else(|| {
            TimeSeriesMetadata::new("gru-generated")
                .with_name("GRU Generated Series")
                .with_description(format!("GRU generated series with {} hidden units", self.hidden_size))
        });

        Ok(TimeSeries::with_data_points(ts_metadata.series_id.clone(), data_points).with_metadata(ts_metadata))
    }

    fn generator_type(&self) -> &'static str {
        "gru"
    }

    fn parameters(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(Into::into)
    }
}

/// TimeGAN (Time-series Generative Adversarial Network) generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeGANGenerator {
    /// Latent dimension
    pub latent_dim: usize,
    /// Sequence length
    pub sequence_length: usize,
    /// Reference pattern to learn
    pub reference_pattern: Vec<f64>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for TimeGANGenerator {
    fn default() -> Self {
        Self {
            latent_dim: 24,
            sequence_length: 24,
            reference_pattern: Self::generate_complex_pattern(),
            seed: None,
        }
    }
}

impl TimeGANGenerator {
    /// Create a new TimeGAN generator
    pub fn new(latent_dim: usize, sequence_length: usize) -> Self {
        Self {
            latent_dim,
            sequence_length,
            ..Default::default()
        }
    }

    /// Set reference pattern
    pub fn with_reference_pattern(mut self, pattern: Vec<f64>) -> Self {
        self.reference_pattern = pattern;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate complex reference pattern
    fn generate_complex_pattern() -> Vec<f64> {
        (0..100)
            .map(|i| {
                let i_f = i as f64;
                (2.0 * std::f64::consts::PI * i_f / 12.0).sin() * 0.8
                    + (2.0 * std::f64::consts::PI * i_f / 24.0).sin() * 0.5
                    + (2.0 * std::f64::consts::PI * i_f / 7.0).sin() * 0.3
            })
            .collect()
    }
}

impl Generator for TimeGANGenerator {
    fn generate(
        &self,
        length: usize,
        start_time: Option<DateTime<Utc>>,
        metadata: Option<TimeSeriesMetadata>,
    ) -> Result<TimeSeries> {
        if length == 0 {
            return Err(Error::validation("length must be positive", Some("length".to_string()), Some(length.to_string())));
        }

        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let normal = Normal::new(0.0, 1.0).map_err(|e| Error::generation(e.to_string(), Some("timegan".to_string())))?;
        let mut values = Vec::with_capacity(length);

        // Simulate TimeGAN generation process
        for i in 0..length {
            // Generate latent vector (random noise)
            let latent_vector: Vec<f64> = (0..self.latent_dim)
                .map(|_| rng.sample(normal))
                .collect();

            // Simplified "generator" network simulation
            let mut generated_value = 0.0;
            for (j, &latent_val) in latent_vector.iter().enumerate() {
                generated_value += latent_val * (2.0 * std::f64::consts::PI * j as f64 / self.latent_dim as f64).sin();
            }
            generated_value /= self.latent_dim as f64;

            // Add reference pattern influence
            let pattern_index = i % self.reference_pattern.len();
            generated_value = 0.7 * generated_value + 0.3 * self.reference_pattern[pattern_index];

            // Temporal consistency (look-back)
            if i > 0 {
                generated_value = 0.8 * generated_value + 0.2 * values[i - 1];
            }

            values.push(generated_value);
        }

        // Create data points
        let start = start_time.unwrap_or_else(Utc::now);
        let data_points: Vec<DataPoint> = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let timestamp = start + Duration::seconds(i as i64);
                DataPoint::new(timestamp, value)
            })
            .collect();

        let ts_metadata = metadata.unwrap_or_else(|| {
            TimeSeriesMetadata::new("timegan-generated")
                .with_name("TimeGAN Generated Series")
                .with_description(format!("TimeGAN generated series with latent dimension {}", self.latent_dim))
        });

        Ok(TimeSeries::with_data_points(ts_metadata.series_id.clone(), data_points).with_metadata(ts_metadata))
    }

    fn generator_type(&self) -> &'static str {
        "timegan"
    }

    fn parameters(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(Into::into)
    }
}

/// Statistical pattern generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalGenerator {
    /// Distribution type
    pub distribution: Distribution,
    /// Distribution parameters
    pub parameters: HashMap<String, f64>,
    /// Trend type
    pub trend: TrendType,
    /// Seasonal pattern
    pub seasonal: Vec<f64>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for StatisticalGenerator {
    fn default() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("mean".to_string(), 0.0);
        parameters.insert("std".to_string(), 1.0);

        Self {
            distribution: Distribution::Normal,
            parameters,
            trend: TrendType::None,
            seasonal: Vec::new(),
            seed: None,
        }
    }
}

impl StatisticalGenerator {
    /// Create a new statistical generator
    pub fn new(distribution: Distribution) -> Self {
        Self {
            distribution,
            ..Default::default()
        }
    }

    /// Set distribution parameters
    pub fn with_parameters(mut self, parameters: HashMap<String, f64>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set trend type
    pub fn with_trend(mut self, trend: TrendType) -> Self {
        self.trend = trend;
        self
    }

    /// Set seasonal pattern
    pub fn with_seasonal(mut self, seasonal: Vec<f64>) -> Self {
        self.seasonal = seasonal;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate value from specified distribution
    fn generate_from_distribution(&self, rng: &mut StdRng) -> Result<f64> {
        match self.distribution {
            Distribution::Normal => {
                let mean = self.parameters.get("mean").unwrap_or(&0.0);
                let std = self.parameters.get("std").unwrap_or(&1.0);
                let normal = Normal::new(*mean, *std).map_err(|e| Error::generation(e.to_string(), Some("statistical".to_string())))?;
                Ok(rng.sample(normal))
            }
            Distribution::Uniform => {
                let min = self.parameters.get("min").unwrap_or(&0.0);
                let max = self.parameters.get("max").unwrap_or(&1.0);
                let uniform = Uniform::new(*min, *max);
                Ok(rng.sample(uniform))
            }
            Distribution::Exponential => {
                let lambda = self.parameters.get("lambda").unwrap_or(&1.0);
                let exp = Exponential::new(*lambda).map_err(|e| Error::generation(e.to_string(), Some("statistical".to_string())))?;
                Ok(rng.sample(exp))
            }
            Distribution::Poisson => {
                let lambda = self.parameters.get("lambda").unwrap_or(&1.0);
                let poisson = Poisson::new(*lambda).map_err(|e| Error::generation(e.to_string(), Some("statistical".to_string())))?;
                Ok(rng.sample(poisson) as f64)
            }
            _ => {
                let uniform = Uniform::new(0.0, 1.0);
                Ok(rng.sample(uniform))
            }
        }
    }

    /// Get trend value for given time point
    fn get_trend_value(&self, t: usize, length: usize) -> f64 {
        let normalized_t = t as f64 / length as f64;
        match self.trend {
            TrendType::None => 0.0,
            TrendType::Linear => normalized_t,
            TrendType::Quadratic => normalized_t * normalized_t,
            TrendType::Exponential => normalized_t.exp() - 1.0,
            TrendType::Logarithmic => (normalized_t + 1.0).ln(),
        }
    }
}

impl Generator for StatisticalGenerator {
    fn generate(
        &self,
        length: usize,
        start_time: Option<DateTime<Utc>>,
        metadata: Option<TimeSeriesMetadata>,
    ) -> Result<TimeSeries> {
        if length == 0 {
            return Err(Error::validation("length must be positive", Some("length".to_string()), Some(length.to_string())));
        }

        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let mut values = Vec::with_capacity(length);

        for i in 0..length {
            let mut value = self.generate_from_distribution(&mut rng)?;

            // Add trend
            value += self.get_trend_value(i, length);

            // Add seasonal component
            if !self.seasonal.is_empty() {
                value += self.seasonal[i % self.seasonal.len()];
            }

            values.push(value);
        }

        // Create data points
        let start = start_time.unwrap_or_else(Utc::now);
        let data_points: Vec<DataPoint> = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let timestamp = start + Duration::seconds(i as i64);
                DataPoint::new(timestamp, value)
            })
            .collect();

        let ts_metadata = metadata.unwrap_or_else(|| {
            TimeSeriesMetadata::new("statistical-generated")
                .with_name("Statistical Generated Series")
                .with_description(format!("Statistical series with {} distribution", self.distribution))
        });

        Ok(TimeSeries::with_data_points(ts_metadata.series_id.clone(), data_points).with_metadata(ts_metadata))
    }

    fn generator_type(&self) -> &'static str {
        "statistical"
    }

    fn parameters(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(Into::into)
    }
}

/// Generator factory for creating generators by type
pub struct GeneratorFactory;

impl GeneratorFactory {
    /// Create a generator by type name
    pub fn create_generator(generator_type: &str, parameters: Option<serde_json::Value>) -> Result<Box<dyn Generator>> {
        match generator_type.to_lowercase().as_str() {
            "arima" => {
                if let Some(params) = parameters {
                    let generator: ARIMAGenerator = serde_json::from_value(params)?;
                    Ok(Box::new(generator))
                } else {
                    Ok(Box::new(ARIMAGenerator::default()))
                }
            }
            "lstm" => {
                if let Some(params) = parameters {
                    let generator: LSTMGenerator = serde_json::from_value(params)?;
                    Ok(Box::new(generator))
                } else {
                    Ok(Box::new(LSTMGenerator::default()))
                }
            }
            "gru" => {
                if let Some(params) = parameters {
                    let generator: GRUGenerator = serde_json::from_value(params)?;
                    Ok(Box::new(generator))
                } else {
                    Ok(Box::new(GRUGenerator::default()))
                }
            }
            "timegan" => {
                if let Some(params) = parameters {
                    let generator: TimeGANGenerator = serde_json::from_value(params)?;
                    Ok(Box::new(generator))
                } else {
                    Ok(Box::new(TimeGANGenerator::default()))
                }
            }
            "statistical" => {
                if let Some(params) = parameters {
                    let generator: StatisticalGenerator = serde_json::from_value(params)?;
                    Ok(Box::new(generator))
                } else {
                    Ok(Box::new(StatisticalGenerator::default()))
                }
            }
            _ => Err(Error::validation(
                format!("Unknown generator type: {}", generator_type),
                Some("generator_type".to_string()),
                Some(generator_type.to_string()),
            )),
        }
    }

    /// Get list of available generator types
    pub fn available_types() -> Vec<&'static str> {
        vec!["arima", "lstm", "gru", "timegan", "statistical"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_arima_generator_creation() {
        let generator = ARIMAGenerator::new(vec![0.5, -0.3], vec![0.2])
            .with_noise(0.5)
            .with_trend(TrendType::Linear);

        assert_eq!(generator.ar_params, vec![0.5, -0.3]);
        assert_eq!(generator.ma_params, vec![0.2]);
        assert_eq!(generator.noise, 0.5);
        assert_eq!(generator.trend, TrendType::Linear);
    }

    #[test]
    fn test_arima_generation() {
        let generator = ARIMAGenerator::default();
        let result = generator.generate(100, None, None);
        assert!(result.is_ok());

        let ts = result.unwrap();
        assert_eq!(ts.len(), 100);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_lstm_generator() {
        let generator = LSTMGenerator::new(32, 8);
        let result = generator.generate(50, None, None);
        assert!(result.is_ok());

        let ts = result.unwrap();
        assert_eq!(ts.len(), 50);
    }

    #[test]
    fn test_statistical_generator() {
        let generator = StatisticalGenerator::new(Distribution::Normal)
            .with_trend(TrendType::Linear);
        let result = generator.generate(100, None, None);
        assert!(result.is_ok());

        let ts = result.unwrap();
        assert_eq!(ts.len(), 100);
    }

    #[test]
    fn test_generator_factory() {
        let generator = GeneratorFactory::create_generator("arima", None);
        assert!(generator.is_ok());
        assert_eq!(generator.unwrap().generator_type(), "arima");

        let invalid_generator = GeneratorFactory::create_generator("invalid", None);
        assert!(invalid_generator.is_err());
    }

    #[test]
    fn test_available_types() {
        let types = GeneratorFactory::available_types();
        assert!(types.contains(&"arima"));
        assert!(types.contains(&"lstm"));
        assert!(types.contains(&"statistical"));
    }
}