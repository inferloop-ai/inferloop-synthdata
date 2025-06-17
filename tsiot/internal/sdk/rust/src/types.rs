//! Common types and enums used throughout the TSIOT SDK.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported data formats for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Parquet format
    Parquet,
    /// Avro format
    Avro,
    /// Arrow format
    Arrow,
}

impl fmt::Display for DataFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataFormat::Json => write!(f, "json"),
            DataFormat::Csv => write!(f, "csv"),
            DataFormat::Parquet => write!(f, "parquet"),
            DataFormat::Avro => write!(f, "avro"),
            DataFormat::Arrow => write!(f, "arrow"),
        }
    }
}

impl Default for DataFormat {
    fn default() -> Self {
        DataFormat::Json
    }
}

/// Time series frequency/granularity options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Frequency {
    /// Nanosecond frequency
    #[serde(rename = "ns")]
    Nanosecond,
    /// Microsecond frequency
    #[serde(rename = "us")]
    Microsecond,
    /// Millisecond frequency
    #[serde(rename = "ms")]
    Millisecond,
    /// Second frequency
    #[serde(rename = "s")]
    Second,
    /// Minute frequency
    #[serde(rename = "min")]
    Minute,
    /// Hour frequency
    #[serde(rename = "h")]
    Hour,
    /// Day frequency
    #[serde(rename = "d")]
    Day,
    /// Week frequency
    #[serde(rename = "w")]
    Week,
    /// Month frequency
    #[serde(rename = "M")]
    Month,
    /// Quarter frequency
    #[serde(rename = "Q")]
    Quarter,
    /// Year frequency
    #[serde(rename = "Y")]
    Year,
}

impl fmt::Display for Frequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Frequency::Nanosecond => write!(f, "ns"),
            Frequency::Microsecond => write!(f, "us"),
            Frequency::Millisecond => write!(f, "ms"),
            Frequency::Second => write!(f, "s"),
            Frequency::Minute => write!(f, "min"),
            Frequency::Hour => write!(f, "h"),
            Frequency::Day => write!(f, "d"),
            Frequency::Week => write!(f, "w"),
            Frequency::Month => write!(f, "M"),
            Frequency::Quarter => write!(f, "Q"),
            Frequency::Year => write!(f, "Y"),
        }
    }
}

/// Aggregation methods for data resampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AggregationMethod {
    /// Mean/average aggregation
    Mean,
    /// Median aggregation
    Median,
    /// Sum aggregation
    Sum,
    /// Minimum value aggregation
    Min,
    /// Maximum value aggregation
    Max,
    /// Count aggregation
    Count,
    /// Standard deviation aggregation
    Std,
    /// Variance aggregation
    Var,
    /// First value aggregation
    First,
    /// Last value aggregation
    Last,
}

impl fmt::Display for AggregationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregationMethod::Mean => write!(f, "mean"),
            AggregationMethod::Median => write!(f, "median"),
            AggregationMethod::Sum => write!(f, "sum"),
            AggregationMethod::Min => write!(f, "min"),
            AggregationMethod::Max => write!(f, "max"),
            AggregationMethod::Count => write!(f, "count"),
            AggregationMethod::Std => write!(f, "std"),
            AggregationMethod::Var => write!(f, "var"),
            AggregationMethod::First => write!(f, "first"),
            AggregationMethod::Last => write!(f, "last"),
        }
    }
}

/// Quality score categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityScore {
    /// Excellent quality (0.9 - 1.0)
    Excellent,
    /// Good quality (0.7 - 0.9)
    Good,
    /// Fair quality (0.5 - 0.7)
    Fair,
    /// Poor quality (0.0 - 0.5)
    Poor,
}

impl QualityScore {
    /// Create a quality score from a numeric value
    pub fn from_score(score: f64) -> Self {
        if score >= 0.9 {
            QualityScore::Excellent
        } else if score >= 0.7 {
            QualityScore::Good
        } else if score >= 0.5 {
            QualityScore::Fair
        } else {
            QualityScore::Poor
        }
    }

    /// Get the numeric range for this quality score
    pub fn score_range(&self) -> (f64, f64) {
        match self {
            QualityScore::Excellent => (0.9, 1.0),
            QualityScore::Good => (0.7, 0.9),
            QualityScore::Fair => (0.5, 0.7),
            QualityScore::Poor => (0.0, 0.5),
        }
    }
}

impl fmt::Display for QualityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityScore::Excellent => write!(f, "excellent"),
            QualityScore::Good => write!(f, "good"),
            QualityScore::Fair => write!(f, "fair"),
            QualityScore::Poor => write!(f, "poor"),
        }
    }
}

/// Statistical distribution types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Distribution {
    /// Normal/Gaussian distribution
    Normal,
    /// Uniform distribution
    Uniform,
    /// Exponential distribution
    Exponential,
    /// Poisson distribution
    Poisson,
    /// Beta distribution
    Beta,
    /// Gamma distribution
    Gamma,
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Distribution::Normal => write!(f, "normal"),
            Distribution::Uniform => write!(f, "uniform"),
            Distribution::Exponential => write!(f, "exponential"),
            Distribution::Poisson => write!(f, "poisson"),
            Distribution::Beta => write!(f, "beta"),
            Distribution::Gamma => write!(f, "gamma"),
        }
    }
}

impl Default for Distribution {
    fn default() -> Self {
        Distribution::Normal
    }
}

/// Trend types for time series generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrendType {
    /// No trend
    None,
    /// Linear trend
    Linear,
    /// Quadratic trend
    Quadratic,
    /// Exponential trend
    Exponential,
    /// Logarithmic trend
    Logarithmic,
}

impl fmt::Display for TrendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendType::None => write!(f, "none"),
            TrendType::Linear => write!(f, "linear"),
            TrendType::Quadratic => write!(f, "quadratic"),
            TrendType::Exponential => write!(f, "exponential"),
            TrendType::Logarithmic => write!(f, "logarithmic"),
        }
    }
}

impl Default for TrendType {
    fn default() -> Self {
        TrendType::None
    }
}

/// Pattern types for time series generation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PatternType {
    /// Random pattern
    Random,
    /// Seasonal pattern
    Seasonal,
    /// Trending pattern
    Trending,
    /// Cyclic pattern
    Cyclic,
    /// Custom pattern
    Custom(String),
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternType::Random => write!(f, "random"),
            PatternType::Seasonal => write!(f, "seasonal"),
            PatternType::Trending => write!(f, "trending"),
            PatternType::Cyclic => write!(f, "cyclic"),
            PatternType::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

impl Default for PatternType {
    fn default() -> Self {
        PatternType::Random
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_format_display() {
        assert_eq!(DataFormat::Json.to_string(), "json");
        assert_eq!(DataFormat::Csv.to_string(), "csv");
        assert_eq!(DataFormat::Parquet.to_string(), "parquet");
    }

    #[test]
    fn test_frequency_display() {
        assert_eq!(Frequency::Second.to_string(), "s");
        assert_eq!(Frequency::Minute.to_string(), "min");
        assert_eq!(Frequency::Hour.to_string(), "h");
    }

    #[test]
    fn test_quality_score_from_score() {
        assert_eq!(QualityScore::from_score(0.95), QualityScore::Excellent);
        assert_eq!(QualityScore::from_score(0.8), QualityScore::Good);
        assert_eq!(QualityScore::from_score(0.6), QualityScore::Fair);
        assert_eq!(QualityScore::from_score(0.3), QualityScore::Poor);
    }

    #[test]
    fn test_quality_score_ranges() {
        assert_eq!(QualityScore::Excellent.score_range(), (0.9, 1.0));
        assert_eq!(QualityScore::Good.score_range(), (0.7, 0.9));
        assert_eq!(QualityScore::Fair.score_range(), (0.5, 0.7));
        assert_eq!(QualityScore::Poor.score_range(), (0.0, 0.5));
    }

    #[test]
    fn test_serde() {
        let format = DataFormat::Json;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"json\"");

        let deserialized: DataFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, format);
    }
}