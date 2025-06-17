//! Time series data structures and utilities.

use crate::error::{Error, Result};
use crate::types::{DataFormat, Frequency};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A single data point in a time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp of the data point
    pub timestamp: DateTime<Utc>,
    /// Numerical value
    pub value: f64,
    /// Data quality score (0.0 to 1.0)
    pub quality: f64,
    /// Optional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(timestamp: DateTime<Utc>, value: f64) -> Self {
        Self {
            timestamp,
            value,
            quality: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a new data point with quality score
    pub fn with_quality(timestamp: DateTime<Utc>, value: f64, quality: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&quality) {
            return Err(Error::validation(
                "quality must be between 0.0 and 1.0",
                Some("quality".to_string()),
                Some(quality.to_string()),
            ));
        }

        Ok(Self {
            timestamp,
            value,
            quality,
            metadata: HashMap::new(),
        })
    }

    /// Add metadata to the data point
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if this data point is valid
    pub fn is_valid(&self) -> bool {
        !self.value.is_nan() && !self.value.is_infinite() && (0.0..=1.0).contains(&self.quality)
    }
}

impl fmt::Display for DataPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataPoint({}, {})", self.timestamp, self.value)
    }
}

/// Metadata for a time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeriesMetadata {
    /// Unique identifier for the time series
    pub series_id: String,
    /// Human-readable name
    pub name: Option<String>,
    /// Description of the time series
    pub description: Option<String>,
    /// Unit of measurement
    pub unit: Option<String>,
    /// Expected frequency/granularity
    pub frequency: Option<Frequency>,
    /// Tags for categorization
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub tags: HashMap<String, String>,
    /// Data source information
    pub source: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl TimeSeriesMetadata {
    /// Create new metadata with required fields
    pub fn new<S: Into<String>>(series_id: S) -> Self {
        let now = Utc::now();
        Self {
            series_id: series_id.into(),
            name: None,
            description: None,
            unit: None,
            frequency: None,
            tags: HashMap::new(),
            source: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the name
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the unit
    pub fn with_unit<S: Into<String>>(mut self, unit: S) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Set the frequency
    pub fn with_frequency(mut self, frequency: Frequency) -> Self {
        self.frequency = Some(frequency);
        self
    }

    /// Add a tag
    pub fn with_tag<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Set the source
    pub fn with_source<S: Into<String>>(mut self, source: S) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Update the updated_at timestamp
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

/// A time series containing data points and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// List of data points
    pub data_points: Vec<DataPoint>,
    /// Time series metadata
    pub metadata: TimeSeriesMetadata,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new<S: Into<String>>(series_id: S) -> Self {
        Self {
            data_points: Vec::new(),
            metadata: TimeSeriesMetadata::new(series_id),
        }
    }

    /// Create a time series with data points
    pub fn with_data_points<S: Into<String>>(series_id: S, data_points: Vec<DataPoint>) -> Self {
        Self {
            data_points,
            metadata: TimeSeriesMetadata::new(series_id),
        }
    }

    /// Create a time series with metadata
    pub fn with_metadata(metadata: TimeSeriesMetadata) -> Self {
        Self {
            data_points: Vec::new(),
            metadata,
        }
    }

    /// Add a data point to the time series
    pub fn add_point(&mut self, point: DataPoint) {
        self.data_points.push(point);
        self.metadata.touch();
    }

    /// Add multiple data points
    pub fn add_points(&mut self, points: Vec<DataPoint>) {
        self.data_points.extend(points);
        self.metadata.touch();
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.data_points.len()
    }

    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.data_points.is_empty()
    }

    /// Get the first timestamp
    pub fn start_time(&self) -> Option<DateTime<Utc>> {
        self.data_points.first().map(|dp| dp.timestamp)
    }

    /// Get the last timestamp
    pub fn end_time(&self) -> Option<DateTime<Utc>> {
        self.data_points.last().map(|dp| dp.timestamp)
    }

    /// Get the duration of the time series
    pub fn duration(&self) -> Option<chrono::Duration> {
        if let (Some(start), Some(end)) = (self.start_time(), self.end_time()) {
            Some(end - start)
        } else {
            None
        }
    }

    /// Get all values as a vector
    pub fn values(&self) -> Vec<f64> {
        self.data_points.iter().map(|dp| dp.value).collect()
    }

    /// Get all timestamps as a vector
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> {
        self.data_points.iter().map(|dp| dp.timestamp).collect()
    }

    /// Sort data points by timestamp
    pub fn sort_by_timestamp(&mut self) {
        self.data_points.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        self.metadata.touch();
    }

    /// Get data points in a time range
    pub fn points_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<DataPoint>> {
        if start >= end {
            return Err(Error::validation(
                "start time must be before end time",
                Some("time_range".to_string()),
                None,
            ));
        }

        let points = self
            .data_points
            .iter()
            .filter(|dp| dp.timestamp >= start && dp.timestamp <= end)
            .cloned()
            .collect();

        Ok(points)
    }

    /// Get a slice of the time series
    pub fn slice(&self, start_idx: usize, end_idx: usize) -> Result<TimeSeries> {
        if start_idx >= self.data_points.len() {
            return Err(Error::validation(
                "start index out of bounds",
                Some("start_idx".to_string()),
                Some(start_idx.to_string()),
            ));
        }

        if end_idx > self.data_points.len() {
            return Err(Error::validation(
                "end index out of bounds",
                Some("end_idx".to_string()),
                Some(end_idx.to_string()),
            ));
        }

        if start_idx >= end_idx {
            return Err(Error::validation(
                "start index must be less than end index",
                Some("indices".to_string()),
                None,
            ));
        }

        let sliced_points = self.data_points[start_idx..end_idx].to_vec();
        let mut metadata = self.metadata.clone();
        metadata.touch();

        Ok(TimeSeries {
            data_points: sliced_points,
            metadata,
        })
    }

    /// Filter data points by quality threshold
    pub fn filter_by_quality(&self, min_quality: f64) -> Result<TimeSeries> {
        if !(0.0..=1.0).contains(&min_quality) {
            return Err(Error::validation(
                "min_quality must be between 0.0 and 1.0",
                Some("min_quality".to_string()),
                Some(min_quality.to_string()),
            ));
        }

        let filtered_points = self
            .data_points
            .iter()
            .filter(|dp| dp.quality >= min_quality)
            .cloned()
            .collect();

        let mut metadata = self.metadata.clone();
        metadata.touch();

        Ok(TimeSeries {
            data_points: filtered_points,
            metadata,
        })
    }

    /// Calculate basic statistics
    pub fn basic_statistics(&self) -> Result<BasicStatistics> {
        if self.is_empty() {
            return Err(Error::validation(
                "cannot calculate statistics for empty time series",
                None,
                None,
            ));
        }

        let values = self.values();
        let n = values.len() as f64;

        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        // Calculate percentiles
        let q1_idx = (0.25 * (sorted_values.len() - 1) as f64) as usize;
        let q3_idx = (0.75 * (sorted_values.len() - 1) as f64) as usize;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];

        Ok(BasicStatistics {
            count: values.len(),
            mean,
            median,
            std_dev,
            variance,
            min,
            max,
            range: max - min,
            q1,
            q3,
            iqr: q3 - q1,
        })
    }

    /// Validate the time series data
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.is_empty() {
            issues.push("Time series is empty".to_string());
            return issues;
        }

        // Check for invalid data points
        let invalid_count = self.data_points.iter().filter(|dp| !dp.is_valid()).count();
        if invalid_count > 0 {
            issues.push(format!("{} invalid data points found", invalid_count));
        }

        // Check for duplicate timestamps
        let mut timestamps = self.timestamps();
        timestamps.sort();
        let unique_count = {
            timestamps.dedup();
            timestamps.len()
        };
        if unique_count != self.len() {
            issues.push("Duplicate timestamps found".to_string());
        }

        // Check for large gaps in timestamps (if frequency is known)
        if let Some(frequency) = &self.metadata.frequency {
            // This is a simplified check - in reality, you'd implement proper gap detection
            let _ = frequency; // Placeholder
        }

        issues
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    /// Create from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }

    /// Convert to CSV string
    #[cfg(feature = "csv")]
    pub fn to_csv(&self) -> Result<String> {
        use std::io::Write;

        let mut writer = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(Vec::new());

        // Write header
        writer.write_record(&["timestamp", "value", "quality"])?;

        // Write data points
        for point in &self.data_points {
            writer.write_record(&[
                point.timestamp.to_rfc3339(),
                point.value.to_string(),
                point.quality.to_string(),
            ])?;
        }

        let data = writer.into_inner().map_err(|e| Error::internal(e.to_string()))?;
        String::from_utf8(data).map_err(|e| Error::internal(e.to_string()))
    }

    /// Create from CSV string
    #[cfg(feature = "csv")]
    pub fn from_csv(csv_data: &str, series_id: String) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_data.as_bytes());

        let mut data_points = Vec::new();

        for result in reader.records() {
            let record = result?;
            if record.len() < 2 {
                continue;
            }

            let timestamp_str = &record[0];
            let value_str = &record[1];

            let timestamp = DateTime::parse_from_rfc3339(timestamp_str)
                .map_err(|e| Error::validation(format!("Invalid timestamp: {}", e), None, None))?
                .with_timezone(&Utc);

            let value = value_str.parse::<f64>()
                .map_err(|e| Error::validation(format!("Invalid value: {}", e), None, None))?;

            let quality = if record.len() > 2 {
                record[2].parse::<f64>().unwrap_or(1.0)
            } else {
                1.0
            };

            data_points.push(DataPoint::with_quality(timestamp, value, quality)?);
        }

        Ok(TimeSeries::with_data_points(series_id, data_points))
    }
}

impl fmt::Display for TimeSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TimeSeries(id={}, length={})",
            self.metadata.series_id,
            self.len()
        )
    }
}

/// Basic statistical measures for a time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasicStatistics {
    /// Number of data points
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Range (max - min)
    pub range: f64,
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f64,
}

impl fmt::Display for BasicStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BasicStatistics(count={}, mean={:.2}, std={:.2}, min={:.2}, max={:.2})",
            self.count, self.mean, self.std_dev, self.min, self.max
        )
    }
}

/// Convenience functions for creating time series

/// Create a time series from values and timestamps
pub fn create_time_series(
    series_id: String,
    values: Vec<f64>,
    timestamps: Vec<DateTime<Utc>>,
) -> Result<TimeSeries> {
    if values.len() != timestamps.len() {
        return Err(Error::validation(
            "values and timestamps must have the same length",
            None,
            None,
        ));
    }

    let data_points = values
        .into_iter()
        .zip(timestamps)
        .map(|(value, timestamp)| DataPoint::new(timestamp, value))
        .collect();

    Ok(TimeSeries::with_data_points(series_id, data_points))
}

/// Create a time series from values with auto-generated timestamps
pub fn create_time_series_with_interval(
    series_id: String,
    values: Vec<f64>,
    start_time: DateTime<Utc>,
    interval: chrono::Duration,
) -> TimeSeries {
    let data_points = values
        .into_iter()
        .enumerate()
        .map(|(i, value)| {
            let timestamp = start_time + interval * i as i32;
            DataPoint::new(timestamp, value)
        })
        .collect();

    TimeSeries::with_data_points(series_id, data_points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_data_point_creation() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let point = DataPoint::new(timestamp, 42.0);

        assert_eq!(point.timestamp, timestamp);
        assert_eq!(point.value, 42.0);
        assert_eq!(point.quality, 1.0);
        assert!(point.is_valid());
    }

    #[test]
    fn test_data_point_with_quality() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let point = DataPoint::with_quality(timestamp, 42.0, 0.8).unwrap();

        assert_eq!(point.quality, 0.8);
        assert!(point.is_valid());

        // Test invalid quality
        let result = DataPoint::with_quality(timestamp, 42.0, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_time_series_creation() {
        let ts = TimeSeries::new("test-series");
        assert_eq!(ts.metadata.series_id, "test-series");
        assert!(ts.is_empty());
        assert_eq!(ts.len(), 0);
    }

    #[test]
    fn test_time_series_with_data() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let points = vec![
            DataPoint::new(timestamp, 1.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(1), 2.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(2), 3.0),
        ];

        let ts = TimeSeries::with_data_points("test", points);
        assert_eq!(ts.len(), 3);
        assert_eq!(ts.values(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_basic_statistics() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let points = vec![
            DataPoint::new(timestamp, 1.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(1), 2.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(2), 3.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(3), 4.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(4), 5.0),
        ];

        let ts = TimeSeries::with_data_points("test", points);
        let stats = ts.basic_statistics().unwrap();

        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
    }

    #[test]
    fn test_slice() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let points = vec![
            DataPoint::new(timestamp, 1.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(1), 2.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(2), 3.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(3), 4.0),
            DataPoint::new(timestamp + chrono::Duration::seconds(4), 5.0),
        ];

        let ts = TimeSeries::with_data_points("test", points);
        let sliced = ts.slice(1, 4).unwrap();

        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.values(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_filter_by_quality() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let points = vec![
            DataPoint::with_quality(timestamp, 1.0, 0.9).unwrap(),
            DataPoint::with_quality(timestamp + chrono::Duration::seconds(1), 2.0, 0.5).unwrap(),
            DataPoint::with_quality(timestamp + chrono::Duration::seconds(2), 3.0, 0.8).unwrap(),
        ];

        let ts = TimeSeries::with_data_points("test", points);
        let filtered = ts.filter_by_quality(0.7).unwrap();

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.values(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_json_serialization() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let points = vec![DataPoint::new(timestamp, 42.0)];
        let ts = TimeSeries::with_data_points("test", points);

        let json = ts.to_json().unwrap();
        let deserialized = TimeSeries::from_json(&json).unwrap();

        assert_eq!(ts.metadata.series_id, deserialized.metadata.series_id);
        assert_eq!(ts.len(), deserialized.len());
        assert_eq!(ts.values(), deserialized.values());
    }

    #[test]
    fn test_validation() {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0);
        let points = vec![
            DataPoint::new(timestamp, 1.0),
            DataPoint::new(timestamp, f64::NAN), // Invalid
            DataPoint::new(timestamp, 3.0),
        ];

        let ts = TimeSeries::with_data_points("test", points);
        let issues = ts.validate();

        assert!(!issues.is_empty());
        assert!(issues[0].contains("invalid data points"));
    }
}