/**
 * Time series data structures and utilities for the TSIOT JavaScript SDK.
 * 
 * This module provides classes for representing and manipulating time series data,
 * including data points, metadata, and various serialization formats.
 */

const dayjs = require('dayjs');
const utc = require('dayjs/plugin/utc');
const { ValidationError, utils } = require('./index');

// Extend dayjs with UTC plugin
dayjs.extend(utc);

/**
 * Represents a single data point in a time series
 */
class DataPoint {
  /**
   * Create a new data point
   * @param {Date|string} timestamp - The timestamp of the data point
   * @param {number} value - The numerical value
   * @param {number} quality - Data quality score (0.0 to 1.0)
   * @param {Object} metadata - Optional metadata object
   */
  constructor(timestamp, value, quality = 1.0, metadata = {}) {
    this.timestamp = timestamp instanceof Date ? timestamp : new Date(timestamp);
    this.value = utils.validatePositiveNumber(value, 'value');
    
    if (quality < 0 || quality > 1) {
      throw new ValidationError('quality must be between 0.0 and 1.0', 'quality', quality);
    }
    this.quality = quality;
    
    this.metadata = metadata || {};
    
    // Validate the data point
    this._validate();
  }

  /**
   * Validate the data point
   * @private
   */
  _validate() {
    if (!(this.timestamp instanceof Date) || isNaN(this.timestamp.getTime())) {
      throw new ValidationError('timestamp must be a valid Date', 'timestamp');
    }
    
    if (typeof this.value !== 'number' || isNaN(this.value)) {
      throw new ValidationError('value must be a valid number', 'value', this.value);
    }
  }

  /**
   * Check if this data point is valid
   * @returns {boolean} True if valid
   */
  isValid() {
    return !isNaN(this.value) && 
           isFinite(this.value) && 
           this.quality >= 0 && 
           this.quality <= 1 &&
           this.timestamp instanceof Date &&
           !isNaN(this.timestamp.getTime());
  }

  /**
   * Convert to plain object
   * @returns {Object} Plain object representation
   */
  toObject() {
    return {
      timestamp: utils.formatTimestamp(this.timestamp),
      value: this.value,
      quality: this.quality,
      metadata: this.metadata
    };
  }

  /**
   * Create from plain object
   * @param {Object} obj - Object with timestamp, value, quality, metadata
   * @returns {DataPoint} New DataPoint instance
   */
  static fromObject(obj) {
    return new DataPoint(
      utils.parseTimestamp(obj.timestamp),
      obj.value,
      obj.quality || 1.0,
      obj.metadata || {}
    );
  }

  /**
   * String representation
   * @returns {string} String representation
   */
  toString() {
    return `DataPoint(${utils.formatTimestamp(this.timestamp)}, ${this.value})`;
  }
}

/**
 * Metadata for a time series
 */
class TimeSeriesMetadata {
  /**
   * Create new time series metadata
   * @param {string} seriesId - Unique identifier for the time series
   * @param {Object} options - Additional metadata options
   */
  constructor(seriesId, options = {}) {
    this.seriesId = utils.validateStringNotEmpty(seriesId, 'seriesId');
    this.name = options.name || null;
    this.description = options.description || null;
    this.unit = options.unit || null;
    this.frequency = options.frequency || null;
    this.tags = options.tags || {};
    this.source = options.source || null;
    this.createdAt = options.createdAt || new Date();
    this.updatedAt = options.updatedAt || new Date();
  }

  /**
   * Update the updatedAt timestamp
   */
  touch() {
    this.updatedAt = new Date();
  }

  /**
   * Convert to plain object
   * @returns {Object} Plain object representation
   */
  toObject() {
    const obj = {
      seriesId: this.seriesId,
      tags: this.tags,
      createdAt: utils.formatTimestamp(this.createdAt),
      updatedAt: utils.formatTimestamp(this.updatedAt)
    };

    // Add optional fields if present
    if (this.name) obj.name = this.name;
    if (this.description) obj.description = this.description;
    if (this.unit) obj.unit = this.unit;
    if (this.frequency) obj.frequency = this.frequency;
    if (this.source) obj.source = this.source;

    return obj;
  }

  /**
   * Create from plain object
   * @param {Object} obj - Object with metadata fields
   * @returns {TimeSeriesMetadata} New metadata instance
   */
  static fromObject(obj) {
    return new TimeSeriesMetadata(obj.seriesId, {
      name: obj.name,
      description: obj.description,
      unit: obj.unit,
      frequency: obj.frequency,
      tags: obj.tags || {},
      source: obj.source,
      createdAt: obj.createdAt ? utils.parseTimestamp(obj.createdAt) : undefined,
      updatedAt: obj.updatedAt ? utils.parseTimestamp(obj.updatedAt) : undefined
    });
  }
}

/**
 * A time series containing data points and metadata
 */
class TimeSeries {
  /**
   * Create a new time series
   * @param {Array<DataPoint>} dataPoints - Array of data points
   * @param {TimeSeriesMetadata} metadata - Time series metadata
   * @param {boolean} autoSort - Whether to automatically sort by timestamp
   */
  constructor(dataPoints = [], metadata = null, autoSort = true) {
    this.dataPoints = Array.isArray(dataPoints) ? dataPoints : [];
    this.metadata = metadata || new TimeSeriesMetadata('default');
    this.autoSort = autoSort;

    if (this.autoSort && this.dataPoints.length > 0) {
      this._sortDataPoints();
    }

    this._validate();
  }

  /**
   * Validate the time series
   * @private
   */
  _validate() {
    if (!Array.isArray(this.dataPoints)) {
      throw new ValidationError('dataPoints must be an array');
    }

    this.dataPoints.forEach((point, index) => {
      if (!(point instanceof DataPoint)) {
        throw new ValidationError(`dataPoints[${index}] must be a DataPoint instance`);
      }
    });

    if (!(this.metadata instanceof TimeSeriesMetadata)) {
      throw new ValidationError('metadata must be a TimeSeriesMetadata instance');
    }
  }

  /**
   * Sort data points by timestamp
   * @private
   */
  _sortDataPoints() {
    this.dataPoints.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  /**
   * Get the number of data points
   * @returns {number} Number of data points
   */
  get length() {
    return this.dataPoints.length;
  }

  /**
   * Check if the time series is empty
   * @returns {boolean} True if empty
   */
  get isEmpty() {
    return this.dataPoints.length === 0;
  }

  /**
   * Get the first timestamp
   * @returns {Date|null} First timestamp or null if empty
   */
  get startTime() {
    return this.dataPoints.length > 0 ? this.dataPoints[0].timestamp : null;
  }

  /**
   * Get the last timestamp
   * @returns {Date|null} Last timestamp or null if empty
   */
  get endTime() {
    return this.dataPoints.length > 0 ? this.dataPoints[this.dataPoints.length - 1].timestamp : null;
  }

  /**
   * Get the duration in milliseconds
   * @returns {number|null} Duration in milliseconds or null if less than 2 points
   */
  get duration() {
    if (this.dataPoints.length < 2) return null;
    return this.endTime.getTime() - this.startTime.getTime();
  }

  /**
   * Get all values as an array
   * @returns {Array<number>} Array of values
   */
  get values() {
    return this.dataPoints.map(dp => dp.value);
  }

  /**
   * Get all timestamps as an array
   * @returns {Array<Date>} Array of timestamps
   */
  get timestamps() {
    return this.dataPoints.map(dp => dp.timestamp);
  }

  /**
   * Add a data point to the time series
   * @param {DataPoint} dataPoint - Data point to add
   */
  addPoint(dataPoint) {
    if (!(dataPoint instanceof DataPoint)) {
      throw new ValidationError('dataPoint must be a DataPoint instance');
    }

    this.dataPoints.push(dataPoint);
    
    if (this.autoSort) {
      this._sortDataPoints();
    }

    this.metadata.touch();
  }

  /**
   * Add multiple data points
   * @param {Array<DataPoint>} dataPoints - Array of data points to add
   */
  addPoints(dataPoints) {
    if (!Array.isArray(dataPoints)) {
      throw new ValidationError('dataPoints must be an array');
    }

    dataPoints.forEach(point => {
      if (!(point instanceof DataPoint)) {
        throw new ValidationError('All items must be DataPoint instances');
      }
    });

    this.dataPoints.push(...dataPoints);
    
    if (this.autoSort) {
      this._sortDataPoints();
    }

    this.metadata.touch();
  }

  /**
   * Get data point at specific index
   * @param {number} index - Index of the data point
   * @returns {DataPoint} Data point at index
   */
  getPointAtIndex(index) {
    if (index < 0 || index >= this.dataPoints.length) {
      throw new ValidationError(`Index ${index} out of range for time series of length ${this.dataPoints.length}`);
    }
    return this.dataPoints[index];
  }

  /**
   * Get data points within a time range
   * @param {Date} startTime - Start of time range
   * @param {Date} endTime - End of time range
   * @param {boolean} inclusive - Whether to include boundary points
   * @returns {Array<DataPoint>} Data points in range
   */
  getPointsInRange(startTime, endTime, inclusive = true) {
    if (!(startTime instanceof Date) || !(endTime instanceof Date)) {
      throw new ValidationError('startTime and endTime must be Date objects');
    }

    if (startTime >= endTime) {
      throw new ValidationError('startTime must be before endTime');
    }

    return this.dataPoints.filter(point => {
      if (inclusive) {
        return point.timestamp >= startTime && point.timestamp <= endTime;
      } else {
        return point.timestamp > startTime && point.timestamp < endTime;
      }
    });
  }

  /**
   * Create a slice of the time series
   * @param {number} startIndex - Start index (inclusive)
   * @param {number} endIndex - End index (exclusive)
   * @returns {TimeSeries} New TimeSeries with sliced data
   */
  slice(startIndex = 0, endIndex = this.dataPoints.length) {
    const slicedPoints = this.dataPoints.slice(startIndex, endIndex);
    return new TimeSeries(slicedPoints, this.metadata, false); // Already sorted
  }

  /**
   * Filter data points by quality threshold
   * @param {number} minQuality - Minimum quality threshold (0.0 to 1.0)
   * @returns {TimeSeries} New TimeSeries with filtered data
   */
  filterByQuality(minQuality) {
    if (minQuality < 0 || minQuality > 1) {
      throw new ValidationError('minQuality must be between 0.0 and 1.0');
    }

    const filteredPoints = this.dataPoints.filter(dp => dp.quality >= minQuality);
    return new TimeSeries(filteredPoints, this.metadata, false); // Already sorted
  }

  /**
   * Calculate basic statistics
   * @returns {Object} Basic statistics object
   */
  basicStatistics() {
    if (this.isEmpty) {
      return {};
    }

    const values = this.values;
    const n = values.length;

    // Sort values for percentile calculations
    const sortedValues = [...values].sort((a, b) => a - b);

    // Basic measures
    const sum = values.reduce((acc, val) => acc + val, 0);
    const mean = sum / n;
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);

    const min = sortedValues[0];
    const max = sortedValues[n - 1];
    const median = n % 2 === 0 
      ? (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2
      : sortedValues[Math.floor(n / 2)];

    // Percentiles
    const q1Index = Math.floor(0.25 * (n - 1));
    const q3Index = Math.floor(0.75 * (n - 1));
    const q1 = sortedValues[q1Index];
    const q3 = sortedValues[q3Index];
    const iqr = q3 - q1;

    // Skewness and kurtosis (simplified calculations)
    const skewness = values.reduce((acc, val) => {
      return acc + Math.pow((val - mean) / stdDev, 3);
    }, 0) / n;

    const kurtosis = values.reduce((acc, val) => {
      return acc + Math.pow((val - mean) / stdDev, 4);
    }, 0) / n - 3; // Excess kurtosis

    return {
      count: n,
      mean,
      median,
      std: stdDev,
      var: variance,
      min,
      max,
      range: max - min,
      q1,
      q3,
      iqr,
      skewness,
      kurtosis
    };
  }

  /**
   * Validate the time series data
   * @returns {Array<string>} Array of validation issues
   */
  validate() {
    const issues = [];

    if (this.isEmpty) {
      issues.push('Time series is empty');
      return issues;
    }

    // Check for invalid data points
    const invalidPoints = this.dataPoints.filter(dp => !dp.isValid());
    if (invalidPoints.length > 0) {
      issues.push(`${invalidPoints.length} invalid data points found`);
    }

    // Check for duplicate timestamps
    const timestamps = this.timestamps.map(ts => ts.getTime());
    const uniqueTimestamps = new Set(timestamps);
    if (uniqueTimestamps.size !== timestamps.length) {
      issues.push('Duplicate timestamps found');
    }

    // Check for NaN or infinite values
    const invalidValues = this.values.filter(val => !isFinite(val));
    if (invalidValues.length > 0) {
      issues.push(`${invalidValues.length} invalid values (NaN or infinite) found`);
    }

    return issues;
  }

  /**
   * Convert to plain object
   * @returns {Object} Plain object representation
   */
  toObject() {
    return {
      dataPoints: this.dataPoints.map(dp => dp.toObject()),
      metadata: this.metadata.toObject(),
      length: this.length
    };
  }

  /**
   * Create from plain object
   * @param {Object} obj - Object with dataPoints and metadata
   * @returns {TimeSeries} New TimeSeries instance
   */
  static fromObject(obj) {
    const dataPoints = obj.dataPoints.map(dpData => DataPoint.fromObject(dpData));
    const metadata = obj.metadata ? TimeSeriesMetadata.fromObject(obj.metadata) : null;
    return new TimeSeries(dataPoints, metadata);
  }

  /**
   * Convert to JSON string
   * @param {number} indent - Indentation for pretty printing
   * @returns {string} JSON string
   */
  toJSON(indent = null) {
    return JSON.stringify(this.toObject(), null, indent);
  }

  /**
   * Create from JSON string
   * @param {string} jsonStr - JSON string
   * @returns {TimeSeries} New TimeSeries instance
   */
  static fromJSON(jsonStr) {
    const obj = JSON.parse(jsonStr);
    return TimeSeries.fromObject(obj);
  }

  /**
   * Convert to CSV string
   * @param {boolean} includeMetadata - Whether to include metadata as JSON
   * @returns {string} CSV string
   */
  toCSV(includeMetadata = true) {
    const headers = ['timestamp', 'value', 'quality'];
    if (includeMetadata) {
      headers.push('metadata');
    }

    const rows = [headers.join(',')];

    this.dataPoints.forEach(dp => {
      const row = [
        utils.formatTimestamp(dp.timestamp),
        dp.value,
        dp.quality
      ];

      if (includeMetadata) {
        row.push(Object.keys(dp.metadata).length > 0 ? JSON.stringify(dp.metadata) : '');
      }

      rows.push(row.join(','));
    });

    return rows.join('\n');
  }

  /**
   * Create from CSV string
   * @param {string} csvStr - CSV string
   * @param {string} seriesId - Series ID for metadata
   * @param {boolean} hasHeader - Whether CSV has header row
   * @returns {TimeSeries} New TimeSeries instance
   */
  static fromCSV(csvStr, seriesId = 'default', hasHeader = true) {
    const lines = csvStr.trim().split('\n');
    const dataPoints = [];

    const startIndex = hasHeader ? 1 : 0;

    for (let i = startIndex; i < lines.length; i++) {
      const row = lines[i].split(',');
      if (row.length < 2) continue;

      const timestamp = utils.parseTimestamp(row[0]);
      const value = parseFloat(row[1]);
      const quality = row.length > 2 && row[2] ? parseFloat(row[2]) : 1.0;

      let metadata = {};
      if (row.length > 3 && row[3]) {
        try {
          metadata = JSON.parse(row[3]);
        } catch (e) {
          metadata = { raw: row[3] };
        }
      }

      dataPoints.push(new DataPoint(timestamp, value, quality, metadata));
    }

    const tsMetadata = new TimeSeriesMetadata(seriesId);
    return new TimeSeries(dataPoints, tsMetadata);
  }

  /**
   * String representation
   * @returns {string} String representation
   */
  toString() {
    return `TimeSeries(id=${this.metadata.seriesId}, length=${this.length})`;
  }

  /**
   * Iterator support
   * @returns {Iterator<DataPoint>} Iterator over data points
   */
  [Symbol.iterator]() {
    return this.dataPoints[Symbol.iterator]();
  }
}

/**
 * Convenience functions for creating time series
 */

/**
 * Create a time series from arrays of values and timestamps
 * @param {Array<number>} values - Array of values
 * @param {Array<Date>} timestamps - Array of timestamps (optional)
 * @param {string} seriesId - Series identifier
 * @param {Object} metadataOptions - Additional metadata options
 * @returns {TimeSeries} New TimeSeries instance
 */
function createTimeSeries(values, timestamps = null, seriesId = 'default', metadataOptions = {}) {
  if (!Array.isArray(values)) {
    throw new ValidationError('values must be an array');
  }

  // Generate timestamps if not provided
  if (!timestamps) {
    const startTime = new Date();
    timestamps = values.map((_, index) => new Date(startTime.getTime() + index * 1000));
  }

  if (values.length !== timestamps.length) {
    throw new ValidationError('values and timestamps arrays must have the same length');
  }

  // Create data points
  const dataPoints = values.map((value, index) => new DataPoint(timestamps[index], value));

  // Create metadata
  const metadata = new TimeSeriesMetadata(seriesId, metadataOptions);

  return new TimeSeries(dataPoints, metadata);
}

/**
 * Create an empty time series with metadata
 * @param {string} seriesId - Series identifier
 * @param {Object} metadataOptions - Additional metadata options
 * @returns {TimeSeries} Empty TimeSeries instance
 */
function createEmptyTimeSeries(seriesId = 'default', metadataOptions = {}) {
  const metadata = new TimeSeriesMetadata(seriesId, metadataOptions);
  return new TimeSeries([], metadata);
}

/**
 * Create a time series with regular intervals
 * @param {Array<number>} values - Array of values
 * @param {Date} startTime - Start timestamp
 * @param {number} intervalMs - Interval in milliseconds
 * @param {string} seriesId - Series identifier
 * @param {Object} metadataOptions - Additional metadata options
 * @returns {TimeSeries} New TimeSeries instance
 */
function createTimeSeriesWithInterval(values, startTime, intervalMs, seriesId = 'default', metadataOptions = {}) {
  if (!Array.isArray(values)) {
    throw new ValidationError('values must be an array');
  }

  if (!(startTime instanceof Date)) {
    throw new ValidationError('startTime must be a Date object');
  }

  utils.validatePositiveNumber(intervalMs, 'intervalMs');

  // Generate timestamps with regular intervals
  const timestamps = values.map((_, index) => new Date(startTime.getTime() + index * intervalMs));

  return createTimeSeries(values, timestamps, seriesId, metadataOptions);
}

module.exports = {
  DataPoint,
  TimeSeriesMetadata,
  TimeSeries,
  createTimeSeries,
  createEmptyTimeSeries,
  createTimeSeriesWithInterval
};