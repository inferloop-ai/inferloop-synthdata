"""
Time series data structures and utilities for the TSIOT Python SDK.

This module provides classes for representing and manipulating time series data,
including data points, metadata, and various serialization formats.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Iterator
from enum import Enum
import json
import csv
import io
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .utils import (
    ValidationError,
    TSIOTError,
    validate_positive_number,
    validate_string_not_empty,
    validate_enum_value,
    format_timestamp,
    parse_timestamp,
    current_timestamp
)


class DataFormat(Enum):
    """Supported data formats for time series serialization."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    ARROW = "arrow"


class Frequency(Enum):
    """Time series frequency/granularity options."""
    NANOSECOND = "ns"
    MICROSECOND = "us"
    MILLISECOND = "ms"
    SECOND = "s"
    MINUTE = "min"
    HOUR = "h"
    DAY = "d"
    WEEK = "w"
    MONTH = "M"
    QUARTER = "Q"
    YEAR = "Y"


class AggregationMethod(Enum):
    """Data aggregation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"


@dataclass
class DataPoint:
    """
    Represents a single data point in a time series.
    
    Attributes:
        timestamp: The timestamp of the data point
        value: The numerical value
        metadata: Optional metadata dictionary
        quality: Data quality score (0.0 to 1.0)
    """
    timestamp: datetime
    value: float
    metadata: Optional[Dict[str, Any]] = None
    quality: float = 1.0
    
    def __post_init__(self):
        """Validate data point after initialization."""
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("timestamp must be a datetime object", field="timestamp")
        
        if not isinstance(self.value, (int, float)):
            raise ValidationError("value must be a number", field="value", value=self.value)
        
        if self.quality < 0.0 or self.quality > 1.0:
            raise ValidationError("quality must be between 0.0 and 1.0", field="quality", value=self.quality)
        
        # Ensure timestamp has timezone info
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Initialize metadata if None
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert data point to dictionary."""
        return {
            "timestamp": format_timestamp(self.timestamp),
            "value": self.value,
            "metadata": self.metadata,
            "quality": self.quality
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPoint":
        """Create data point from dictionary."""
        return cls(
            timestamp=parse_timestamp(data["timestamp"]),
            value=float(data["value"]),
            metadata=data.get("metadata"),
            quality=data.get("quality", 1.0)
        )
    
    def __str__(self) -> str:
        return f"DataPoint({format_timestamp(self.timestamp)}, {self.value})"
    
    def __repr__(self) -> str:
        return f"DataPoint(timestamp={self.timestamp!r}, value={self.value}, quality={self.quality})"


@dataclass
class TimeSeriesMetadata:
    """
    Metadata for a time series.
    
    Attributes:
        series_id: Unique identifier for the time series
        name: Human-readable name
        description: Description of the time series
        unit: Unit of measurement
        frequency: Expected frequency/granularity
        tags: Dictionary of tags for categorization
        source: Data source information
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    series_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    frequency: Optional[Frequency] = None
    tags: Dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        self.series_id = validate_string_not_empty(self.series_id, "series_id")
        
        if self.frequency is not None:
            self.frequency = validate_enum_value(self.frequency, Frequency, "frequency")
        
        # Set timestamps if not provided
        now = current_timestamp()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        
        # Ensure timestamps have timezone info
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        result = {
            "series_id": self.series_id,
            "tags": self.tags,
            "created_at": format_timestamp(self.created_at),
            "updated_at": format_timestamp(self.updated_at)
        }
        
        # Add optional fields if present
        if self.name is not None:
            result["name"] = self.name
        if self.description is not None:
            result["description"] = self.description
        if self.unit is not None:
            result["unit"] = self.unit
        if self.frequency is not None:
            result["frequency"] = self.frequency.value
        if self.source is not None:
            result["source"] = self.source
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesMetadata":
        """Create metadata from dictionary."""
        kwargs = {
            "series_id": data["series_id"],
            "tags": data.get("tags", {}),
            "created_at": parse_timestamp(data["created_at"]) if "created_at" in data else None,
            "updated_at": parse_timestamp(data["updated_at"]) if "updated_at" in data else None
        }
        
        # Add optional fields if present
        for field_name in ["name", "description", "unit", "source"]:
            if field_name in data:
                kwargs[field_name] = data[field_name]
        
        if "frequency" in data:
            kwargs["frequency"] = Frequency(data["frequency"])
        
        return cls(**kwargs)


class TimeSeries:
    """
    Represents a time series with data points and metadata.
    
    This class provides comprehensive functionality for working with time series data,
    including validation, manipulation, serialization, and basic analytics.
    """
    
    def __init__(
        self,
        data_points: List[DataPoint] = None,
        metadata: TimeSeriesMetadata = None,
        auto_sort: bool = True
    ):
        """
        Initialize a time series.
        
        Args:
            data_points: List of data points
            metadata: Time series metadata
            auto_sort: Whether to automatically sort data points by timestamp
        """
        self._data_points = data_points or []
        self._metadata = metadata
        self._auto_sort = auto_sort
        
        if self._auto_sort and self._data_points:
            self._sort_data_points()
        
        self._validate()
    
    def _validate(self):
        """Validate the time series data."""
        if not isinstance(self._data_points, list):
            raise ValidationError("data_points must be a list")
        
        for i, point in enumerate(self._data_points):
            if not isinstance(point, DataPoint):
                raise ValidationError(f"data_points[{i}] must be a DataPoint instance")
    
    def _sort_data_points(self):
        """Sort data points by timestamp."""
        self._data_points.sort(key=lambda dp: dp.timestamp)
    
    @property
    def data_points(self) -> List[DataPoint]:
        """Get the list of data points."""
        return self._data_points.copy()
    
    @property
    def metadata(self) -> Optional[TimeSeriesMetadata]:
        """Get the time series metadata."""
        return self._metadata
    
    @property
    def length(self) -> int:
        """Get the number of data points."""
        return len(self._data_points)
    
    @property
    def is_empty(self) -> bool:
        """Check if the time series is empty."""
        return len(self._data_points) == 0
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Get the timestamp of the first data point."""
        return self._data_points[0].timestamp if self._data_points else None
    
    @property
    def end_time(self) -> Optional[datetime]:
        """Get the timestamp of the last data point."""
        return self._data_points[-1].timestamp if self._data_points else None
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the time series in seconds."""
        if len(self._data_points) < 2:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def values(self) -> List[float]:
        """Get list of values."""
        return [dp.value for dp in self._data_points]
    
    @property
    def timestamps(self) -> List[datetime]:
        """Get list of timestamps."""
        return [dp.timestamp for dp in self._data_points]
    
    def add_point(self, data_point: DataPoint):
        """
        Add a data point to the time series.
        
        Args:
            data_point: DataPoint to add
        """
        if not isinstance(data_point, DataPoint):
            raise ValidationError("data_point must be a DataPoint instance")
        
        self._data_points.append(data_point)
        
        if self._auto_sort:
            self._sort_data_points()
    
    def add_points(self, data_points: List[DataPoint]):
        """
        Add multiple data points to the time series.
        
        Args:
            data_points: List of DataPoints to add
        """
        for point in data_points:
            if not isinstance(point, DataPoint):
                raise ValidationError("All items must be DataPoint instances")
        
        self._data_points.extend(data_points)
        
        if self._auto_sort:
            self._sort_data_points()
    
    def get_point_at_index(self, index: int) -> DataPoint:
        """Get data point at specific index."""
        try:
            return self._data_points[index]
        except IndexError:
            raise ValidationError(f"Index {index} out of range for time series of length {len(self._data_points)}")
    
    def get_points_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
        inclusive: bool = True
    ) -> List[DataPoint]:
        """
        Get data points within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            inclusive: Whether to include boundary points
        
        Returns:
            List of data points in the specified range
        """
        if start_time >= end_time:
            raise ValidationError("start_time must be before end_time")
        
        result = []
        for point in self._data_points:
            if inclusive:
                if start_time <= point.timestamp <= end_time:
                    result.append(point)
            else:
                if start_time < point.timestamp < end_time:
                    result.append(point)
        
        return result
    
    def slice(self, start_index: int = None, end_index: int = None) -> "TimeSeries":
        """
        Create a new time series with a slice of data points.
        
        Args:
            start_index: Start index (inclusive)
            end_index: End index (exclusive)
        
        Returns:
            New TimeSeries with sliced data
        """
        sliced_points = self._data_points[start_index:end_index]
        return TimeSeries(
            data_points=sliced_points,
            metadata=self._metadata,
            auto_sort=False  # Already sorted
        )
    
    def filter_by_quality(self, min_quality: float = 0.0) -> "TimeSeries":
        """
        Filter data points by quality threshold.
        
        Args:
            min_quality: Minimum quality threshold (0.0 to 1.0)
        
        Returns:
            New TimeSeries with filtered data
        """
        if not 0.0 <= min_quality <= 1.0:
            raise ValidationError("min_quality must be between 0.0 and 1.0")
        
        filtered_points = [dp for dp in self._data_points if dp.quality >= min_quality]
        return TimeSeries(
            data_points=filtered_points,
            metadata=self._metadata,
            auto_sort=False  # Already sorted
        )
    
    def resample(
        self,
        frequency: Frequency,
        aggregation: AggregationMethod = AggregationMethod.MEAN
    ) -> "TimeSeries":
        """
        Resample the time series to a different frequency.
        
        Args:
            frequency: Target frequency
            aggregation: Aggregation method to use
        
        Returns:
            New resampled TimeSeries
        """
        if self.is_empty:
            return TimeSeries(metadata=self._metadata)
        
        # Convert to pandas for resampling
        df = self.to_dataframe()
        
        # Map frequency enum to pandas frequency string
        freq_map = {
            Frequency.SECOND: "1S",
            Frequency.MINUTE: "1min",
            Frequency.HOUR: "1H",
            Frequency.DAY: "1D",
            Frequency.WEEK: "1W",
            Frequency.MONTH: "1M",
            Frequency.QUARTER: "1Q",
            Frequency.YEAR: "1Y"
        }
        
        if frequency not in freq_map:
            raise ValidationError(f"Resampling not supported for frequency: {frequency}")
        
        pandas_freq = freq_map[frequency]
        
        # Resample based on aggregation method
        if aggregation == AggregationMethod.MEAN:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].mean()
        elif aggregation == AggregationMethod.MEDIAN:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].median()
        elif aggregation == AggregationMethod.SUM:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].sum()
        elif aggregation == AggregationMethod.MIN:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].min()
        elif aggregation == AggregationMethod.MAX:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].max()
        elif aggregation == AggregationMethod.COUNT:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].count()
        elif aggregation == AggregationMethod.STD:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].std()
        elif aggregation == AggregationMethod.FIRST:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].first()
        elif aggregation == AggregationMethod.LAST:
            resampled = df.resample(pandas_freq, on="timestamp")["value"].last()
        else:
            raise ValidationError(f"Unsupported aggregation method: {aggregation}")
        
        # Convert back to TimeSeries
        new_points = []
        for timestamp, value in resampled.dropna().items():
            new_points.append(DataPoint(timestamp=timestamp, value=value))
        
        return TimeSeries(
            data_points=new_points,
            metadata=self._metadata,
            auto_sort=False  # Already sorted
        )
    
    def basic_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics for the time series."""
        if self.is_empty:
            return {}
        
        values = np.array(self.values)
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "var": float(np.var(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "skewness": float(self._calculate_skewness(values)),
            "kurtosis": float(self._calculate_kurtosis(values))
        }
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of the values."""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of the values."""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 4) - 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert time series to dictionary."""
        result = {
            "data_points": [dp.to_dict() for dp in self._data_points],
            "length": self.length
        }
        
        if self._metadata:
            result["metadata"] = self._metadata.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeries":
        """Create time series from dictionary."""
        data_points = [DataPoint.from_dict(dp_data) for dp_data in data["data_points"]]
        
        metadata = None
        if "metadata" in data:
            metadata = TimeSeriesMetadata.from_dict(data["metadata"])
        
        return cls(data_points=data_points, metadata=metadata)
    
    def to_json(self, indent: int = None) -> str:
        """Convert time series to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TimeSeries":
        """Create time series from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def to_csv(self, include_metadata: bool = True) -> str:
        """Convert time series to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = ["timestamp", "value", "quality"]
        if include_metadata:
            headers.append("metadata")
        writer.writerow(headers)
        
        # Write data points
        for dp in self._data_points:
            row = [format_timestamp(dp.timestamp), dp.value, dp.quality]
            if include_metadata:
                row.append(json.dumps(dp.metadata) if dp.metadata else "")
            writer.writerow(row)
        
        return output.getvalue()
    
    @classmethod
    def from_csv(cls, csv_str: str, has_header: bool = True) -> "TimeSeries":
        """Create time series from CSV string."""
        input_stream = io.StringIO(csv_str)
        reader = csv.reader(input_stream)
        
        data_points = []
        
        if has_header:
            next(reader)  # Skip header row
        
        for row in reader:
            if len(row) < 2:
                continue
            
            timestamp = parse_timestamp(row[0])
            value = float(row[1])
            quality = float(row[2]) if len(row) > 2 and row[2] else 1.0
            
            metadata = None
            if len(row) > 3 and row[3]:
                try:
                    metadata = json.loads(row[3])
                except json.JSONDecodeError:
                    metadata = {"raw": row[3]}
            
            data_points.append(DataPoint(
                timestamp=timestamp,
                value=value,
                quality=quality,
                metadata=metadata
            ))
        
        return cls(data_points=data_points)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series to pandas DataFrame."""
        if self.is_empty:
            return pd.DataFrame(columns=["timestamp", "value", "quality"])
        
        data = {
            "timestamp": self.timestamps,
            "value": self.values,
            "quality": [dp.quality for dp in self._data_points]
        }
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, timestamp_col: str = None) -> "TimeSeries":
        """
        Create time series from pandas DataFrame.
        
        Args:
            df: DataFrame with timestamp and value columns
            timestamp_col: Name of timestamp column (if not index)
        
        Returns:
            New TimeSeries instance
        """
        df_copy = df.copy()
        
        # Handle timestamp column
        if timestamp_col:
            if timestamp_col not in df_copy.columns:
                raise ValidationError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            df_copy.set_index(timestamp_col, inplace=True)
        
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            raise ValidationError("DataFrame index must be a DatetimeIndex")
        
        # Determine value column
        value_col = "value"
        if "value" not in df_copy.columns:
            if len(df_copy.columns) == 1:
                value_col = df_copy.columns[0]
            else:
                raise ValidationError("DataFrame must have a 'value' column or single numeric column")
        
        # Create data points
        data_points = []
        for timestamp, row in df_copy.iterrows():
            value = row[value_col]
            quality = row.get("quality", 1.0)
            
            data_points.append(DataPoint(
                timestamp=timestamp.to_pydatetime(),
                value=float(value),
                quality=float(quality)
            ))
        
        return cls(data_points=data_points)
    
    def __len__(self) -> int:
        """Get the number of data points."""
        return len(self._data_points)
    
    def __iter__(self) -> Iterator[DataPoint]:
        """Iterate over data points."""
        return iter(self._data_points)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[DataPoint, "TimeSeries"]:
        """Get data point(s) by index or slice."""
        if isinstance(key, int):
            return self._data_points[key]
        elif isinstance(key, slice):
            return self.slice(key.start, key.stop)
        else:
            raise TypeError("Key must be int or slice")
    
    def __str__(self) -> str:
        series_id = self._metadata.series_id if self._metadata else "unknown"
        return f"TimeSeries(id={series_id}, length={self.length})"
    
    def __repr__(self) -> str:
        return f"TimeSeries(data_points={len(self._data_points)}, metadata={self._metadata!r})"


# Convenience functions for creating time series
def create_time_series(
    values: List[float],
    timestamps: List[datetime] = None,
    series_id: str = "default",
    frequency: Frequency = None,
    **metadata_kwargs
) -> TimeSeries:
    """
    Create a time series from lists of values and timestamps.
    
    Args:
        values: List of numeric values
        timestamps: List of timestamps (auto-generated if None)
        series_id: Series identifier
        frequency: Time series frequency
        **metadata_kwargs: Additional metadata fields
    
    Returns:
        New TimeSeries instance
    """
    if timestamps is None:
        # Generate timestamps with 1-second intervals
        start_time = current_timestamp()
        timestamps = [start_time + pd.Timedelta(seconds=i) for i in range(len(values))]
    
    if len(values) != len(timestamps):
        raise ValidationError("Values and timestamps lists must have the same length")
    
    # Create data points
    data_points = [
        DataPoint(timestamp=ts, value=val)
        for ts, val in zip(timestamps, values)
    ]
    
    # Create metadata
    metadata = TimeSeriesMetadata(
        series_id=series_id,
        frequency=frequency,
        **metadata_kwargs
    )
    
    return TimeSeries(data_points=data_points, metadata=metadata)


def create_empty_time_series(series_id: str = "default", **metadata_kwargs) -> TimeSeries:
    """Create an empty time series with metadata."""
    metadata = TimeSeriesMetadata(series_id=series_id, **metadata_kwargs)
    return TimeSeries(metadata=metadata)