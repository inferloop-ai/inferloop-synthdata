"""
Data profiling capabilities for understanding dataset characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings

from scipy import stats
from scipy.stats import normaltest, kstest, chi2_contingency


@dataclass
class ColumnProfile:
    """Profile information for a single column"""
    name: str
    dtype: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Statistical properties
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Distribution info
    distribution_type: Optional[str] = None
    distribution_params: Optional[Dict[str, float]] = None
    normality_test: Optional[Dict[str, float]] = None
    
    # Categorical properties
    top_values: Optional[Dict[str, int]] = None
    value_counts: Optional[Dict[str, int]] = None
    
    # Data quality
    outliers_count: int = 0
    outliers_percentage: float = 0.0
    suspicious_values: List[Any] = field(default_factory=list)
    
    # Patterns
    patterns: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class DatasetProfile:
    """Complete profile of a dataset"""
    name: str
    shape: Tuple[int, int]
    memory_usage_mb: float
    column_profiles: Dict[str, ColumnProfile]
    correlations: Optional[pd.DataFrame] = None
    duplicates_count: int = 0
    duplicates_percentage: float = 0.0
    profile_timestamp: datetime = field(default_factory=datetime.now)
    
    # Dataset-level statistics
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    
    # Relationships
    potential_keys: List[str] = field(default_factory=list)
    column_relationships: Optional[Dict[str, List[str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'name': self.name,
            'shape': self.shape,
            'memory_usage_mb': self.memory_usage_mb,
            'column_profiles': {k: v.to_dict() for k, v in self.column_profiles.items()},
            'duplicates_count': self.duplicates_count,
            'duplicates_percentage': self.duplicates_percentage,
            'profile_timestamp': self.profile_timestamp.isoformat(),
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_columns': self.datetime_columns,
            'constant_columns': self.constant_columns,
            'high_cardinality_columns': self.high_cardinality_columns,
            'potential_keys': self.potential_keys
        }
        
        if self.correlations is not None:
            result['correlations'] = self.correlations.to_dict()
        
        if self.column_relationships:
            result['column_relationships'] = self.column_relationships
        
        return result


class DataProfiler:
    """Comprehensive data profiling tool"""
    
    def __init__(self, 
                 detect_patterns: bool = True,
                 detect_distributions: bool = True,
                 outlier_method: str = 'iqr',
                 cardinality_threshold: float = 0.95):
        self.detect_patterns = detect_patterns
        self.detect_distributions = detect_distributions
        self.outlier_method = outlier_method
        self.cardinality_threshold = cardinality_threshold
    
    def profile_dataset(self, df: pd.DataFrame, name: str = "Dataset") -> DatasetProfile:
        """Generate comprehensive profile of dataset"""
        # Basic info
        shape = df.shape
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Profile each column
        column_profiles = {}
        for col in df.columns:
            column_profiles[col] = self.profile_column(df[col])
        
        # Classify columns
        numerical_cols = []
        categorical_cols = []
        datetime_cols = []
        constant_cols = []
        high_cardinality_cols = []
        
        for col, profile in column_profiles.items():
            if profile.unique_count == 1:
                constant_cols.append(col)
            elif profile.dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            elif profile.dtype == 'datetime64[ns]':
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
                if profile.unique_percentage > self.cardinality_threshold:
                    high_cardinality_cols.append(col)
        
        # Calculate correlations for numerical columns
        correlations = None
        if len(numerical_cols) > 1:
            correlations = df[numerical_cols].corr()
        
        # Find duplicates
        duplicates_count = df.duplicated().sum()
        duplicates_percentage = (duplicates_count / len(df)) * 100
        
        # Find potential keys
        potential_keys = self._find_potential_keys(df, column_profiles)
        
        # Find column relationships
        column_relationships = self._find_column_relationships(df, column_profiles)
        
        return DatasetProfile(
            name=name,
            shape=shape,
            memory_usage_mb=memory_usage_mb,
            column_profiles=column_profiles,
            correlations=correlations,
            duplicates_count=duplicates_count,
            duplicates_percentage=duplicates_percentage,
            numerical_columns=numerical_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            constant_columns=constant_cols,
            high_cardinality_columns=high_cardinality_cols,
            potential_keys=potential_keys,
            column_relationships=column_relationships
        )
    
    def profile_column(self, series: pd.Series) -> ColumnProfile:
        """Profile a single column"""
        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            count=len(series),
            null_count=series.isnull().sum(),
            null_percentage=(series.isnull().sum() / len(series)) * 100,
            unique_count=series.nunique(),
            unique_percentage=(series.nunique() / len(series)) * 100
        )
        
        # Handle different data types
        if pd.api.types.is_numeric_dtype(series):
            self._profile_numeric_column(series, profile)
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            self._profile_categorical_column(series, profile)
        elif pd.api.types.is_datetime64_any_dtype(series):
            self._profile_datetime_column(series, profile)
        
        # Detect patterns if enabled
        if self.detect_patterns:
            profile.patterns = self._detect_patterns(series)
        
        return profile
    
    def _profile_numeric_column(self, series: pd.Series, profile: ColumnProfile):
        """Profile numeric column"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return
        
        # Basic statistics
        profile.mean = float(clean_series.mean())
        profile.median = float(clean_series.median())
        profile.std = float(clean_series.std())
        profile.min = float(clean_series.min())
        profile.max = float(clean_series.max())
        profile.q1 = float(clean_series.quantile(0.25))
        profile.q3 = float(clean_series.quantile(0.75))
        profile.iqr = profile.q3 - profile.q1
        
        # Higher moments
        profile.skewness = float(clean_series.skew())
        profile.kurtosis = float(clean_series.kurtosis())
        
        # Mode (if meaningful)
        mode_result = clean_series.mode()
        if len(mode_result) > 0:
            profile.mode = float(mode_result.iloc[0])
        
        # Detect outliers
        outliers = self._detect_outliers(clean_series)
        profile.outliers_count = len(outliers)
        profile.outliers_percentage = (len(outliers) / len(clean_series)) * 100
        
        # Distribution detection
        if self.detect_distributions and len(clean_series) > 30:
            profile.distribution_type, profile.distribution_params = self._detect_distribution(clean_series)
            profile.normality_test = self._test_normality(clean_series)
    
    def _profile_categorical_column(self, series: pd.Series, profile: ColumnProfile):
        """Profile categorical column"""
        # Value counts
        value_counts = series.value_counts()
        profile.value_counts = value_counts.to_dict()
        
        # Top values
        top_n = min(10, len(value_counts))
        profile.top_values = value_counts.head(top_n).to_dict()
        
        # Mode
        if len(value_counts) > 0:
            profile.mode = value_counts.index[0]
    
    def _profile_datetime_column(self, series: pd.Series, profile: ColumnProfile):
        """Profile datetime column"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return
        
        # Convert to numeric for statistics
        numeric_series = pd.to_numeric(clean_series)
        
        profile.min = clean_series.min()
        profile.max = clean_series.max()
        
        # Additional datetime-specific patterns
        profile.patterns = {
            'date_range': f"{profile.min} to {profile.max}",
            'span_days': (profile.max - profile.min).days
        }
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers in numeric series"""
        if self.outlier_method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return series[(series < lower_bound) | (series > upper_bound)]
        
        elif self.outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            return series[z_scores > 3]
        
        return pd.Series()
    
    def _detect_distribution(self, series: pd.Series) -> Tuple[str, Dict[str, float]]:
        """Detect statistical distribution of numeric series"""
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'exponential': stats.expon,
            'uniform': stats.uniform,
            'beta': stats.beta,
            'gamma': stats.gamma
        }
        
        best_dist = None
        best_params = None
        best_stat = np.inf
        
        for dist_name, dist_func in distributions.items():
            try:
                # Fit distribution
                params = dist_func.fit(series)
                
                # Kolmogorov-Smirnov test
                stat, _ = kstest(series, lambda x: dist_func.cdf(x, *params))
                
                if stat < best_stat:
                    best_stat = stat
                    best_dist = dist_name
                    best_params = params
            except:
                continue
        
        if best_dist and best_params:
            return best_dist, {'params': list(best_params), 'ks_statistic': best_stat}
        
        return 'unknown', {}
    
    def _test_normality(self, series: pd.Series) -> Dict[str, float]:
        """Test for normality"""
        result = {}
        
        # Shapiro-Wilk test (for smaller samples)
        if len(series) <= 5000:
            stat, p_value = stats.shapiro(series)
            result['shapiro_wilk'] = {'statistic': stat, 'p_value': p_value}
        
        # D'Agostino's K^2 test
        stat, p_value = normaltest(series)
        result['dagostino'] = {'statistic': stat, 'p_value': p_value}
        
        return result
    
    def _detect_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Detect patterns in column data"""
        patterns = {}
        
        if pd.api.types.is_object_dtype(series):
            # Check for email pattern
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if series.dropna().str.match(email_pattern).any():
                patterns['possible_email'] = True
            
            # Check for phone pattern
            phone_pattern = r'^\+?1?\d{9,15}$'
            if series.dropna().str.match(phone_pattern).any():
                patterns['possible_phone'] = True
            
            # Check for URL pattern
            url_pattern = r'^https?://[^\s]+$'
            if series.dropna().str.match(url_pattern).any():
                patterns['possible_url'] = True
            
            # Check string lengths
            str_lengths = series.dropna().str.len()
            if len(str_lengths) > 0:
                patterns['avg_length'] = str_lengths.mean()
                patterns['min_length'] = str_lengths.min()
                patterns['max_length'] = str_lengths.max()
        
        return patterns
    
    def _find_potential_keys(self, df: pd.DataFrame, 
                            column_profiles: Dict[str, ColumnProfile]) -> List[str]:
        """Find columns that could be primary keys"""
        potential_keys = []
        
        for col, profile in column_profiles.items():
            # Check if column has unique values and no nulls
            if (profile.unique_count == len(df) and 
                profile.null_count == 0):
                potential_keys.append(col)
        
        return potential_keys
    
    def _find_column_relationships(self, df: pd.DataFrame,
                                 column_profiles: Dict[str, ColumnProfile]) -> Dict[str, List[str]]:
        """Find relationships between columns"""
        relationships = {}
        
        # Find columns with similar cardinality
        for col1, profile1 in column_profiles.items():
            similar_columns = []
            
            for col2, profile2 in column_profiles.items():
                if col1 != col2:
                    # Similar unique count might indicate relationship
                    if abs(profile1.unique_count - profile2.unique_count) < 5:
                        similar_columns.append(col2)
            
            if similar_columns:
                relationships[col1] = similar_columns
        
        return relationships
    
    def compare_profiles(self, profile1: DatasetProfile, 
                        profile2: DatasetProfile) -> Dict[str, Any]:
        """Compare two dataset profiles"""
        comparison = {
            'shape_match': profile1.shape == profile2.shape,
            'columns_match': set(profile1.column_profiles.keys()) == set(profile2.column_profiles.keys()),
            'common_columns': list(set(profile1.column_profiles.keys()) & set(profile2.column_profiles.keys())),
            'profile1_only': list(set(profile1.column_profiles.keys()) - set(profile2.column_profiles.keys())),
            'profile2_only': list(set(profile2.column_profiles.keys()) - set(profile1.column_profiles.keys())),
            'column_comparisons': {}
        }
        
        # Compare common columns
        for col in comparison['common_columns']:
            col_profile1 = profile1.column_profiles[col]
            col_profile2 = profile2.column_profiles[col]
            
            comparison['column_comparisons'][col] = {
                'dtype_match': col_profile1.dtype == col_profile2.dtype,
                'null_diff': abs(col_profile1.null_percentage - col_profile2.null_percentage),
                'unique_diff': abs(col_profile1.unique_percentage - col_profile2.unique_percentage)
            }
            
            # Compare statistics for numeric columns
            if col_profile1.mean is not None and col_profile2.mean is not None:
                comparison['column_comparisons'][col]['mean_diff'] = abs(col_profile1.mean - col_profile2.mean)
                comparison['column_comparisons'][col]['std_diff'] = abs(col_profile1.std - col_profile2.std)
        
        return comparison
    
    def generate_report(self, profile: DatasetProfile) -> str:
        """Generate human-readable profiling report"""
        report = []
        report.append(f"# Data Profile Report: {profile.name}")
        report.append(f"Generated at: {profile.profile_timestamp}")
        report.append("")
        
        # Overview
        report.append("## Dataset Overview")
        report.append(f"- Shape: {profile.shape[0]:,} rows × {profile.shape[1]} columns")
        report.append(f"- Memory Usage: {profile.memory_usage_mb:.2f} MB")
        report.append(f"- Duplicates: {profile.duplicates_count:,} ({profile.duplicates_percentage:.2f}%)")
        report.append("")
        
        # Column types
        report.append("## Column Types")
        report.append(f"- Numerical: {len(profile.numerical_columns)} columns")
        report.append(f"- Categorical: {len(profile.categorical_columns)} columns")
        report.append(f"- Datetime: {len(profile.datetime_columns)} columns")
        report.append(f"- Constant: {len(profile.constant_columns)} columns")
        report.append(f"- High Cardinality: {len(profile.high_cardinality_columns)} columns")
        report.append("")
        
        # Potential keys
        if profile.potential_keys:
            report.append("## Potential Primary Keys")
            for key in profile.potential_keys:
                report.append(f"- {key}")
            report.append("")
        
        # Column details
        report.append("## Column Details")
        for col_name, col_profile in profile.column_profiles.items():
            report.append(f"\n### {col_name}")
            report.append(f"- Type: {col_profile.dtype}")
            report.append(f"- Missing: {col_profile.null_count:,} ({col_profile.null_percentage:.2f}%)")
            report.append(f"- Unique: {col_profile.unique_count:,} ({col_profile.unique_percentage:.2f}%)")
            
            if col_profile.mean is not None:
                report.append(f"- Mean: {col_profile.mean:.2f}")
                report.append(f"- Std: {col_profile.std:.2f}")
                report.append(f"- Min: {col_profile.min:.2f}")
                report.append(f"- Max: {col_profile.max:.2f}")
                
                if col_profile.outliers_count > 0:
                    report.append(f"- Outliers: {col_profile.outliers_count} ({col_profile.outliers_percentage:.2f}%)")
                
                if col_profile.distribution_type:
                    report.append(f"- Distribution: {col_profile.distribution_type}")
            
            if col_profile.top_values:
                report.append("- Top Values:")
                for value, count in list(col_profile.top_values.items())[:5]:
                    report.append(f"  - {value}: {count:,}")
        
        return "\n".join(report)
    
    def export_profile(self, profile: DatasetProfile, output_path: str):
        """Export profile to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2, default=str)