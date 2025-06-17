package timegan

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/stat"
)

// DataScaler handles data normalization and denormalization
type DataScaler struct {
	scalerType string
	min        float64
	max        float64
	mean       float64
	stddev     float64
	fitted     bool
}

// NewDataScaler creates a new data scaler
func NewDataScaler(scalerType string) *DataScaler {
	return &DataScaler{
		scalerType: scalerType,
		fitted:     false,
	}
}

// Fit calculates scaling parameters from training data
func (ds *DataScaler) Fit(data []float64) error {
	if len(data) == 0 {
		return fmt.Errorf("cannot fit scaler on empty data")
	}
	
	switch ds.scalerType {
	case "minmax":
		ds.min = data[0]
		ds.max = data[0]
		for _, val := range data {
			if val < ds.min {
				ds.min = val
			}
			if val > ds.max {
				ds.max = val
			}
		}
		// Add small epsilon to avoid division by zero
		if ds.max == ds.min {
			ds.max = ds.min + 1e-8
		}
		
	case "zscore":
		ds.mean = stat.Mean(data, nil)
		variance := stat.Variance(data, nil)
		ds.stddev = math.Sqrt(variance)
		// Add small epsilon to avoid division by zero
		if ds.stddev == 0 {
			ds.stddev = 1e-8
		}
		
	default:
		return fmt.Errorf("unknown scaler type: %s", ds.scalerType)
	}
	
	ds.fitted = true
	return nil
}

// Transform scales the input data
func (ds *DataScaler) Transform(data []float64) []float64 {
	if !ds.fitted {
		return data // Return original data if not fitted
	}
	
	result := make([]float64, len(data))
	
	switch ds.scalerType {
	case "minmax":
		for i, val := range data {
			result[i] = (val - ds.min) / (ds.max - ds.min)
		}
		
	case "zscore":
		for i, val := range data {
			result[i] = (val - ds.mean) / ds.stddev
		}
		
	default:
		copy(result, data)
	}
	
	return result
}

// InverseTransform reverses the scaling transformation
func (ds *DataScaler) InverseTransform(data []float64) []float64 {
	if !ds.fitted {
		return data // Return original data if not fitted
	}
	
	result := make([]float64, len(data))
	
	switch ds.scalerType {
	case "minmax":
		for i, val := range data {
			result[i] = val*(ds.max-ds.min) + ds.min
		}
		
	case "zscore":
		for i, val := range data {
			result[i] = val*ds.stddev + ds.mean
		}
		
	default:
		copy(result, data)
	}
	
	return result
}

// FitTransform fits the scaler and transforms the data in one step
func (ds *DataScaler) FitTransform(data []float64) ([]float64, error) {
	if err := ds.Fit(data); err != nil {
		return nil, err
	}
	return ds.Transform(data), nil
}

// IsFitted returns whether the scaler has been fitted
func (ds *DataScaler) IsFitted() bool {
	return ds.fitted
}

// GetParameters returns the scaling parameters
func (ds *DataScaler) GetParameters() map[string]float64 {
	params := make(map[string]float64)
	
	switch ds.scalerType {
	case "minmax":
		params["min"] = ds.min
		params["max"] = ds.max
		
	case "zscore":
		params["mean"] = ds.mean
		params["stddev"] = ds.stddev
	}
	
	return params
}

// SetParameters sets the scaling parameters (useful for loading trained scalers)
func (ds *DataScaler) SetParameters(params map[string]float64) error {
	switch ds.scalerType {
	case "minmax":
		if min, ok := params["min"]; ok {
			ds.min = min
		} else {
			return fmt.Errorf("missing 'min' parameter for minmax scaler")
		}
		if max, ok := params["max"]; ok {
			ds.max = max
		} else {
			return fmt.Errorf("missing 'max' parameter for minmax scaler")
		}
		
	case "zscore":
		if mean, ok := params["mean"]; ok {
			ds.mean = mean
		} else {
			return fmt.Errorf("missing 'mean' parameter for zscore scaler")
		}
		if stddev, ok := params["stddev"]; ok {
			ds.stddev = stddev
		} else {
			return fmt.Errorf("missing 'stddev' parameter for zscore scaler")
		}
		
	default:
		return fmt.Errorf("unknown scaler type: %s", ds.scalerType)
	}
	
	ds.fitted = true
	return nil
}

// GetScalerType returns the scaler type
func (ds *DataScaler) GetScalerType() string {
	return ds.scalerType
}

// Reset resets the scaler to unfitted state
func (ds *DataScaler) Reset() {
	ds.fitted = false
	ds.min = 0
	ds.max = 0
	ds.mean = 0
	ds.stddev = 0
}
