package math

import (
	"errors"
	"math"
	"sort"
)

// Point represents a 2D point for interpolation
type Point struct {
	X, Y float64
}

// InterpolationMethod defines the type of interpolation
type InterpolationMethod int

const (
	Linear InterpolationMethod = iota
	Cubic
	Spline
	Polynomial
	Nearest
)

// Interpolator interface for different interpolation methods
type Interpolator interface {
	Interpolate(x float64) (float64, error)
	InterpolateRange(xValues []float64) ([]float64, error)
}

// LinearInterpolator implements linear interpolation
type LinearInterpolator struct {
	points []Point
}

// NewLinearInterpolator creates a new linear interpolator
func NewLinearInterpolator(points []Point) *LinearInterpolator {
	// Sort points by x value
	sorted := make([]Point, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].X < sorted[j].X
	})
	
	return &LinearInterpolator{points: sorted}
}

// Interpolate performs linear interpolation at point x
func (li *LinearInterpolator) Interpolate(x float64) (float64, error) {
	if len(li.points) < 2 {
		return 0, errors.New("need at least 2 points for linear interpolation")
	}
	
	// Find the two points that bracket x
	for i := 0; i < len(li.points)-1; i++ {
		if x >= li.points[i].X && x <= li.points[i+1].X {
			x1, y1 := li.points[i].X, li.points[i].Y
			x2, y2 := li.points[i+1].X, li.points[i+1].Y
			
			// Linear interpolation formula
			if x2 == x1 {
				return y1, nil
			}
			
			return y1 + (y2-y1)*(x-x1)/(x2-x1), nil
		}
	}
	
	// Extrapolation
	if x < li.points[0].X {
		// Extrapolate using first two points
		x1, y1 := li.points[0].X, li.points[0].Y
		x2, y2 := li.points[1].X, li.points[1].Y
		slope := (y2 - y1) / (x2 - x1)
		return y1 + slope*(x-x1), nil
	}
	
	// Extrapolate using last two points
	n := len(li.points)
	x1, y1 := li.points[n-2].X, li.points[n-2].Y
	x2, y2 := li.points[n-1].X, li.points[n-1].Y
	slope := (y2 - y1) / (x2 - x1)
	return y2 + slope*(x-x2), nil
}

// InterpolateRange interpolates multiple x values
func (li *LinearInterpolator) InterpolateRange(xValues []float64) ([]float64, error) {
	results := make([]float64, len(xValues))
	for i, x := range xValues {
		y, err := li.Interpolate(x)
		if err != nil {
			return nil, err
		}
		results[i] = y
	}
	return results, nil
}

// CubicInterpolator implements cubic spline interpolation
type CubicInterpolator struct {
	points []Point
	a, b, c, d []float64 // Spline coefficients
}

// NewCubicInterpolator creates a new cubic spline interpolator
func NewCubicInterpolator(points []Point) (*CubicInterpolator, error) {
	if len(points) < 3 {
		return nil, errors.New("need at least 3 points for cubic interpolation")
	}
	
	// Sort points by x value
	sorted := make([]Point, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].X < sorted[j].X
	})
	
	ci := &CubicInterpolator{points: sorted}
	err := ci.computeCoefficients()
	if err != nil {
		return nil, err
	}
	
	return ci, nil
}

// computeCoefficients calculates the cubic spline coefficients
func (ci *CubicInterpolator) computeCoefficients() error {
	n := len(ci.points)
	h := make([]float64, n-1)
	alpha := make([]float64, n-1)
	
	// Calculate h and alpha
	for i := 0; i < n-1; i++ {
		h[i] = ci.points[i+1].X - ci.points[i].X
		if h[i] == 0 {
			return errors.New("duplicate x values not allowed")
		}
	}
	
	for i := 1; i < n-1; i++ {
		alpha[i] = (3/h[i])*(ci.points[i+1].Y-ci.points[i].Y) - (3/h[i-1])*(ci.points[i].Y-ci.points[i-1].Y)
	}
	
	// Solve tridiagonal system
	l := make([]float64, n)
	mu := make([]float64, n)
	z := make([]float64, n)
	c := make([]float64, n)
	
	l[0] = 1
	mu[0] = 0
	z[0] = 0
	
	for i := 1; i < n-1; i++ {
		l[i] = 2*(ci.points[i+1].X-ci.points[i-1].X) - h[i-1]*mu[i-1]
		mu[i] = h[i] / l[i]
		z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
	}
	
	l[n-1] = 1
	z[n-1] = 0
	c[n-1] = 0
	
	// Back substitution
	for j := n - 2; j >= 0; j-- {
		c[j] = z[j] - mu[j]*c[j+1]
	}
	
	// Calculate remaining coefficients
	ci.a = make([]float64, n-1)
	ci.b = make([]float64, n-1)
	ci.c = make([]float64, n-1)
	ci.d = make([]float64, n-1)
	
	for j := 0; j < n-1; j++ {
		ci.a[j] = ci.points[j].Y
		ci.b[j] = (ci.points[j+1].Y-ci.points[j].Y)/h[j] - h[j]*(c[j+1]+2*c[j])/3
		ci.c[j] = c[j]
		ci.d[j] = (c[j+1] - c[j]) / (3 * h[j])
	}
	
	return nil
}

// Interpolate performs cubic spline interpolation at point x
func (ci *CubicInterpolator) Interpolate(x float64) (float64, error) {
	// Find the appropriate interval
	for i := 0; i < len(ci.points)-1; i++ {
		if x >= ci.points[i].X && x <= ci.points[i+1].X {
			dx := x - ci.points[i].X
			return ci.a[i] + ci.b[i]*dx + ci.c[i]*dx*dx + ci.d[i]*dx*dx*dx, nil
		}
	}
	
	// Extrapolation
	if x < ci.points[0].X {
		dx := x - ci.points[0].X
		return ci.a[0] + ci.b[0]*dx + ci.c[0]*dx*dx + ci.d[0]*dx*dx*dx, nil
	}
	
	// Use last interval for extrapolation
	i := len(ci.points) - 2
	dx := x - ci.points[i].X
	return ci.a[i] + ci.b[i]*dx + ci.c[i]*dx*dx + ci.d[i]*dx*dx*dx, nil
}

// InterpolateRange interpolates multiple x values
func (ci *CubicInterpolator) InterpolateRange(xValues []float64) ([]float64, error) {
	results := make([]float64, len(xValues))
	for i, x := range xValues {
		y, err := ci.Interpolate(x)
		if err != nil {
			return nil, err
		}
		results[i] = y
	}
	return results, nil
}

// NearestInterpolator implements nearest neighbor interpolation
type NearestInterpolator struct {
	points []Point
}

// NewNearestInterpolator creates a new nearest neighbor interpolator
func NewNearestInterpolator(points []Point) *NearestInterpolator {
	sorted := make([]Point, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].X < sorted[j].X
	})
	
	return &NearestInterpolator{points: sorted}
}

// Interpolate performs nearest neighbor interpolation
func (ni *NearestInterpolator) Interpolate(x float64) (float64, error) {
	if len(ni.points) == 0 {
		return 0, errors.New("no points available for interpolation")
	}
	
	minDistance := math.Inf(1)
	nearestY := ni.points[0].Y
	
	for _, point := range ni.points {
		distance := math.Abs(x - point.X)
		if distance < minDistance {
			minDistance = distance
			nearestY = point.Y
		}
	}
	
	return nearestY, nil
}

// InterpolateRange interpolates multiple x values
func (ni *NearestInterpolator) InterpolateRange(xValues []float64) ([]float64, error) {
	results := make([]float64, len(xValues))
	for i, x := range xValues {
		y, err := ni.Interpolate(x)
		if err != nil {
			return nil, err
		}
		results[i] = y
	}
	return results, nil
}

// PolynomialInterpolator implements polynomial interpolation using Lagrange method
type PolynomialInterpolator struct {
	points []Point
}

// NewPolynomialInterpolator creates a new polynomial interpolator
func NewPolynomialInterpolator(points []Point) *PolynomialInterpolator {
	return &PolynomialInterpolator{points: points}
}

// Interpolate performs Lagrange polynomial interpolation
func (pi *PolynomialInterpolator) Interpolate(x float64) (float64, error) {
	if len(pi.points) == 0 {
		return 0, errors.New("no points available for interpolation")
	}
	
	result := 0.0
	n := len(pi.points)
	
	for i := 0; i < n; i++ {
		term := pi.points[i].Y
		
		for j := 0; j < n; j++ {
			if i != j {
				term *= (x - pi.points[j].X) / (pi.points[i].X - pi.points[j].X)
			}
		}
		
		result += term
	}
	
	return result, nil
}

// InterpolateRange interpolates multiple x values
func (pi *PolynomialInterpolator) InterpolateRange(xValues []float64) ([]float64, error) {
	results := make([]float64, len(xValues))
	for i, x := range xValues {
		y, err := pi.Interpolate(x)
		if err != nil {
			return nil, err
		}
		results[i] = y
	}
	return results, nil
}

// Interpolate1D is a convenience function for 1D interpolation
func Interpolate1D(x, y, xNew []float64, method InterpolationMethod) ([]float64, error) {
	if len(x) != len(y) {
		return nil, errors.New("x and y must have the same length")
	}
	
	// Create points
	points := make([]Point, len(x))
	for i := range x {
		points[i] = Point{X: x[i], Y: y[i]}
	}
	
	var interpolator Interpolator
	var err error
	
	switch method {
	case Linear:
		interpolator = NewLinearInterpolator(points)
	case Cubic:
		interpolator, err = NewCubicInterpolator(points)
		if err != nil {
			return nil, err
		}
	case Nearest:
		interpolator = NewNearestInterpolator(points)
	case Polynomial:
		interpolator = NewPolynomialInterpolator(points)
	default:
		return nil, errors.New("unsupported interpolation method")
	}
	
	return interpolator.InterpolateRange(xNew)
}

// ResampleTimeSeries resamples a time series to a new time grid
func ResampleTimeSeries(times, values []float64, newTimes []float64, method InterpolationMethod) ([]float64, error) {
	return Interpolate1D(times, values, newTimes, method)
}

// FillMissingValues fills missing values in a time series using interpolation
func FillMissingValues(values []float64, method InterpolationMethod) []float64 {
	result := make([]float64, len(values))
	copy(result, values)
	
	// Find valid points
	var validPoints []Point
	for i, val := range values {
		if !math.IsNaN(val) {
			validPoints = append(validPoints, Point{X: float64(i), Y: val})
		}
	}
	
	if len(validPoints) < 2 {
		return result
	}
	
	// Create interpolator
	var interpolator Interpolator
	switch method {
	case Linear:
		interpolator = NewLinearInterpolator(validPoints)
	case Cubic:
		if len(validPoints) >= 3 {
			var err error
			interpolator, err = NewCubicInterpolator(validPoints)
			if err != nil {
				interpolator = NewLinearInterpolator(validPoints)
			}
		} else {
			interpolator = NewLinearInterpolator(validPoints)
		}
	case Nearest:
		interpolator = NewNearestInterpolator(validPoints)
	default:
		interpolator = NewLinearInterpolator(validPoints)
	}
	
	// Fill missing values
	for i, val := range values {
		if math.IsNaN(val) {
			interpolated, err := interpolator.Interpolate(float64(i))
			if err == nil {
				result[i] = interpolated
			}
		}
	}
	
	return result
}

// UpSample increases the sampling rate of a time series
func UpSample(values []float64, factor int, method InterpolationMethod) ([]float64, error) {
	if factor <= 1 {
		return values, nil
	}
	
	originalTimes := make([]float64, len(values))
	for i := range originalTimes {
		originalTimes[i] = float64(i)
	}
	
	newLength := (len(values)-1)*factor + 1
	newTimes := make([]float64, newLength)
	for i := range newTimes {
		newTimes[i] = float64(i) / float64(factor)
	}
	
	return Interpolate1D(originalTimes, values, newTimes, method)
}

// DownSample decreases the sampling rate of a time series
func DownSample(values []float64, factor int) []float64 {
	if factor <= 1 {
		return values
	}
	
	newLength := (len(values) + factor - 1) / factor
	result := make([]float64, newLength)
	
	for i := 0; i < newLength; i++ {
		index := i * factor
		if index < len(values) {
			result[i] = values[index]
		}
	}
	
	return result
}