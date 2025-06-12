package arima

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// TrendAnalyzer analyzes and detects trends in time series data
type TrendAnalyzer struct {
	significanceLevel float64
}

// NewTrendAnalyzer creates a new trend analyzer
func NewTrendAnalyzer(significanceLevel float64) *TrendAnalyzer {
	if significanceLevel <= 0 || significanceLevel >= 1 {
		significanceLevel = 0.05 // Default 5% significance level
	}

	return &TrendAnalyzer{
		significanceLevel: significanceLevel,
	}
}

// TrendAnalysisResult contains the results of trend analysis
type TrendAnalysisResult struct {
	TrendType          string               `json:"trend_type"`          // "none", "linear", "exponential", "polynomial", "seasonal"
	TrendStrength      float64              `json:"trend_strength"`      // 0-1, strength of trend component
	TrendDirection     string               `json:"trend_direction"`     // "increasing", "decreasing", "stable"
	LinearTrend        *LinearTrendResult   `json:"linear_trend,omitempty"`
	PolynomialTrend    *PolynomialTrendResult `json:"polynomial_trend,omitempty"`
	ExponentialTrend   *ExponentialTrendResult `json:"exponential_trend,omitempty"`
	BreakpointAnalysis *BreakpointResult    `json:"breakpoint_analysis,omitempty"`
	MannKendallTest    *MannKendallResult   `json:"mann_kendall_test,omitempty"`
	ChangePoints       []int                `json:"change_points,omitempty"`
	TrendComponents    []float64            `json:"trend_components"`
	DetrendedSeries    []float64            `json:"detrended_series"`
}

// LinearTrendResult contains linear trend analysis results
type LinearTrendResult struct {
	Slope              float64 `json:"slope"`
	Intercept          float64 `json:"intercept"`
	RSquared           float64 `json:"r_squared"`
	PValue             float64 `json:"p_value"`
	StandardError      float64 `json:"standard_error"`
	ConfidenceInterval [2]float64 `json:"confidence_interval"`
	IsSignificant      bool    `json:"is_significant"`
}

// PolynomialTrendResult contains polynomial trend analysis results
type PolynomialTrendResult struct {
	Degree             int       `json:"degree"`
	Coefficients       []float64 `json:"coefficients"`
	RSquared           float64   `json:"r_squared"`
	AdjustedRSquared   float64   `json:"adjusted_r_squared"`
	AIC                float64   `json:"aic"`
	BIC                float64   `json:"bic"`
	IsSignificant      bool      `json:"is_significant"`
}

// ExponentialTrendResult contains exponential trend analysis results
type ExponentialTrendResult struct {
	GrowthRate         float64 `json:"growth_rate"`
	InitialValue       float64 `json:"initial_value"`
	RSquared           float64 `json:"r_squared"`
	PValue             float64 `json:"p_value"`
	DoubleTime         float64 `json:"double_time,omitempty"` // Time to double (if growing)
	HalfLife           float64 `json:"half_life,omitempty"`   // Half-life (if decaying)
	IsSignificant      bool    `json:"is_significant"`
}

// BreakpointResult contains structural break analysis results
type BreakpointResult struct {
	BreakpointIndices  []int     `json:"breakpoint_indices"`
	BreakpointDates    []string  `json:"breakpoint_dates,omitempty"`
	PreBreakTrend      float64   `json:"pre_break_trend"`
	PostBreakTrend     float64   `json:"post_break_trend"`
	TrendChangeSignificance float64 `json:"trend_change_significance"`
	StructuralBreakTest *ChowtestResult `json:"structural_break_test,omitempty"`
}

// ChowtestResult contains Chow test results for structural breaks
type ChowtestResult struct {
	FStatistic        float64 `json:"f_statistic"`
	PValue            float64 `json:"p_value"`
	DegreesOfFreedom1 int     `json:"degrees_of_freedom_1"`
	DegreesOfFreedom2 int     `json:"degrees_of_freedom_2"`
	IsSignificant     bool    `json:"is_significant"`
}

// MannKendallResult contains Mann-Kendall trend test results
type MannKendallResult struct {
	Statistic     float64 `json:"statistic"`
	PValue        float64 `json:"p_value"`
	TauB          float64 `json:"tau_b"`          // Kendall's tau-b correlation coefficient
	SlopeEstimate float64 `json:"slope_estimate"` // Sen's slope estimator
	TrendDirection string `json:"trend_direction"`
	IsSignificant bool    `json:"is_significant"`
}

// AnalyzeTrend performs comprehensive trend analysis
func (ta *TrendAnalyzer) AnalyzeTrend(data []float64) (*TrendAnalysisResult, error) {
	if len(data) < 10 {
		return &TrendAnalysisResult{
			TrendType:      "none",
			TrendStrength:  0.0,
			TrendDirection: "stable",
		}, nil
	}

	result := &TrendAnalysisResult{
		TrendComponents: make([]float64, len(data)),
		DetrendedSeries: make([]float64, len(data)),
	}

	// 1. Linear trend analysis
	linearResult := ta.analyzeLinearTrend(data)
	result.LinearTrend = linearResult

	// 2. Polynomial trend analysis
	polyResult := ta.analyzePolynomialTrend(data, 3) // Up to cubic
	result.PolynomialTrend = polyResult

	// 3. Exponential trend analysis
	expResult := ta.analyzeExponentialTrend(data)
	result.ExponentialTrend = expResult

	// 4. Mann-Kendall test for monotonic trend
	mkResult := ta.mannKendallTest(data)
	result.MannKendallTest = mkResult

	// 5. Structural break analysis
	breakResult := ta.detectStructuralBreaks(data)
	result.BreakpointAnalysis = breakResult

	// 6. Change point detection
	result.ChangePoints = ta.detectChangePoints(data)

	// 7. Determine best trend type and calculate components
	result.TrendType, result.TrendStrength, result.TrendDirection = ta.determineBestTrend(linearResult, polyResult, expResult, mkResult)

	// 8. Calculate trend components and detrended series
	result.TrendComponents, result.DetrendedSeries = ta.calculateTrendComponents(data, result.TrendType, linearResult, polyResult, expResult)

	return result, nil
}

// analyzeLinearTrend performs linear regression analysis
func (ta *TrendAnalyzer) analyzeLinearTrend(data []float64) *LinearTrendResult {
	n := len(data)
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = float64(i)
	}

	// Calculate linear regression
	slope, intercept := stat.LinearRegression(x, data, nil)
	
	// Calculate R-squared
	var ssRes, ssTot float64
	meanY := stat.Mean(data, nil)
	
	for i := 0; i < n; i++ {
		predicted := slope*x[i] + intercept
		ssRes += (data[i] - predicted) * (data[i] - predicted)
		ssTot += (data[i] - meanY) * (data[i] - meanY)
	}
	
	rSquared := 1.0 - ssRes/ssTot
	
	// Calculate standard error and t-statistic
	residualVariance := ssRes / float64(n-2)
	
	var sumXSquared float64
	meanX := stat.Mean(x, nil)
	for i := 0; i < n; i++ {
		sumXSquared += (x[i] - meanX) * (x[i] - meanX)
	}
	
	standardError := math.Sqrt(residualVariance / sumXSquared)
	tStatistic := slope / standardError
	
	// Approximate p-value (would use t-distribution in practice)
	pValue := 2.0 * (1.0 - math.Abs(tStatistic)/3.0) // Rough approximation
	if pValue < 0 {
		pValue = 0.001
	}
	if pValue > 1 {
		pValue = 0.999
	}
	
	// 95% confidence interval
	tCritical := 1.96 // Approximate for large samples
	margin := tCritical * standardError
	confInterval := [2]float64{slope - margin, slope + margin}
	
	return &LinearTrendResult{
		Slope:              slope,
		Intercept:          intercept,
		RSquared:           rSquared,
		PValue:             pValue,
		StandardError:      standardError,
		ConfidenceInterval: confInterval,
		IsSignificant:      pValue < ta.significanceLevel,
	}
}

// analyzePolynomialTrend fits polynomial trends of different degrees
func (ta *TrendAnalyzer) analyzePolynomialTrend(data []float64, maxDegree int) *PolynomialTrendResult {
	n := len(data)
	if n < maxDegree+2 {
		maxDegree = n - 2
	}
	if maxDegree < 1 {
		maxDegree = 1
	}

	bestResult := &PolynomialTrendResult{
		Degree:       1,
		Coefficients: []float64{0, 0},
		RSquared:     0,
		AIC:          math.Inf(1),
		BIC:          math.Inf(1),
	}

	// Try different polynomial degrees
	for degree := 1; degree <= maxDegree; degree++ {
		result := ta.fitPolynomial(data, degree)
		
		// Use AIC for model selection
		if result.AIC < bestResult.AIC {
			bestResult = result
		}
	}

	return bestResult
}

// fitPolynomial fits a polynomial of given degree
func (ta *TrendAnalyzer) fitPolynomial(data []float64, degree int) *PolynomialTrendResult {
	n := len(data)
	
	// Create design matrix
	X := mat.NewDense(n, degree+1, nil)
	y := mat.NewVecDense(n, data)
	
	for i := 0; i < n; i++ {
		x := float64(i)
		for j := 0; j <= degree; j++ {
			X.Set(i, j, math.Pow(x, float64(j)))
		}
	}
	
	// Solve normal equations: (X'X)² = X'y
	var XtX mat.Dense
	XtX.Mul(X.T(), X)
	
	var Xty mat.VecDense
	Xty.MulVec(X.T(), y)
	
	// Solve for coefficients
	var beta mat.VecDense
	err := beta.SolveVec(&XtX, &Xty)
	if err != nil {
		// Return simple linear if polynomial fitting fails
		return &PolynomialTrendResult{
			Degree:       1,
			Coefficients: []float64{0, 0},
			RSquared:     0,
			AIC:          math.Inf(1),
			BIC:          math.Inf(1),
		}
	}
	
	// Extract coefficients
	coeffs := make([]float64, degree+1)
	for i := 0; i <= degree; i++ {
		coeffs[i] = beta.AtVec(i)
	}
	
	// Calculate fitted values and R-squared
	var ssRes, ssTot float64
	meanY := stat.Mean(data, nil)
	
	for i := 0; i < n; i++ {
		fitted := 0.0
		x := float64(i)
		for j := 0; j <= degree; j++ {
			fitted += coeffs[j] * math.Pow(x, float64(j))
		}
		
		ssRes += (data[i] - fitted) * (data[i] - fitted)
		ssTot += (data[i] - meanY) * (data[i] - meanY)
	}
	
	rSquared := 1.0 - ssRes/ssTot
	adjustedRSquared := 1.0 - (ssRes/float64(n-degree-1))/(ssTot/float64(n-1))
	
	// Calculate AIC and BIC
	logLikelihood := -float64(n)/2.0 * (math.Log(2*math.Pi) + math.Log(ssRes/float64(n)) + 1)
	k := float64(degree + 1)
	aic := -2*logLikelihood + 2*k
	bic := -2*logLikelihood + k*math.Log(float64(n))
	
	return &PolynomialTrendResult{
		Degree:           degree,
		Coefficients:     coeffs,
		RSquared:         rSquared,
		AdjustedRSquared: adjustedRSquared,
		AIC:              aic,
		BIC:              bic,
		IsSignificant:    rSquared > 0.1, // Simple threshold
	}
}

// analyzeExponentialTrend fits exponential growth/decay model
func (ta *TrendAnalyzer) analyzeExponentialTrend(data []float64) *ExponentialTrendResult {
	// Check if data is suitable for exponential fitting (all positive)
	allPositive := true
	for _, v := range data {
		if v <= 0 {
			allPositive = false
			break
		}
	}
	
	if !allPositive {
		return &ExponentialTrendResult{
			GrowthRate:    0,
			InitialValue:  0,
			RSquared:      0,
			PValue:        1.0,
			IsSignificant: false,
		}
	}
	
	// Transform to log space for linear regression
	logData := make([]float64, len(data))
	for i, v := range data {
		logData[i] = math.Log(v)
	}
	
	// Perform linear regression on log-transformed data
	linearResult := ta.analyzeLinearTrend(logData)
	
	// Transform back to exponential parameters
	growthRate := linearResult.Slope
	initialValue := math.Exp(linearResult.Intercept)
	
	// Calculate R-squared in original space
	var ssRes, ssTot float64
	meanY := stat.Mean(data, nil)
	
	for i, actual := range data {
		predicted := initialValue * math.Exp(growthRate*float64(i))
		ssRes += (actual - predicted) * (actual - predicted)
		ssTot += (actual - meanY) * (actual - meanY)
	}
	
	rSquared := 1.0 - ssRes/ssTot
	
	// Calculate doubling time or half-life
	var doubleTime, halfLife float64
	if growthRate > 0 {
		doubleTime = math.Log(2) / growthRate
	} else if growthRate < 0 {
		halfLife = math.Log(2) / (-growthRate)
	}
	
	return &ExponentialTrendResult{
		GrowthRate:    growthRate,
		InitialValue:  initialValue,
		RSquared:      rSquared,
		PValue:        linearResult.PValue,
		DoubleTime:    doubleTime,
		HalfLife:      halfLife,
		IsSignificant: linearResult.IsSignificant,
	}
}

// mannKendallTest performs Mann-Kendall test for monotonic trend
func (ta *TrendAnalyzer) mannKendallTest(data []float64) *MannKendallResult {
	n := len(data)
	if n < 3 {
		return &MannKendallResult{
			Statistic:      0,
			PValue:         1.0,
			TauB:           0,
			SlopeEstimate:  0,
			TrendDirection: "stable",
			IsSignificant:  false,
		}
	}
	
	// Calculate Mann-Kendall statistic
	var S float64
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if data[j] > data[i] {
				S++
			} else if data[j] < data[i] {
				S--
			}
		}
	}
	
	// Calculate variance of S
	varS := float64(n*(n-1)*(2*n+5)) / 18.0
	
	// Normalized test statistic
	var Z float64
	if S > 0 {
		Z = (S - 1) / math.Sqrt(varS)
	} else if S < 0 {
		Z = (S + 1) / math.Sqrt(varS)
	} else {
		Z = 0
	}
	
	// Calculate p-value (two-tailed)
	pValue := 2.0 * (1.0 - math.Abs(Z)/2.0) // Rough approximation
	if pValue < 0 {
		pValue = 0.001
	}
	if pValue > 1 {
		pValue = 0.999
	}
	
	// Calculate Kendall's tau-b
	tauB := S / (float64(n*(n-1)) / 2.0)
	
	// Sen's slope estimator
	slopes := []float64{}
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if j != i {
				slope := (data[j] - data[i]) / float64(j-i)
				slopes = append(slopes, slope)
			}
		}
	}
	
	// Median of slopes
	var slopeEstimate float64
	if len(slopes) > 0 {
		slopeEstimate = ta.median(slopes)
	}
	
	// Determine trend direction
	var direction string
	if S > 0 {
		direction = "increasing"
	} else if S < 0 {
		direction = "decreasing"
	} else {
		direction = "stable"
	}
	
	return &MannKendallResult{
		Statistic:      S,
		PValue:         pValue,
		TauB:           tauB,
		SlopeEstimate:  slopeEstimate,
		TrendDirection: direction,
		IsSignificant:  pValue < ta.significanceLevel,
	}
}

// detectStructuralBreaks detects structural breaks in the trend
func (ta *TrendAnalyzer) detectStructuralBreaks(data []float64) *BreakpointResult {
	n := len(data)
	if n < 20 { // Need sufficient data for break detection
		return &BreakpointResult{
			BreakpointIndices: []int{},
		}
	}
	
	// Simple breakpoint detection using moving window regression
	minSegmentSize := 10
	breakpoints := []int{}
	
	// Test potential breakpoints
	for i := minSegmentSize; i < n-minSegmentSize; i++ {
		// Fit linear trends before and after potential breakpoint
		preData := data[:i]
		postData := data[i:]
		
		preTrend := ta.analyzeLinearTrend(preData)
		postTrend := ta.analyzeLinearTrend(postData)
		
		// Test if slopes are significantly different
		slopeDiff := math.Abs(preTrend.Slope - postTrend.Slope)
		combinedSE := math.Sqrt(preTrend.StandardError*preTrend.StandardError + 
								postTrend.StandardError*postTrend.StandardError)
		
		if combinedSE > 0 {
			tStat := slopeDiff / combinedSE
			if tStat > 2.0 { // Rough threshold for significance
				breakpoints = append(breakpoints, i)
			}
		}
	}
	
	// Calculate trends before and after main breakpoint
	var preBreakTrend, postBreakTrend float64
	if len(breakpoints) > 0 {
		mainBreak := breakpoints[0]
		preResult := ta.analyzeLinearTrend(data[:mainBreak])
		postResult := ta.analyzeLinearTrend(data[mainBreak:])
		preBreakTrend = preResult.Slope
		postBreakTrend = postResult.Slope
	}
	
	return &BreakpointResult{
		BreakpointIndices: breakpoints,
		PreBreakTrend:     preBreakTrend,
		PostBreakTrend:    postBreakTrend,
	}
}

// detectChangePoints detects change points in the series
func (ta *TrendAnalyzer) detectChangePoints(data []float64) []int {
	n := len(data)
	if n < 10 {
		return []int{}
	}
	
	changePoints := []int{}
	windowSize := 5
	threshold := 2.0 // Standard deviations
	
	// Calculate moving statistics
	for i := windowSize; i < n-windowSize; i++ {
		// Calculate statistics for windows before and after point i
		beforeWindow := data[i-windowSize : i]
		afterWindow := data[i : i+windowSize]
		
		beforeMean := stat.Mean(beforeWindow, nil)
		afterMean := stat.Mean(afterWindow, nil)
		beforeVar := stat.Variance(beforeWindow, nil)
		afterVar := stat.Variance(afterWindow, nil)
		
		// Test for mean change
		pooledSD := math.Sqrt((beforeVar + afterVar) / 2.0)
		if pooledSD > 0 {
			tStat := math.Abs(beforeMean-afterMean) / (pooledSD * math.Sqrt(2.0/float64(windowSize)))
			if tStat > threshold {
				changePoints = append(changePoints, i)
			}
		}
	}
	
	return changePoints
}

// determineBestTrend determines the best trend type based on analysis results
func (ta *TrendAnalyzer) determineBestTrend(linear *LinearTrendResult, poly *PolynomialTrendResult, 
	exp *ExponentialTrendResult, mk *MannKendallResult) (string, float64, string) {
	
	// Prioritize based on significance and goodness of fit
	bestRSquared := 0.0
	bestType := "none"
	
	if linear.IsSignificant && linear.RSquared > bestRSquared {
		bestRSquared = linear.RSquared
		bestType = "linear"
	}
	
	if poly.IsSignificant && poly.RSquared > bestRSquared && poly.Degree > 1 {
		bestRSquared = poly.RSquared
		bestType = "polynomial"
	}
	
	if exp.IsSignificant && exp.RSquared > bestRSquared {
		bestRSquared = exp.RSquared
		bestType = "exponential"
	}
	
	// Use Mann-Kendall for direction if no clear trend detected
	direction := mk.TrendDirection
	if bestType != "none" && linear != nil {
		if linear.Slope > 0 {
			direction = "increasing"
		} else if linear.Slope < 0 {
			direction = "decreasing"
		} else {
			direction = "stable"
		}
	}
	
	return bestType, bestRSquared, direction
}

// calculateTrendComponents calculates trend components based on best fit
func (ta *TrendAnalyzer) calculateTrendComponents(data []float64, trendType string, 
	linear *LinearTrendResult, poly *PolynomialTrendResult, exp *ExponentialTrendResult) ([]float64, []float64) {
	
	n := len(data)
	trendComponents := make([]float64, n)
	detrendedSeries := make([]float64, n)
	
	switch trendType {
	case "linear":
		for i := 0; i < n; i++ {
			trendComponents[i] = linear.Slope*float64(i) + linear.Intercept
			detrendedSeries[i] = data[i] - trendComponents[i]
		}
		
	case "polynomial":
		for i := 0; i < n; i++ {
			x := float64(i)
			trend := 0.0
			for j, coeff := range poly.Coefficients {
				trend += coeff * math.Pow(x, float64(j))
			}
			trendComponents[i] = trend
			detrendedSeries[i] = data[i] - trend
		}
		
	case "exponential":
		for i := 0; i < n; i++ {
			trendComponents[i] = exp.InitialValue * math.Exp(exp.GrowthRate*float64(i))
			detrendedSeries[i] = data[i] - trendComponents[i]
		}
		
	default: // no trend
		copy(detrendedSeries, data)
	}
	
	return trendComponents, detrendedSeries
}

// Helper functions

func (ta *TrendAnalyzer) median(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	
	// Create a copy and sort it
	sorted := make([]float64, len(data))
	copy(sorted, data)
	
	// Simple bubble sort for small arrays
	n := len(sorted)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}
	
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return sorted[n/2]
}