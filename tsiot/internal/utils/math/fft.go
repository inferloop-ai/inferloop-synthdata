package math

import (
	"math"
	"math/cmplx"
)

// Complex represents a complex number
type Complex struct {
	Real, Imag float64
}

// NewComplex creates a new complex number
func NewComplex(real, imag float64) Complex {
	return Complex{Real: real, Imag: imag}
}

// Add adds two complex numbers
func (c Complex) Add(other Complex) Complex {
	return Complex{
		Real: c.Real + other.Real,
		Imag: c.Imag + other.Imag,
	}
}

// Sub subtracts two complex numbers
func (c Complex) Sub(other Complex) Complex {
	return Complex{
		Real: c.Real - other.Real,
		Imag: c.Imag - other.Imag,
	}
}

// Mul multiplies two complex numbers
func (c Complex) Mul(other Complex) Complex {
	return Complex{
		Real: c.Real*other.Real - c.Imag*other.Imag,
		Imag: c.Real*other.Imag + c.Imag*other.Real,
	}
}

// Magnitude returns the magnitude of the complex number
func (c Complex) Magnitude() float64 {
	return math.Sqrt(c.Real*c.Real + c.Imag*c.Imag)
}

// Phase returns the phase angle of the complex number
func (c Complex) Phase() float64 {
	return math.Atan2(c.Imag, c.Real)
}

// FFT performs the Fast Fourier Transform on the input signal
func FFT(signal []float64) []Complex {
	n := len(signal)
	if n == 0 {
		return nil
	}
	
	// Convert to complex numbers
	complexSignal := make([]Complex, n)
	for i, val := range signal {
		complexSignal[i] = NewComplex(val, 0)
	}
	
	return fftRecursive(complexSignal)
}

// IFFT performs the Inverse Fast Fourier Transform
func IFFT(spectrum []Complex) []Complex {
	n := len(spectrum)
	if n == 0 {
		return nil
	}
	
	// Conjugate the complex numbers
	conjugated := make([]Complex, n)
	for i, c := range spectrum {
		conjugated[i] = NewComplex(c.Real, -c.Imag)
	}
	
	// Perform FFT on conjugated spectrum
	result := fftRecursive(conjugated)
	
	// Conjugate and scale the result
	for i := range result {
		result[i] = NewComplex(result[i].Real/float64(n), -result[i].Imag/float64(n))
	}
	
	return result
}

// fftRecursive implements the Cooley-Tukey FFT algorithm
func fftRecursive(signal []Complex) []Complex {
	n := len(signal)
	
	// Base case
	if n <= 1 {
		return signal
	}
	
	// Divide
	even := make([]Complex, n/2)
	odd := make([]Complex, n/2)
	
	for i := 0; i < n/2; i++ {
		even[i] = signal[2*i]
		odd[i] = signal[2*i+1]
	}
	
	// Conquer
	evenFFT := fftRecursive(even)
	oddFFT := fftRecursive(odd)
	
	// Combine
	result := make([]Complex, n)
	for k := 0; k < n/2; k++ {
		t := NewComplex(
			math.Cos(-2*math.Pi*float64(k)/float64(n)),
			math.Sin(-2*math.Pi*float64(k)/float64(n)),
		).Mul(oddFFT[k])
		
		result[k] = evenFFT[k].Add(t)
		result[k+n/2] = evenFFT[k].Sub(t)
	}
	
	return result
}

// PowerSpectrum calculates the power spectrum from FFT results
func PowerSpectrum(spectrum []Complex) []float64 {
	power := make([]float64, len(spectrum))
	for i, c := range spectrum {
		power[i] = c.Magnitude() * c.Magnitude()
	}
	return power
}

// Magnitude calculates the magnitude spectrum from FFT results
func MagnitudeSpectrum(spectrum []Complex) []float64 {
	magnitude := make([]float64, len(spectrum))
	for i, c := range spectrum {
		magnitude[i] = c.Magnitude()
	}
	return magnitude
}

// PhaseSpectrum calculates the phase spectrum from FFT results
func PhaseSpectrum(spectrum []Complex) []float64 {
	phase := make([]float64, len(spectrum))
	for i, c := range spectrum {
		phase[i] = c.Phase()
	}
	return phase
}

// FrequencyBins calculates the frequency bins for a given sample rate
func FrequencyBins(n int, sampleRate float64) []float64 {
	bins := make([]float64, n)
	for i := 0; i < n; i++ {
		bins[i] = float64(i) * sampleRate / float64(n)
	}
	return bins
}

// Window functions for preprocessing signals before FFT

// HammingWindow applies a Hamming window to the signal
func HammingWindow(signal []float64) []float64 {
	n := len(signal)
	windowed := make([]float64, n)
	
	for i := 0; i < n; i++ {
		window := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
		windowed[i] = signal[i] * window
	}
	
	return windowed
}

// HanningWindow applies a Hanning window to the signal
func HanningWindow(signal []float64) []float64 {
	n := len(signal)
	windowed := make([]float64, n)
	
	for i := 0; i < n; i++ {
		window := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
		windowed[i] = signal[i] * window
	}
	
	return windowed
}

// BlackmanWindow applies a Blackman window to the signal
func BlackmanWindow(signal []float64) []float64 {
	n := len(signal)
	windowed := make([]float64, n)
	
	a0 := 0.42
	a1 := 0.5
	a2 := 0.08
	
	for i := 0; i < n; i++ {
		window := a0 - a1*math.Cos(2*math.Pi*float64(i)/float64(n-1)) + a2*math.Cos(4*math.Pi*float64(i)/float64(n-1))
		windowed[i] = signal[i] * window
	}
	
	return windowed
}

// Spectrogram computes the spectrogram of a signal
func Spectrogram(signal []float64, windowSize int, overlap int) [][]float64 {
	if windowSize <= 0 || overlap < 0 || overlap >= windowSize {
		return nil
	}
	
	hop := windowSize - overlap
	numWindows := (len(signal)-windowSize)/hop + 1
	
	if numWindows <= 0 {
		return nil
	}
	
	spectrogram := make([][]float64, numWindows)
	
	for i := 0; i < numWindows; i++ {
		start := i * hop
		end := start + windowSize
		
		if end > len(signal) {
			break
		}
		
		window := signal[start:end]
		windowed := HammingWindow(window)
		spectrum := FFT(windowed)
		power := PowerSpectrum(spectrum)
		
		// Take only the first half (positive frequencies)
		spectrogram[i] = power[:len(power)/2]
	}
	
	return spectrogram
}

// PeakDetection finds peaks in the frequency spectrum
func PeakDetection(spectrum []float64, threshold float64) []int {
	var peaks []int
	
	for i := 1; i < len(spectrum)-1; i++ {
		if spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1] && spectrum[i] > threshold {
			peaks = append(peaks, i)
		}
	}
	
	return peaks
}

// DominantFrequency finds the frequency with the highest magnitude
func DominantFrequency(spectrum []Complex, sampleRate float64) float64 {
	maxMagnitude := 0.0
	maxIndex := 0
	
	for i, c := range spectrum[:len(spectrum)/2] {
		magnitude := c.Magnitude()
		if magnitude > maxMagnitude {
			maxMagnitude = magnitude
			maxIndex = i
		}
	}
	
	return float64(maxIndex) * sampleRate / float64(len(spectrum))
}

// Crosscorrelation using FFT for efficiency
func CrosscorrelationFFT(x, y []float64) []float64 {
	// Pad signals to avoid circular correlation artifacts
	n := len(x) + len(y) - 1
	paddedSize := nextPowerOf2(n)
	
	// Pad x
	xPadded := make([]float64, paddedSize)
	copy(xPadded, x)
	
	// Pad y and reverse it for correlation
	yPadded := make([]float64, paddedSize)
	for i := 0; i < len(y); i++ {
		yPadded[i] = y[len(y)-1-i]
	}
	
	// Compute FFTs
	X := FFT(xPadded)
	Y := FFT(yPadded)
	
	// Multiply in frequency domain
	result := make([]Complex, paddedSize)
	for i := 0; i < paddedSize; i++ {
		result[i] = X[i].Mul(Y[i])
	}
	
	// Inverse FFT
	correlation := IFFT(result)
	
	// Extract real part and trim to original size
	output := make([]float64, n)
	for i := 0; i < n; i++ {
		output[i] = correlation[i].Real
	}
	
	return output
}

// nextPowerOf2 finds the next power of 2 greater than or equal to n
func nextPowerOf2(n int) int {
	power := 1
	for power < n {
		power *= 2
	}
	return power
}

// BandpassFilter applies a bandpass filter to the spectrum
func BandpassFilter(spectrum []Complex, lowFreq, highFreq, sampleRate float64) []Complex {
	n := len(spectrum)
	filtered := make([]Complex, n)
	
	for i := 0; i < n; i++ {
		freq := float64(i) * sampleRate / float64(n)
		
		if freq >= lowFreq && freq <= highFreq {
			filtered[i] = spectrum[i]
		} else {
			filtered[i] = NewComplex(0, 0)
		}
	}
	
	return filtered
}

// LowpassFilter applies a lowpass filter to the spectrum
func LowpassFilter(spectrum []Complex, cutoffFreq, sampleRate float64) []Complex {
	return BandpassFilter(spectrum, 0, cutoffFreq, sampleRate)
}

// HighpassFilter applies a highpass filter to the spectrum
func HighpassFilter(spectrum []Complex, cutoffFreq, sampleRate float64) []Complex {
	return BandpassFilter(spectrum, cutoffFreq, sampleRate/2, sampleRate)
}