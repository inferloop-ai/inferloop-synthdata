package math

import (
	"errors"
	"math"
)

// Matrix represents a 2D matrix
type Matrix struct {
	Rows, Cols int
	Data       [][]float64
}

// Vector represents a mathematical vector
type Vector []float64

// NewMatrix creates a new matrix with the specified dimensions
func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// NewMatrixFromData creates a matrix from existing data
func NewMatrixFromData(data [][]float64) *Matrix {
	if len(data) == 0 {
		return &Matrix{Rows: 0, Cols: 0, Data: [][]float64{}}
	}
	
	rows := len(data)
	cols := len(data[0])
	
	// Validate that all rows have the same length
	for i, row := range data {
		if len(row) != cols {
			panic("all rows must have the same length")
		}
	}
	
	// Deep copy the data
	matrixData := make([][]float64, rows)
	for i := range matrixData {
		matrixData[i] = make([]float64, cols)
		copy(matrixData[i], data[i])
	}
	
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: matrixData,
	}
}

// Identity creates an identity matrix of the specified size
func Identity(size int) *Matrix {
	matrix := NewMatrix(size, size)
	for i := 0; i < size; i++ {
		matrix.Data[i][i] = 1.0
	}
	return matrix
}

// Zeros creates a matrix filled with zeros
func Zeros(rows, cols int) *Matrix {
	return NewMatrix(rows, cols)
}

// Ones creates a matrix filled with ones
func Ones(rows, cols int) *Matrix {
	matrix := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix.Data[i][j] = 1.0
		}
	}
	return matrix
}

// Get returns the element at position (i, j)
func (m *Matrix) Get(i, j int) float64 {
	if i < 0 || i >= m.Rows || j < 0 || j >= m.Cols {
		panic("index out of bounds")
	}
	return m.Data[i][j]
}

// Set sets the element at position (i, j)
func (m *Matrix) Set(i, j int, value float64) {
	if i < 0 || i >= m.Rows || j < 0 || j >= m.Cols {
		panic("index out of bounds")
	}
	m.Data[i][j] = value
}

// Add adds two matrices
func (m *Matrix) Add(other *Matrix) (*Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return nil, errors.New("matrices must have the same dimensions")
	}
	
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] + other.Data[i][j]
		}
	}
	return result, nil
}

// Subtract subtracts another matrix from this matrix
func (m *Matrix) Subtract(other *Matrix) (*Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return nil, errors.New("matrices must have the same dimensions")
	}
	
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] - other.Data[i][j]
		}
	}
	return result, nil
}

// Multiply multiplies two matrices
func (m *Matrix) Multiply(other *Matrix) (*Matrix, error) {
	if m.Cols != other.Rows {
		return nil, errors.New("number of columns in first matrix must equal number of rows in second matrix")
	}
	
	result := NewMatrix(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * other.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result, nil
}

// Scale multiplies the matrix by a scalar
func (m *Matrix) Scale(scalar float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}
	return result
}

// Transpose returns the transpose of the matrix
func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

// Determinant calculates the determinant of a square matrix
func (m *Matrix) Determinant() (float64, error) {
	if m.Rows != m.Cols {
		return 0, errors.New("determinant can only be calculated for square matrices")
	}
	
	if m.Rows == 1 {
		return m.Data[0][0], nil
	}
	
	if m.Rows == 2 {
		return m.Data[0][0]*m.Data[1][1] - m.Data[0][1]*m.Data[1][0], nil
	}
	
	// Use LU decomposition for larger matrices
	lu, perm, sign, err := m.LUDecomposition()
	if err != nil {
		return 0, err
	}
	
	det := float64(sign)
	for i := 0; i < lu.Rows; i++ {
		det *= lu.Data[i][i]
	}
	
	return det, nil
}

// LUDecomposition performs LU decomposition with partial pivoting
func (m *Matrix) LUDecomposition() (*Matrix, []int, int, error) {
	if m.Rows != m.Cols {
		return nil, nil, 0, errors.New("LU decomposition requires a square matrix")
	}
	
	n := m.Rows
	lu := NewMatrixFromData(m.Data)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	sign := 1
	
	for k := 0; k < n-1; k++ {
		// Find pivot
		maxRow := k
		for i := k + 1; i < n; i++ {
			if math.Abs(lu.Data[i][k]) > math.Abs(lu.Data[maxRow][k]) {
				maxRow = i
			}
		}
		
		// Swap rows if necessary
		if maxRow != k {
			lu.Data[k], lu.Data[maxRow] = lu.Data[maxRow], lu.Data[k]
			perm[k], perm[maxRow] = perm[maxRow], perm[k]
			sign = -sign
		}
		
		// Check for singular matrix
		if math.Abs(lu.Data[k][k]) < 1e-14 {
			return nil, nil, 0, errors.New("matrix is singular")
		}
		
		// Eliminate column
		for i := k + 1; i < n; i++ {
			lu.Data[i][k] /= lu.Data[k][k]
			for j := k + 1; j < n; j++ {
				lu.Data[i][j] -= lu.Data[i][k] * lu.Data[k][j]
			}
		}
	}
	
	return lu, perm, sign, nil
}

// Inverse calculates the inverse of the matrix using LU decomposition
func (m *Matrix) Inverse() (*Matrix, error) {
	if m.Rows != m.Cols {
		return nil, errors.New("inverse can only be calculated for square matrices")
	}
	
	n := m.Rows
	lu, perm, _, err := m.LUDecomposition()
	if err != nil {
		return nil, err
	}
	
	inverse := NewMatrix(n, n)
	
	// Solve for each column of the inverse
	for j := 0; j < n; j++ {
		// Create unit vector
		b := make([]float64, n)
		b[j] = 1.0
		
		// Apply permutation
		permB := make([]float64, n)
		for i := 0; i < n; i++ {
			permB[i] = b[perm[i]]
		}
		
		// Forward substitution (solve Ly = b)
		y := make([]float64, n)
		for i := 0; i < n; i++ {
			y[i] = permB[i]
			for k := 0; k < i; k++ {
				y[i] -= lu.Data[i][k] * y[k]
			}
		}
		
		// Back substitution (solve Ux = y)
		x := make([]float64, n)
		for i := n - 1; i >= 0; i-- {
			x[i] = y[i]
			for k := i + 1; k < n; k++ {
				x[i] -= lu.Data[i][k] * x[k]
			}
			x[i] /= lu.Data[i][i]
		}
		
		// Store solution in inverse matrix
		for i := 0; i < n; i++ {
			inverse.Data[i][j] = x[i]
		}
	}
	
	return inverse, nil
}

// Rank calculates the rank of the matrix using SVD
func (m *Matrix) Rank(tolerance float64) int {
	if tolerance <= 0 {
		tolerance = 1e-12
	}
	
	// For simplicity, use a basic row reduction approach
	// In a production system, you'd want to use SVD
	temp := NewMatrixFromData(m.Data)
	rank := 0
	
	for col := 0; col < temp.Cols && rank < temp.Rows; col++ {
		// Find pivot
		pivotRow := -1
		for row := rank; row < temp.Rows; row++ {
			if math.Abs(temp.Data[row][col]) > tolerance {
				pivotRow = row
				break
			}
		}
		
		if pivotRow == -1 {
			continue // This column is all zeros
		}
		
		// Swap rows
		if pivotRow != rank {
			temp.Data[rank], temp.Data[pivotRow] = temp.Data[pivotRow], temp.Data[rank]
		}
		
		// Eliminate below
		for row := rank + 1; row < temp.Rows; row++ {
			if math.Abs(temp.Data[row][col]) > tolerance {
				factor := temp.Data[row][col] / temp.Data[rank][col]
				for j := col; j < temp.Cols; j++ {
					temp.Data[row][j] -= factor * temp.Data[rank][j]
				}
			}
		}
		rank++
	}
	
	return rank
}

// Trace calculates the trace (sum of diagonal elements) of a square matrix
func (m *Matrix) Trace() (float64, error) {
	if m.Rows != m.Cols {
		return 0, errors.New("trace can only be calculated for square matrices")
	}
	
	trace := 0.0
	for i := 0; i < m.Rows; i++ {
		trace += m.Data[i][i]
	}
	return trace, nil
}

// FrobeniusNorm calculates the Frobenius norm of the matrix
func (m *Matrix) FrobeniusNorm() float64 {
	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j] * m.Data[i][j]
		}
	}
	return math.Sqrt(sum)
}

// Vector operations

// NewVector creates a new vector
func NewVector(size int) Vector {
	return make(Vector, size)
}

// NewVectorFromSlice creates a vector from a slice
func NewVectorFromSlice(data []float64) Vector {
	vector := make(Vector, len(data))
	copy(vector, data)
	return vector
}

// Dot calculates the dot product of two vectors
func (v Vector) Dot(other Vector) (float64, error) {
	if len(v) != len(other) {
		return 0, errors.New("vectors must have the same length")
	}
	
	result := 0.0
	for i := 0; i < len(v); i++ {
		result += v[i] * other[i]
	}
	return result, nil
}

// Add adds two vectors
func (v Vector) Add(other Vector) (Vector, error) {
	if len(v) != len(other) {
		return nil, errors.New("vectors must have the same length")
	}
	
	result := make(Vector, len(v))
	for i := 0; i < len(v); i++ {
		result[i] = v[i] + other[i]
	}
	return result, nil
}

// Subtract subtracts another vector from this vector
func (v Vector) Subtract(other Vector) (Vector, error) {
	if len(v) != len(other) {
		return nil, errors.New("vectors must have the same length")
	}
	
	result := make(Vector, len(v))
	for i := 0; i < len(v); i++ {
		result[i] = v[i] - other[i]
	}
	return result, nil
}

// Scale multiplies the vector by a scalar
func (v Vector) Scale(scalar float64) Vector {
	result := make(Vector, len(v))
	for i := 0; i < len(v); i++ {
		result[i] = v[i] * scalar
	}
	return result
}

// Norm calculates the Euclidean norm of the vector
func (v Vector) Norm() float64 {
	sum := 0.0
	for _, val := range v {
		sum += val * val
	}
	return math.Sqrt(sum)
}

// Normalize normalizes the vector to unit length
func (v Vector) Normalize() Vector {
	norm := v.Norm()
	if norm == 0 {
		return v
	}
	return v.Scale(1.0 / norm)
}

// Cross calculates the cross product of two 3D vectors
func (v Vector) Cross(other Vector) (Vector, error) {
	if len(v) != 3 || len(other) != 3 {
		return nil, errors.New("cross product is only defined for 3D vectors")
	}
	
	result := make(Vector, 3)
	result[0] = v[1]*other[2] - v[2]*other[1]
	result[1] = v[2]*other[0] - v[0]*other[2]
	result[2] = v[0]*other[1] - v[1]*other[0]
	
	return result, nil
}

// Distance calculates the Euclidean distance between two vectors
func (v Vector) Distance(other Vector) (float64, error) {
	diff, err := v.Subtract(other)
	if err != nil {
		return 0, err
	}
	return diff.Norm(), nil
}

// ToMatrix converts the vector to a column matrix
func (v Vector) ToMatrix() *Matrix {
	matrix := NewMatrix(len(v), 1)
	for i := 0; i < len(v); i++ {
		matrix.Data[i][0] = v[i]
	}
	return matrix
}

// MatrixVectorMultiply multiplies a matrix by a vector
func MatrixVectorMultiply(matrix *Matrix, vector Vector) (Vector, error) {
	if matrix.Cols != len(vector) {
		return nil, errors.New("matrix columns must equal vector length")
	}
	
	result := make(Vector, matrix.Rows)
	for i := 0; i < matrix.Rows; i++ {
		sum := 0.0
		for j := 0; j < matrix.Cols; j++ {
			sum += matrix.Data[i][j] * vector[j]
		}
		result[i] = sum
	}
	return result, nil
}

// SolveLinearSystem solves Ax = b using LU decomposition
func SolveLinearSystem(A *Matrix, b Vector) (Vector, error) {
	if A.Rows != A.Cols {
		return nil, errors.New("coefficient matrix must be square")
	}
	
	if A.Rows != len(b) {
		return nil, errors.New("matrix rows must equal vector length")
	}
	
	n := A.Rows
	lu, perm, _, err := A.LUDecomposition()
	if err != nil {
		return nil, err
	}
	
	// Apply permutation to b
	permB := make(Vector, n)
	for i := 0; i < n; i++ {
		permB[i] = b[perm[i]]
	}
	
	// Forward substitution (solve Ly = b)
	y := make(Vector, n)
	for i := 0; i < n; i++ {
		y[i] = permB[i]
		for k := 0; k < i; k++ {
			y[i] -= lu.Data[i][k] * y[k]
		}
	}
	
	// Back substitution (solve Ux = y)
	x := make(Vector, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = y[i]
		for k := i + 1; k < n; k++ {
			x[i] -= lu.Data[i][k] * x[k]
		}
		x[i] /= lu.Data[i][i]
	}
	
	return x, nil
}

// QRDecomposition performs QR decomposition using Gram-Schmidt process
func (m *Matrix) QRDecomposition() (*Matrix, *Matrix, error) {
	if m.Rows < m.Cols {
		return nil, nil, errors.New("QR decomposition requires rows >= cols")
	}
	
	Q := NewMatrix(m.Rows, m.Cols)
	R := NewMatrix(m.Cols, m.Cols)
	
	// Convert matrix columns to vectors for easier manipulation
	columns := make([]Vector, m.Cols)
	for j := 0; j < m.Cols; j++ {
		columns[j] = make(Vector, m.Rows)
		for i := 0; i < m.Rows; i++ {
			columns[j][i] = m.Data[i][j]
		}
	}
	
	// Gram-Schmidt process
	for j := 0; j < m.Cols; j++ {
		// Start with the original column
		qj := columns[j]
		
		// Subtract projections onto previous Q columns
		for k := 0; k < j; k++ {
			qk := make(Vector, m.Rows)
			for i := 0; i < m.Rows; i++ {
				qk[i] = Q.Data[i][k]
			}
			
			// Calculate R[k][j] = qk · columns[j]
			dot, _ := qk.Dot(columns[j])
			R.Data[k][j] = dot
			
			// Subtract projection
			projection := qk.Scale(dot)
			qj, _ = qj.Subtract(projection)
		}
		
		// Normalize qj
		norm := qj.Norm()
		if norm < 1e-14 {
			return nil, nil, errors.New("matrix is rank deficient")
		}
		
		R.Data[j][j] = norm
		qj = qj.Normalize()
		
		// Store in Q
		for i := 0; i < m.Rows; i++ {
			Q.Data[i][j] = qj[i]
		}
	}
	
	return Q, R, nil
}

// EigenvaluesSymmetric calculates eigenvalues of a symmetric matrix using the QR algorithm
// This is a simplified implementation for educational purposes
func (m *Matrix) EigenvaluesSymmetric(maxIterations int) ([]float64, error) {
	if m.Rows != m.Cols {
		return nil, errors.New("eigenvalue calculation requires a square matrix")
	}
	
	// Check if matrix is symmetric
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if math.Abs(m.Data[i][j]-m.Data[j][i]) > 1e-12 {
				return nil, errors.New("this implementation only works for symmetric matrices")
			}
		}
	}
	
	A := NewMatrixFromData(m.Data)
	
	// QR algorithm
	for iter := 0; iter < maxIterations; iter++ {
		Q, R, err := A.QRDecomposition()
		if err != nil {
			return nil, err
		}
		
		A, err = R.Multiply(Q)
		if err != nil {
			return nil, err
		}
		
		// Check for convergence (simplified)
		if iter > 10 && isUpperTriangular(A, 1e-10) {
			break
		}
	}
	
	// Extract eigenvalues from diagonal
	eigenvalues := make([]float64, A.Rows)
	for i := 0; i < A.Rows; i++ {
		eigenvalues[i] = A.Data[i][i]
	}
	
	return eigenvalues, nil
}

// isUpperTriangular checks if a matrix is upper triangular within tolerance
func isUpperTriangular(m *Matrix, tolerance float64) bool {
	for i := 1; i < m.Rows; i++ {
		for j := 0; j < i; j++ {
			if math.Abs(m.Data[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}

// PseudoInverse calculates the Moore-Penrose pseudoinverse using SVD
// This is a simplified implementation
func (m *Matrix) PseudoInverse(tolerance float64) (*Matrix, error) {
	// For full SVD implementation, we'd need more complex algorithms
	// This provides a basic implementation using normal equations for overdetermined systems
	
	if m.Rows >= m.Cols {
		// Overdetermined system: pinv(A) = (A^T * A)^(-1) * A^T
		At := m.Transpose()
		AtA, err := At.Multiply(m)
		if err != nil {
			return nil, err
		}
		
		AtAInv, err := AtA.Inverse()
		if err != nil {
			return nil, err
		}
		
		return AtAInv.Multiply(At)
	} else {
		// Underdetermined system: pinv(A) = A^T * (A * A^T)^(-1)
		At := m.Transpose()
		AAt, err := m.Multiply(At)
		if err != nil {
			return nil, err
		}
		
		AAtInv, err := AAt.Inverse()
		if err != nil {
			return nil, err
		}
		
		return At.Multiply(AAtInv)
	}
}

// LeastSquares solves the least squares problem min ||Ax - b||²
func LeastSquares(A *Matrix, b Vector) (Vector, error) {
	if A.Rows != len(b) {
		return nil, errors.New("matrix rows must equal vector length")
	}
	
	// Normal equation: A^T * A * x = A^T * b
	At := A.Transpose()
	AtA, err := At.Multiply(A)
	if err != nil {
		return nil, err
	}
	
	AtB, err := MatrixVectorMultiply(At, b)
	if err != nil {
		return nil, err
	}
	
	return SolveLinearSystem(AtA, AtB)
}