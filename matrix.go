package main

import (
	"fmt"
	"math/rand"
)

// Row Major matrix
type Matrix []Vector

func MakeMatrixWithData(r, c int, data []float32) Matrix {
	if len(data) != r*c {
		panic("Wrong amount of data for newly created matrix")
	}
	ret := make(Matrix, r)
	for i := 0; i < r; i++ {
		ret[i] = data[i*c : (i+1)*c]
	}
	return ret
}

func MakeMatrix(r, c int) Matrix {
	data := make([]float32, r*c)
	for i, _ := range data {
		data[i] = Sigmoid(float32(i))
	}
	return MakeMatrixWithData(r, c, data)
}

func MakeRandomMatrix(r, c int) Matrix {
	data := make([]float32, r*c)
	for i, _ := range data {
		data[i] = Sigmoid(rand.Float32())
	}
	return MakeMatrixWithData(r, c, data)
}

func (m Matrix) Clone() Matrix {
	ret := make(Matrix, m.Rows())

	for i := 0; i < m.Rows(); i++ {
		ret[i] = make(Vector, m.Cols())
		row := m[i]
		for j := 0; j < m.Cols(); j++ {
			ret[i][j] = row[j]
		}
	}
	return ret
}

func (m Matrix) T() Matrix {
	ret := make(Matrix, m.Cols())

	for i := 0; i < m.Cols(); i++ {
		ret[i] = make(Vector, m.Rows())
		for j := 0; j < m.Rows(); j++ {
			ret[i][j] = m[j][i]
		}

	}
	return ret
}

func (m Matrix) Dot(other Matrix) Matrix {
	if m.Rows() != other.Cols() && m.Cols() != other.Rows() {
		panic("mismatched matrices in dot multiplication")
	}

	ret := MakeMatrix(m.Rows(), other.Cols())
	for c := 0; c < other.Cols(); c++ {
		col := other.Col(c)
		for r, row := range m {
			ret[r][c] = col.Dot(row)
		}
	}
	return ret
}

func (m Matrix) DotPlusAct(other Matrix, adder Matrix, activation func(float32) float32) Matrix {
	if m.Cols() != other.Rows() {
		panic("mismatched matrices in dot multiplication")
	}

	ret := MakeMatrix(m.Rows(), other.Cols())
	if adder.Cols() != ret.Cols() {
		panic("bias is not the right length")
	}

	for c := 0; c < other.Cols(); c++ {
		col := other.Col(c)
		for r, row := range m {
			ret[r][c] = row.DotPlusAct(col, adder[0][c], activation)
		}
	}
	return ret
}

func (m Matrix) Cols() int {
	return len(m[0])
}

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Col(c int) Vector {
	ret := make(Vector, len(m))
	for i, row := range m {
		ret[i] = row[c]
	}
	return ret
}

func (m Matrix) Add(a float32) Matrix {
	ret := MakeMatrix(len(m), m.Cols())
	for i, row := range m {
		ret[i] = row.Add(a)
	}
	return ret
}

func (m Matrix) Total() float32 {
	ret := float32(0)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			ret += m[i][j]
		}
	}
	return ret
}

func (m Matrix) DiffSquared(other Matrix) Matrix {
	if m.Cols() != other.Cols() || m.Rows() != other.Rows() {
		panic("Other matrix not the same size on Matrix.DiffSquared")
	}

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			diff := m[i][j] - other[i][j]
			m[i][j] = (diff * diff)
		}
	}
	return m
}

func (m Matrix) Diff(other Matrix) Matrix {
	if m.Cols() != other.Cols() || len(m) != len(other) {
		panic("Other matrix not the same size on Matrix.Diff")
	}
	ret := m.Clone()
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			ret[i][j] = m[i][j] - other[i][j]
		}
	}
	return ret
}

func (m Matrix) Mult(scalar float32) Matrix {
	ret := m.Clone()
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			ret[i][j] = m[i][j] * scalar
		}
	}
	return ret
}

// Elementwise multiplication
func (m Matrix) Hadamard(other Matrix) Matrix {
	if m.Cols() != other.Cols() || len(m) != len(other) {
		panic("Other matrix not the same size on Matrix.Hadamard")
	}
	ret := m.Clone()

	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			ret[i][j] = m[i][j] * other[i][j]
		}
	}
	return ret
}

func (m Matrix) Activation(act func(float32) float32) Matrix {
	ret := m.Clone()
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			ret[i][j] = act(m[i][j])
		}
	}
	return ret
}

func (m Matrix) Set(f float32) Matrix {
	ret := m.Clone()
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			ret[i][j] = f
		}
	}
	return ret
}

func (m Matrix) String() string {
	ret := ""
	for _, row := range m {
		ret += "\n["
		for _, el := range row {
			ret += fmt.Sprintf("%.8f ", el)
		}
		ret += "]"
	}
	return ret
}
