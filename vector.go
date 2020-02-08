package main

import (
	"math"
	"math/rand"
)

type Vector []float32

func MakeRandomVector(l int) Vector {
	data := make(Vector, l)
	for i, _ := range data {
		data[i] = rand.Float32()
	}
	return data
}

func MakeVector(l int) Vector {
	data := make(Vector, l)
	for i, _ := range data {
		data[i] = float32(i)
	}
	return data
}

func (v Vector) Clone() Vector {
	ret := make(Vector, len(v))

	for j := 0; j < len(v); j++ {
		ret[j] = v[j]
	}

	return ret
}

func (v Vector) Dot(other Vector) float32 {
	if len(v) != len(other) {
		panic("Vector lengths don't match")
	}

	total := float32(0)
	for i, el := range v {
		total += (el * other[i])
	}
	return total
}

func (v Vector) DotPlusAct(other Vector, adder float32, activation func(float32) float32) float32 {
	if len(v) != len(other) {
		panic("Vector lengths don't match")
	}

	total := float32(0)
	for i, el := range v {
		total += (el * other[i])
	}
	if activation != nil {
		return activation(total + adder)
	}
	return total + adder
}

func (v Vector) Sigmoid() Vector {
	for i, el := range v {
		v[i] = float32(1 / (1 + math.Exp(float64(-el))))
	}
	return v
}

func (v Vector) Add(a float32) Vector {
	for i, el := range v {
		v[i] = el + a
	}
	return v
}

func (v Vector) Minus(other Vector) Vector {
	if len(v) != len(other) {
		panic("Vector lengths don't match")
	}

	ret := v.Clone()
	for i, el := range v {
		ret[i] = el - other[i]
	}
	return ret
}

func (v Vector) Diff(a float32) Vector {
	ret := v.Clone()
	for i, el := range v {
		ret[i] = el - a
	}
	return ret
}
