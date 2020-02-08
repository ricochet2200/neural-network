package main

import "math"

func Sigmoid(el float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-el))))
}

func SigmoidDerivative(el float32) float32 {
	return Sigmoid(el) * (1.0 - Sigmoid(el))
}
