package main

type TrainParams struct {
	LossDerivative  func(Matrix, Matrix) Matrix
	Loss            func(Matrix, Matrix) Matrix
	LearningRate    float32
	BatchSize       int
	BatchesPerEpoch int
	Epochs          int
}
