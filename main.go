package main

import (
	"log"
	"math"
	"math/rand"
)

func main() {
	data := make(chan Data, 10)
	go func() {
		for {
			one := float32(math.Round(rand.Float64()))
			two := float32(math.Round(rand.Float64()))
			y := float32(int(one) & int(two))

			data <- Data{
				MakeMatrixWithData(1, 2, []float32{one, two}),
				MakeMatrixWithData(1, 1, []float32{y}),
			}
		}
	}()

	input := &Input{1, 2, "input", nil}
	dense := MakeDense(input, 2, Sigmoid, SigmoidDerivative, "dense1")
	hidden := MakeDense(dense, 2, Sigmoid, SigmoidDerivative, "dense_hidden")
	output := MakeDense(hidden, 1, Sigmoid, SigmoidDerivative, "dense_output")

	params := TrainParams{
		LeastSquaredDerivative,
		LeastSquared,
		.4,
		1,
		1,
		200000,
	}

	network := Network{
		output,
		input,
	}

	network.Train(data, params)

	log.Println(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 0})))
	log.Println(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 1})))
	log.Println(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 0})))
	log.Println(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 1})))
}
