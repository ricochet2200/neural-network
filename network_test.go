package main

import (
	"math"
	"math/rand"
	"testing"
)

func tinyNetwork() Network {
	input := &Input{1, 2, "input", nil}
	dense := MakeDense(input, 2, Sigmoid, SigmoidDerivative, "dense1")
	output := MakeDense(dense, 1, Sigmoid, SigmoidDerivative, "dense_output")

	return Network{
		output,
		input,
	}
}

func oneHiddenNetwork() Network {
	input := &Input{1, 2, "input", nil}
	dense := MakeDense(input, 2, Sigmoid, SigmoidDerivative, "dense1")
	hidden := MakeDense(dense, 3, Sigmoid, SigmoidDerivative, "dense_hidden")
	output := MakeDense(hidden, 1, Sigmoid, SigmoidDerivative, "dense_output")

	return Network{
		output,
		input,
	}
}

type Operator int

const (
	And Operator = iota
	Nand
	Or
	Nor
	Xor
)

var params = TrainParams{
	LeastSquaredDerivative,
	LeastSquared,
	.4,
	1,
	1,
	200000,
}

func data(op Operator) chan Data {
	data := make(chan Data, 10)
	go func() {
		for {
			one := float32(math.Round(rand.Float64()))
			two := float32(math.Round(rand.Float64()))
			y := float32(0)
			if op == And {
				y = float32(int(one) & int(two))
			} else if op == Nand {
				y = float32(int(one) &^ int(two))
			} else if op == Or {
				y = float32(int(one) | int(two))
			} else if op == Nor {
				y = float32(int(one) | ^int(two))
			} else if op == Xor {
				y = float32(int(one) ^ int(two))
			}

			data <- Data{
				MakeMatrixWithData(1, 2, []float32{one, two}),
				MakeMatrixWithData(1, 1, []float32{y}),
			}
		}
	}()
	return data
}

func testNetwork(t *testing.T, n func() Network) {

	network := n()
	network.Train(data(And), params)
	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 0}))[0][0])); o != 0 {
		t.Errorf("AND 1 & 0 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 1}))[0][0])); o != 0 {
		t.Errorf("AND 0 & 1 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 0}))[0][0])); o != 0 {
		t.Errorf("AND 0 & 0 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 1}))[0][0])); o != 1 {
		t.Errorf("AND 1 & 1 is not correct: %v", o)
	}

	network = n()
	network.Train(data(Or), params)
	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 0}))[0][0])); o != 1 {
		t.Errorf("OR 1 | 0 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 1}))[0][0])); o != 1 {
		t.Errorf("OR 0 | 1 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 0}))[0][0])); o != 0 {
		t.Errorf("OR 0 | 0 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 1}))[0][0])); o != 1 {
		t.Errorf("OR 1 | 1 is not correct: %v", o)
	}

	network = n()
	network.Train(data(Xor), params)
	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 0}))[0][0])); o != 1 {
		t.Errorf("XOR 1 | 0 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 1}))[0][0])); o != 1 {
		t.Errorf("XOR 0 | 1 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{0, 0}))[0][0])); o != 0 {
		t.Errorf("XOR 0 | 0 is not correct: %v", o)
	}

	if o := math.Round(float64(network.Predict(MakeMatrixWithData(1, 2, []float32{1, 1}))[0][0])); o != 0 {
		t.Errorf("XOR 1 | 1 is not correct: %v", o)
	}

}

func TestTrain(t *testing.T) {
	t.Log("Testing tiny network")
	testNetwork(t, tinyNetwork)

	t.Log("Testing network with one hidden layer")
	testNetwork(t, oneHiddenNetwork)

}
