package main

import (
	"log"
	"testing"
)

func TestMakeDense(t *testing.T) {
	input := &Input{1, 3, "input", nil}
	dense := MakeDense(input, 2, Sigmoid, SigmoidDerivative, "dense1Test")

	x := MakeMatrix(1, 3)
	_, a := dense.Forward(x)
	if a.Cols() != 2 || a.Rows() != 1 {
		log.Println(a)
		t.Errorf("Dense.Forward() with 1 layer did not work")
	}
	log.Println(a)

	MakeDense(dense, 3, Sigmoid, SigmoidDerivative, "dense2Test")
	a2 := dense.Predict(x)

	if a2.Cols() != 3 || a2.Rows() != 1 {
		log.Println(a2)
		t.Errorf("Dense.Forward() with 2 layers did not work")
	}

}
