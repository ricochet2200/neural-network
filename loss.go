package main

func LeastSquaredDerivative(pred Matrix, truth Matrix) Matrix {
	return pred.Diff(truth).Mult(2)
}

func LeastSquared(pred Matrix, truth Matrix) Matrix {
	return pred.DiffSquared(truth)
}
