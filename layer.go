package main

type Layer interface {
	Forward(Matrix) (Matrix, Matrix)
	Back(Matrix, Matrix, Matrix, Matrix, float32) Matrix
	Train(Matrix, Matrix, TrainParams) Matrix
	Predict(Matrix) Matrix
	Prev() Layer
	Next() Layer
	SetNext(Layer)
	NumCols() int
	NumRows() int
	Name() string
}
