package main

type Input struct {
	Rows int
	Cols int
	name string
	next Layer
}

func (i Input) Predict(m Matrix) Matrix {
	return i.next.Predict(m)
}

func (i Input) Train(x Matrix, y Matrix, params TrainParams) Matrix {
	gradient := i.next.Train(x, y, params)
	//	log.Println("input gradient:", gradient)
	return gradient
}

func (i Input) Forward(m Matrix) (Matrix, Matrix) {
	return i.next.Forward(m)
}

func (i Input) Back(Matrix, Matrix, Matrix, Matrix, float32) Matrix {
	return nil
}

func (i Input) Prev() Layer {
	return nil
}

func (i Input) Next() Layer {
	return i.next
}

func (i *Input) SetNext(next Layer) {
	i.next = next
}

func (i Input) NumCols() int {
	return i.Cols
}

func (i Input) NumRows() int {
	return i.Rows
}

func (i Input) Name() string {
	return i.name
}

func (i Input) Output() Matrix {
	return nil
}
