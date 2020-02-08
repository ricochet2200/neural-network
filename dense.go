package main

import "log"

type Dense struct {
	Weights              Matrix
	Bias                 Matrix
	Cols                 int
	prev                 Layer
	next                 Layer
	Activation           func(float32) float32
	ActivationDerivative func(float32) float32
	name                 string
}

func MakeDense(prev Layer, cols int, activation func(float32) float32, derAct func(float32) float32, name string) *Dense {

	d := &Dense{
		MakeRandomMatrix(prev.NumCols(), cols),
		MakeRandomMatrix(1, cols),
		cols,
		prev,
		nil,
		activation,
		derAct,
		name,
	}
	prev.SetNext(d)
	return d
}

func (d *Dense) Predict(m Matrix) Matrix {

	ret := m.DotPlusAct(d.Weights, d.Bias, d.Activation)
	if d.next != nil {
		return d.next.Predict(ret)
	}
	return ret
}

func (d *Dense) Forward(m Matrix) (Matrix, Matrix) {
	z := m.DotPlusAct(d.Weights, d.Bias, nil)
	return z, z.Activation(d.Activation)
}

func (d *Dense) Train(x Matrix, y Matrix, params TrainParams) Matrix {

	z, a := d.Forward(x)
	if d.next == nil {
		/* loss := params.Loss(a, y).Total()
		log.Println("train loss:", loss)
		if loss > .5 {
			log.Println("high loss for", y)
		}*/
		gradient := params.LossDerivative(a, y)
		return d.Back(x, z, a, gradient, params.LearningRate)
	}

	gradient := d.next.Train(a, y, params)
	return d.Back(x, z, a, gradient, params.LearningRate)
}

func (d *Dense) Back(x Matrix, z Matrix, a Matrix, gradient Matrix, lr float32) Matrix {
	if false {
		log.Println("x", x)
		log.Println("z", z)
		log.Println("a", a)
		log.Println("gradient", gradient)
	}
	weightsCopy := d.Weights.Clone()
	// lossSig is used in all functions
	// lossSig := act' * loss'
	sigPrime := z.Activation(d.ActivationDerivative)
	zPrimeLossPrime := sigPrime.Hadamard(gradient)

	//	log.Println("zPrimeLossPrime", zPrimeLossPrime)

	// 1 * z' * loss'
	d.Bias = d.Bias.Diff(zPrimeLossPrime.Mult(lr))

	//log.Println("weights", d.Weights)

	// a(prev layer) * loss' * z'
	d.Weights = d.Weights.Diff(d.weightUpdate(x, zPrimeLossPrime).Mult(lr))
	//log.Println("weights updated!")
	//log.Println("weights", d.Weights)

	// weights * z' * loss'
	//	log.Println(zPrimeLossPrime.T())
	ret := weightsCopy.Dot(zPrimeLossPrime.T()).T()
	//	log.Println("zPrev loss calculated", ret)
	return ret
}

func (d Dense) weightUpdate(aPrev Matrix, zPrimeLossPrime Matrix) Matrix {
	ret := d.Weights.Clone()
	for i := 0; i < aPrev.Cols(); i++ {
		for j := 0; j < zPrimeLossPrime.Cols(); j++ {
			ret[i][j] = zPrimeLossPrime[0][j] * aPrev[0][i]
		}
	}
	return ret
}

func (d Dense) Prev() Layer {
	return d.prev
}

func (d Dense) Next() Layer {
	return d.next
}

func (d *Dense) SetNext(next Layer) {
	d.next = next
}

func (d Dense) NumCols() int {
	return d.Cols
}

func (d Dense) NumRows() int {
	return len(d.Weights)
}

func (d Dense) Name() string {
	return d.name

}
