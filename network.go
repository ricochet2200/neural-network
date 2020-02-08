package main

type Network struct {
	Output Layer
	Input  Layer
}

func (n Network) Train(data <-chan Data, params TrainParams) {
	for epochs := 0; epochs < params.BatchesPerEpoch*params.Epochs; epochs++ {
		data := <-data
		n.Input.Train(data.X, data.Y, params)
		//	log.Println(epochs, data.X, data.Y)
	}
}

func (n Network) Predict(m Matrix) Matrix {
	return n.Input.Predict(m)
}
