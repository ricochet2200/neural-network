package main

import "testing"

func a() Matrix {
	return MakeMatrixWithData(2, 2, []float32{1, 2, 3, 4})
}

func b() Matrix {
	return MakeMatrixWithData(2, 2, []float32{5, 6, 7, 8})
}

func c() Matrix {
	return MakeMatrixWithData(2, 3, []float32{1, 2, 3, 4, 5, 6})
}

func d() Matrix {
	return MakeMatrixWithData(3, 2, []float32{1, 2, 3, 4, 5, 6})
}

func TestMakeMatrix(t *testing.T) {
	m := MakeMatrix(1, 2)
	if len(m) != 1 || len(m[0]) != 2 {
		t.Errorf("Failed to create correctly sized matrix")
	}
}

func TestMakeMatrixWithData(t *testing.T) {
	data := []float32{0, 1, 2, 3, 4, 5}

	if MakeMatrixWithData(2, 3, data).Cols() != 3 {
		t.Errorf("Failed to create matrix with data for 3 cols")
	}

	if MakeMatrixWithData(3, 2, data).Cols() != 2 {
		t.Errorf("Failed to create matrix with data for 2 cols")
	}

	v := a().Col(1)
	if v[0] != 2 || v[1] != 4 {
		t.Errorf("Failed to create matrix, col 1 was %v", v)
	}
}

func TestT(t *testing.T) {
	at := a().T()
	if at[0][0] != 1 || at[0][1] != 3 || at[1][0] != 2 || at[1][1] != 4 {
		t.Errorf("Failed to transpose matrix %v", at)
	}

	ct := c().T()
	if ct[0][0] != 1 || ct[0][1] != 4 || ct[1][0] != 2 || ct[1][1] != 5 || ct[2][0] != 3 || ct[2][1] != 6 {
		t.Errorf("Failed to transpose matrix %v", ct)
	}
}

func TestDiffSquared(t *testing.T) {
	ds := a().DiffSquared(a())
	if ds.Total() != 0 {
		t.Errorf("DiffSqured isn't working")
	}
}

func TestCols(t *testing.T) {
	if a().Cols() != 2 {
		t.Errorf("Matrix.Cols() isn't working")
	}
}

func TestDotMatrix(t *testing.T) {
	out1 := a().Dot(b())
	if out1[0][0] != 19 || out1[0][1] != 22 || out1[1][0] != 43 || out1[1][1] != 50 {
		t.Errorf("Bad Matrix.Dot()")
		t.Errorf("%v", a())
		t.Errorf("%v", out1)

	}
	out2 := c().Dot(d())
	if out2[0][0] != 22 || out2[0][1] != 28 || out2[1][0] != 49 || out2[1][1] != 64 {
		t.Errorf("Bad Matrix.Dot()")
		t.Errorf("%v", c())
		t.Errorf("%v", d())
		t.Errorf("%v", out2)

	}
}

func TestDiff(t *testing.T) {
	a := MakeMatrix(1, 2)
	b := MakeMatrix(1, 2)
	c := a.Diff(b)
	if false {
		t.Errorf("%v", a)
		t.Errorf("%v", b)
		t.Errorf("%v", c)
	}
}
