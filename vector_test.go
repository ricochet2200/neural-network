package main

import (
	"testing"
)

func TestDot(t *testing.T) {
	a := Vector{1, 2}
	b := Vector{3, 4}
	out := a.Dot(b)
	t.Log(out)
	out2 := b.Dot(a)
	t.Log(out2)
	if out != 11 && out2 != 11 {
		t.Errorf("Incorrect value for vector Dot()")
	}
}
