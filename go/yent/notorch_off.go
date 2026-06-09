//go:build !notorch

// notorch_off.go — default build (no notorch tag). The two Yents' matvec stays
// on the native packed kernels; `go build -tags notorch` swaps in nt_qmatvec.
package yent

const useNotorch = false

func notorchQMatvec(out []float32, w []byte, dtype uint32, x []float32, rows, cols int) bool {
	return false
}
