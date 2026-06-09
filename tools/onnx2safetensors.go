// onnx2safetensors — reconstruct a diffusers .safetensors from a diffusers-
// exported ONNX. No Python. Two passes over ModelProto.graph:
//   1. initializers (TensorProto): conv weights / biases / norms already carry
//      their diffusers names; Linear/MatMul weights carry mangled ONNX names
//      (onnx::MatMul_NNNN).
//   2. nodes (NodeProto): each MatMul node's NAME keeps the diffusers module
//      path (/text_model/.../q_proj/MatMul). Map the mangled weight input to
//      "<path>.weight" and TRANSPOSE it — ONNX MatMul wants W[in,out], a
//      PyTorch nn.Linear (which yent.yo's loader expects) wants W[out,in].
//
// Build: go build -o onnx2st onnx2safetensors.go
// Run:   ./onnx2st model.onnx out.safetensors
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type tensor struct {
	name  string
	dims  []int64
	dtype int32
	raw   []byte
}

type node struct {
	name, op string
	inputs   []string
}

// --- minimal protobuf reader -------------------------------------------------

type pb struct {
	b   []byte
	pos int
}

func (p *pb) eof() bool { return p.pos >= len(p.b) }
func (p *pb) varint() uint64 {
	var x uint64
	var s uint
	for p.pos < len(p.b) {
		c := p.b[p.pos]
		p.pos++
		x |= uint64(c&0x7f) << s
		if c < 0x80 {
			break
		}
		s += 7
	}
	return x
}
func (p *pb) tag() (int, int) { t := p.varint(); return int(t >> 3), int(t & 7) }
func (p *pb) ld() []byte {
	n := int(p.varint())
	s := p.b[p.pos : p.pos+n]
	p.pos += n
	return s
}
func (p *pb) skip(wire int) {
	switch wire {
	case 0:
		p.varint()
	case 1:
		p.pos += 8
	case 2:
		p.ld()
	case 5:
		p.pos += 4
	}
}

func parseTensor(b []byte) tensor {
	p := &pb{b: b}
	var t tensor
	for !p.eof() {
		f, w := p.tag()
		switch {
		case f == 1 && w == 2:
			dp := &pb{b: p.ld()}
			for !dp.eof() {
				t.dims = append(t.dims, int64(dp.varint()))
			}
		case f == 1 && w == 0:
			t.dims = append(t.dims, int64(p.varint()))
		case f == 2 && w == 0:
			t.dtype = int32(p.varint())
		case f == 8 && w == 2:
			t.name = string(p.ld())
		case f == 9 && w == 2:
			t.raw = p.ld()
		default:
			p.skip(w)
		}
	}
	return t
}

func parseNode(b []byte) node {
	p := &pb{b: b}
	var n node
	for !p.eof() {
		f, w := p.tag()
		switch {
		case f == 1 && w == 2: // input
			n.inputs = append(n.inputs, string(p.ld()))
		case f == 3 && w == 2: // name
			n.name = string(p.ld())
		case f == 4 && w == 2: // op_type
			n.op = string(p.ld())
		default:
			p.skip(w)
		}
	}
	return n
}

func parseONNX(b []byte) ([]tensor, []node) {
	var ts []tensor
	var ns []node
	p := &pb{b: b}
	for !p.eof() {
		f, w := p.tag()
		if f == 7 && w == 2 { // graph
			g := &pb{b: p.ld()}
			for !g.eof() {
				gf, gw := g.tag()
				switch {
				case gf == 5 && gw == 2: // initializer
					ts = append(ts, parseTensor(g.ld()))
				case gf == 1 && gw == 2: // node
					ns = append(ns, parseNode(g.ld()))
				default:
					g.skip(gw)
				}
			}
		} else {
			p.skip(w)
		}
	}
	return ts, ns
}

// /text_model/encoder/layers.0/self_attn/q_proj/MatMul -> text_model.encoder.layers.0.self_attn.q_proj.weight
func nodeToDiffusers(name string) string {
	name = strings.TrimPrefix(name, "/")
	parts := strings.Split(name, "/")
	if len(parts) > 1 {
		parts = parts[:len(parts)-1] // drop trailing op (MatMul)
	}
	return strings.Join(parts, ".") + ".weight"
}

// transpose a row-major [d0,d1] fp16 matrix into [d1,d0].
func transposeF16(raw []byte, d0, d1 int) []byte {
	out := make([]byte, len(raw))
	for i := 0; i < d0; i++ {
		for j := 0; j < d1; j++ {
			src := (i*d1 + j) * 2
			dst := (j*d0 + i) * 2
			out[dst] = raw[src]
			out[dst+1] = raw[src+1]
		}
	}
	return out
}

func stDtype(o int32) string {
	switch o {
	case 1:
		return "F32"
	case 10:
		return "F16"
	case 11:
		return "F64"
	default:
		return ""
	}
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: onnx2st model.onnx out.safetensors")
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	tensors, nodes := parseONNX(data)

	initSet := map[string]bool{}
	for _, t := range tensors {
		initSet[t.name] = true
	}
	// map mangled MatMul-weight initializer -> diffusers name (and mark for transpose)
	rename := map[string]string{}
	for _, n := range nodes {
		if n.op != "MatMul" && n.op != "Gemm" {
			continue
		}
		for _, in := range n.inputs {
			if initSet[in] {
				rename[in] = nodeToDiffusers(n.name)
			}
		}
	}

	header := map[string]any{}
	var blob []byte
	var transposed, skipped int
	for _, t := range tensors {
		dt := stDtype(t.dtype)
		if dt == "" || len(t.raw) == 0 {
			skipped++
			continue
		}
		name := t.name
		raw := t.raw
		dims := append([]int64(nil), t.dims...)
		if nn, ok := rename[name]; ok {
			name = nn
			if len(dims) == 2 && dt == "F16" {
				raw = transposeF16(raw, int(dims[0]), int(dims[1]))
				dims[0], dims[1] = dims[1], dims[0]
				transposed++
			}
		}
		// VAE ONNX prefixes its initializers with "vae."; yent.yo's loader does not.
		name = strings.TrimPrefix(name, "vae.")
		start := len(blob)
		blob = append(blob, raw...)
		header[name] = map[string]any{
			"dtype":        dt,
			"shape":        dims,
			"data_offsets": []int{start, start + len(raw)},
		}
	}
	hj, _ := json.Marshal(header)
	f, err := os.Create(os.Args[2])
	if err != nil {
		panic(err)
	}
	defer f.Close()
	var lb [8]byte
	binary.LittleEndian.PutUint64(lb[:], uint64(len(hj)))
	f.Write(lb[:])
	f.Write(hj)
	f.Write(blob)
	fmt.Printf("wrote %d tensors (%d MatMul renamed+transposed, %d skipped), %d MB -> %s\n",
		len(header), transposed, skipped, (8+len(hj)+len(blob))/(1<<20), os.Args[2])
}
