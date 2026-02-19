package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// TensorInfo describes a tensor in the safetensors file
type TensorInfo struct {
	Dtype       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets [2]int  `json:"data_offsets"`
}

// SafeTensors holds a memory-mapped safetensors file
type SafeTensors struct {
	Meta    map[string]TensorInfo
	Data    []byte // raw tensor data (after header)
}

// OpenSafeTensors opens and parses a safetensors file
func OpenSafeTensors(path string) (*SafeTensors, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("file too small: %d bytes", len(data))
	}

	headerLen := binary.LittleEndian.Uint64(data[:8])
	if int(headerLen)+8 > len(data) {
		return nil, fmt.Errorf("header length %d exceeds file size %d", headerLen, len(data))
	}

	headerJSON := data[8 : 8+headerLen]
	tensorData := data[8+headerLen:]

	// Parse header — may contain __metadata__ key which is not a tensor
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &raw); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}

	meta := make(map[string]TensorInfo)
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(v, &info); err != nil {
			return nil, fmt.Errorf("parse tensor %s: %w", k, err)
		}
		meta[k] = info
	}

	return &SafeTensors{Meta: meta, Data: tensorData}, nil
}

// GetFloat32 reads a tensor as float32 slice (converting from float16 if needed)
func (st *SafeTensors) GetFloat32(name string) ([]float32, []int, error) {
	info, ok := st.Meta[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not found", name)
	}

	raw := st.Data[info.DataOffsets[0]:info.DataOffsets[1]]

	numel := 1
	for _, s := range info.Shape {
		numel *= s
	}

	result := make([]float32, numel)

	switch info.Dtype {
	case "F32":
		for i := 0; i < numel; i++ {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "F16":
		for i := 0; i < numel; i++ {
			result[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "I64":
		// Used for position_ids — convert to float32
		for i := 0; i < numel; i++ {
			v := int64(binary.LittleEndian.Uint64(raw[i*8:]))
			result[i] = float32(v)
		}
	default:
		return nil, nil, fmt.Errorf("unsupported dtype %q for tensor %q", info.Dtype, name)
	}

	return result, info.Shape, nil
}

// float16ToFloat32 converts IEEE 754 half-precision to single-precision
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31) // ±0
		}
		// Denormalized — convert to normalized float32
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	case exp == 0x1F:
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000) // ±Inf
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000) // NaN
	default:
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	}
}
