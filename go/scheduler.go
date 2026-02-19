package main

import "math"

// DDIMScheduler implements deterministic DDIM sampling (eta=0)
// Compatible with BK-SDM-Tiny (trained with PNDM, inference works with any scheduler)
type DDIMScheduler struct {
	alphasCumprod     []float64
	numTrainTimesteps int
	numInferenceSteps int
}

// NewDDIMScheduler creates scheduler with scaled_linear beta schedule
// Matches config: beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000
func NewDDIMScheduler(numTrain int, betaStart, betaEnd float64) *DDIMScheduler {
	// scaled_linear: betas = linspace(sqrt(start), sqrt(end), steps)^2
	betas := make([]float64, numTrain)
	sqrtStart := math.Sqrt(betaStart)
	sqrtEnd := math.Sqrt(betaEnd)
	for i := 0; i < numTrain; i++ {
		beta := sqrtStart + float64(i)/float64(numTrain-1)*(sqrtEnd-sqrtStart)
		betas[i] = beta * beta
	}

	// alphas_cumprod = cumprod(1 - betas)
	alphasCumprod := make([]float64, numTrain)
	prod := 1.0
	for i := 0; i < numTrain; i++ {
		prod *= 1.0 - betas[i]
		alphasCumprod[i] = prod
	}

	return &DDIMScheduler{
		alphasCumprod:     alphasCumprod,
		numTrainTimesteps: numTrain,
	}
}

// SetTimesteps returns the DDIM timestep schedule for inference
// With steps_offset=1: timesteps are [T-step+1, T-2*step+1, ..., 1]
func (s *DDIMScheduler) SetTimesteps(numSteps int) []int {
	s.numInferenceSteps = numSteps
	stepRatio := s.numTrainTimesteps / numSteps
	timesteps := make([]int, numSteps)
	for i := 0; i < numSteps; i++ {
		// Reversed: largest timestep first
		timesteps[i] = (numSteps-1-i)*stepRatio + 1 // +1 for steps_offset=1
	}
	return timesteps
}

// Step performs one DDIM denoising step (eta=0 = deterministic, no added noise)
//
// DDIM update:
//   pred_x0 = (sample - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
//   prev_sample = sqrt(alpha_prev) * pred_x0 + sqrt(1-alpha_prev) * noise_pred
func (s *DDIMScheduler) Step(noisePred *Tensor, timestep int, sample *Tensor) *Tensor {
	stepRatio := s.numTrainTimesteps / s.numInferenceSteps
	prevTimestep := timestep - stepRatio

	// Current and previous alpha_cumprod
	alphaT := s.alphasCumprod[timestep]
	var alphaPrev float64
	if prevTimestep >= 0 {
		alphaPrev = s.alphasCumprod[prevTimestep]
	} else {
		// set_alpha_to_one=false: use alphas_cumprod[0]
		alphaPrev = s.alphasCumprod[0]
	}

	sqrtAlphaT := float32(math.Sqrt(alphaT))
	sqrtOneMinusAlphaT := float32(math.Sqrt(1.0 - alphaT))
	sqrtAlphaPrev := float32(math.Sqrt(alphaPrev))
	sqrtOneMinusAlphaPrev := float32(math.Sqrt(1.0 - alphaPrev))

	out := NewTensor(sample.Shape...)
	for i := range sample.Data {
		// Predict clean sample
		predX0 := (sample.Data[i] - sqrtOneMinusAlphaT*noisePred.Data[i]) / sqrtAlphaT
		// Compute previous noisy sample
		out.Data[i] = sqrtAlphaPrev*predX0 + sqrtOneMinusAlphaPrev*noisePred.Data[i]
	}
	return out
}
