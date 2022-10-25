---
tags: [2D-frequency-analyser, 3D-spectrogram, bin, window, spectral-analysis, pulse-wave, sine-wave, sawtooth-wave, fourier-transformation, discrete-cosine-transformation, synthesis-with-sum-of-cosine, dct-iv]
aliases: [ISP T3, Intelligent Signal Processing Topic 3]
---

# Introduction to spectral analysers

Coursera video gave a few demonstration of spectral analysers as well as the things you can view using the 3D spectrogram. It would be clearer and easier to view them there (also I am lazy heh): 

## Summary of video:

- What's wrong with the time domain?
- 2D frequency analyser
- Different types of scales
- 3D spectrogram
	- bin
	- window

[3.001 Review of spectral analysis](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/PBNuj/3-001-review-of-spectral-analysis)

# Spectral analysis of periodic signals

## Sine wave

![[sine_wave_with_spectrogram.png]]

## Sawtooth wave

![[sawtooth_wave_with_spectrogram.png]]

## Pulse or square wave

![[pulse_wave_with_spectrogram.png]]

## Adding sines together

![[adding_sines_together.png]]

- note about the frequencies of the five sines: 
	- all 4 frequencies are of the same loudness whereas the the lower harmonics of sawtooth wave is louder than the higher harmonics

## Conclusion

- You can build something like a sawtooth signals out of sine waves
- The spectrum tells us which sine waves we need to build a signal

# Fourier decomposition

> Is a formalisation of the process of figuring out how to make a given signal out of a set of sine waves

$$ S_N(x) = \frac{a_0}{2} + \sum_{n=1}^N(a_ncos(\frac{2\pi}{P}nx) + b_nsin(\frac{2\pi}{P}nx)) $$

# Discrete Cosine Transform

$$ X_k = \sum_{n=0}^{N-1}x_ncos[\frac{\pi}{N}(n + \frac{1}{2})(k + \frac{1}{2})] $$
$$ k = 0, ..., N-1 $$

# Synthesis with sum of cosines

$$ M = cos(2\pi t⨂f) $$
$$ y = Ma $$

a = vector of amplitudes for the 'partials'
f = vector of frequencies for the partials
⨂ = outer product
M will be size of t * size of f

## In code 

``` python
def synthesize2(amps, fs, ts):
	args = np.outer(ts, fs)
	M = np.cos(np.pi * 2 * args)
	ys = np.dot(M, amps)
	return ys

framerate = 11025
ts = np.linspace(0, 1, framerate)
freqs = np.array([1, 2, 3])
amps = np.array([0.5, 0.25, 0.125])
y = synthesize2(amps, freqs, ts)
plt.plot(y)
```

Output:

![[combined_waveform.png]]

## Solving for a with <code>np.linalg.solve(M, ys)</code> 

We can calculate M:

$$ M = cos(2\pi t⨂f) $$
But we need to solve for:

$$ y = Ma $$

We can do that with <code>np.linalg.solve(M, ys)</code> 

### In code

``` python
def analyze1(ys, fs, ts):
	args = np.outer(ts, fs)
	M = np.cos(np.pi * 2 * args)
	amps = np.linalg.solve(M, ys)
	return amps
```

Output:

![[solving_for_a.png]]

# Linalg.solve is a bit slow, Dct-iv does the same thing, given some constraints / features of matrix M

``` python
def dct_iv(ys):
	N = len(ys)
	ts = (0.5 + np.arange(N)) / N
	fs = (0.5 + np.arange(N)) / 2
	args = np.outer(ts, fs)
	M = np.cos(np.pi * 2 * args)
	amps = np.dot(M, ys) / (N/2)
	return amps
```

## Constraints / features of M:
- M is symmetrical (it is its own transpose)
- Choosing ts and fs carefully makes M almost orthogonal except by a factor of 2

## DCT-IV

The choice of frequencies and time steps dictates which version of DCT we are using.

$$ X_k = \sum_{n=0}^{N-1}x_ncos[\frac{\pi}{N}(n + \frac{1}{2})(k + \frac{1}{2})] $$
$$ k = 0, ..., N - 1 $$

