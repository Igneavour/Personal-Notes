---
tags: [2D-frequency-analyser, 3D-spectrogram, bin, window, spectral-analysis, pulse-wave, sine-wave, sawtooth-wave, fourier-transformation, discrete-cosine-transformation, synthesis-with-sum-of-cosine, dct-iv, discrete-fourier-transform, complex-numbers]
aliases: [ISP T3, Intelligent Signal Processing Topic 3]
---

# Reading resources

Refer to the notebooks in studies folder, code demonstrations are found there

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

## Analysis problem

> Given a signal and a set of frequencies, how can we find the amplitude of each frequency component?

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

# Discrete Fourier Transform

## Properties of sinusoids: frequency, phase, amplitude

![[properties_of_sinusoidal_waves.png]]

## Updated analysis problem from [[frequency_domain_representations(ISP_T3)#Analysis problem|DCT]]

> Given a signal and a set of frequencies, how can we find the amplitude <b>and phase</b> of each frequency component?

## Complex numbers

- We will be using complex numbers to store phase and amplitude in the same place
- We need a way to compute waveforms from complex numbers: the exponential function
	- np.exp(x)

![[np_exp.png]]

## Complex synthesis

``` python
def synthesize_complex(amps, fs, ts):
	args = np.outer(ts, fs)
	M = np.exp(1j * np.pi * 2 * args)
	ys = np.dot(M, amps)
	return ys
```

## Complex analysis with np.linalg.solve

``` python
def analyze_complex(ys, fs, ts):
	args = np.outer(ts, fs)
	M = np.exp(1j * np.pi * 2 * args)
	amps = np.linalg.solve(M, ys)
	return amps
```

## Complex analysis with DFT

### Nearly DFT

``` python
# nearly DFT as the DFT removes the \N on the last line
def analyze_nearly_dft(ys, fs, ts):
	N = len(fs)
	args = np.outer(ts, fs)
	M = np.exp(1j * np.pi * 2 * args)
	amps = M.conj().transpose().dot(ys) / N
	return amps
```

### Actual DFT

- Calculate freq and time matrix

``` python
def synthesis_matrix(N):
	ts = np.arange(N) / N
	fs = np.arange(N)
	args = np.outer(ts, fs)
	M = np.exp(1j * np.pi * 2 * args)
	return M
```

- Transform

``` python
def dft(ys):
	N = len(ys)
	M = synthesis_matrix(N)
	amps = M.conj().transpose().dot(ys) # no more / N
	return amps
```

## Some subtleties of removing the / N in DFT

### Nearly DFT

![[complex_analysis_nearly_DFT.png]]

### DFT and np.fft

![[nearly_DFT_vs_DFT.png]]

- values for nearly DFT is correct, but by removing the / N, you get DFT. The values of DFT and np.fft are wrong in the sense that they are scaled versions of the real values.

# Analysing real signals

Refer to coursera on how real signals made from 500 and 700hz waves generated ny Audacity is analysed:

[3.197 analysing real signals](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/KbyVF/3-107-analysing-real-signals)

# Fast convolution with DFT

![[fast_convolution_with_DFT.png]]

## Inverse DFT

- 1 extra tool needed for convolution

``` python
def idft(ys):
	N = len(ys)
	M = synthesis_matrix(N)
	amps = M.dot(ys) / N
	return amps
```

# Convolution theorem

> Convolving signals in the time domain is equivalent to multiplying their Fourier transforms in frequency domain.