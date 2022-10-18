---
tags: [audio-editing, audio-processing, audio-normalisation, amplitude-control, fade-in, fade-out, spectrogram, audio-effect, filter, delay-effect, dynamic-range-effect, waveshaping, distortion, spatialisation, reverberation, averaging-low-pass-filter, finite-impulse-response-filter]
aliases: [ISP T2, Intelligent Signal Processing Topic 2]
---

# Reading resources for topic

1. [Tutorial - click and pop removal techniques in Audacity](https://manual.audacityteam.org/man/tutorial_click_and_pop_removal_techniques.html)

# Audio editing in Audacity

[2.001 Audio Editing In Audacity](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/Yfqi2/2-001-audio-editing-in-audacity)

The video in Coursera will cover the following:

- Zooming in Audacity

- Cutting and deleting digital audio

- Trimming and silencing audio regions

- Splitting and moving audio clips

- Copying and pasting audio regions

- Amplitude enveloping

# Audio processing

[2.101 Introduction to audio processing](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/GJwoe/2-101-introduction-to-audio-processing)

The video in Coursera covers the following:

- Amplitude control
- Normalisation
- Fading in and out

## Spectrogram
> It is a visual representation of the spectrum of frequencies of a signal as it varies with time.

## Audio Normalisation
> To normalise audio is to change its overall volume by a fixed amount to reach a target level.

## Audio effects

Watch the video for visual demonstrations, but the points below summarizes what was mentioned:

[2.103 A range of audio effects](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/yFzuR/2-103-a-range-of-audio-effects)

### Filters

- Low pass
	- allows low frequencies to pass and block high frequencies
- High pass
	- allows high frequencies to pass and block low frequencies
- Notch
	- allows low and high frequencies but removes a notch or a certain frequency

#### Properties of filters

Cutoff or centre frequency
- where the filter starts taking effect

Q
- in the case of notch, it allows you to control how wide the notch is

### Delay effects

- Echo
	- Repeat the signal
- Phasing
	- Overlay the signal on itself with slight delay

### Dynamic range effects

- Gate
	- Block when quiet
- Compressor
	- Boost gain when quiet

### Waveshaping and distortion

Applying a function to a signal

- Distortion
	- if amp > x, amp = 0.75. Adds partials to the spectrum

### Spatialisation, reverberation

- Many complex echos
- Simulating an acoustic space

# Implementation of audio processing

## Averaging low-pass filter

[2.201 Filtering. Implementation of an averaging low-pass filter](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/ugCVh/2-201-filtering-implementation-of-an-averaging-low-pass-filter)

> Signal averaging is a signal processing technique that tries to remove unwanted random disturbances from a signal through the process of averaging. Averaging often takes the form of summing a series of signal samples and then dividing that sum by the number of individual samples.
> The following equation represents a N-point moving average filter, with input the array x and outputs the averaged array y:

$$
y(n) = \frac{1}{N}\sum_{k=0}^{N-1}x(n-k)
$$

- The moving average filter
- Implementation of a low-pass moving-average filter in Python
- It is a kind of filter called finite impulse response filters

## Finite Impulse Response (FIR) filter

[2.203 FIR](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/f7z3n/2-203-fir-finite-impulse-response-filters)

> In signal processing, a finite impulse response (FIR) filter is a filter whose impulse response (or response to any finite length input) is of finite duration, because it settles to zero in finite time.

For a general N-tap FIR filter, the nth output is: 
$$
y(n) = \sum_{k=0}^{N-1}h(k)x(n-k)
$$

We have already used this formula, since the moving average filter is a kind of FIR filter.
$$ h(n) = \frac{1}{N}, n = 0,1, ... , N-1 $$

