---
tags: [audio-editing, audio-processing, audio-normalisation, amplitude-control, fade-in, fade-out, spectrogram, audio-effect, filter, delay-effect, dynamic-range-effect, waveshaping, distortion, spatialisation, reverberation, averaging-low-pass-filter, finite-impulse-response-filter, linear-time-invariant-system, impulse-response]
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

## Linear Time Invariant Systems

### What is a signal?

- A list of numbers
- For audio signals, it represents the amplitude sampled over time

### What is a system?

- Something we use to process signals
- If it happens to be a linear, time invariant (LTI) system, we can represent its behaviour as a list of numbers known as an <b>impulse response</b>

### Impulse response
- It is the response of an LTI system to the impulse signal

An impulse is one single maximum amplitude sample as seen in the image below:

![[impulse_signal.png]]

An impulse response would be the response and comes out from a system after we pass an impulse through it:

![[impulse_response.png]]

### What are the characteristics of LTI systems?

Linear systems have very specific characteristics which enable us to do the convolution:
1. Homogeneity
2. Additivity
3. Shift invariance

#### Homogeneity
> Linear with respect to scale

![[homogeneity.png]]

As seen from the graphs, the initial signal has a scale factor of 0.5. When it is passed through the system, the output can also be seen with the same scale factor of 0.5. This proves that the system has homogeneity.

#### Additivity
> Can separately process simple signals and add results together

![[additivity.png]]

Signals 1 and 2 are added together to produce signal 3. The output has the same feature where output 1 and 2 combined gives you output 3. This means that we can predict output 3 if we were only given signals 1 and 2.

#### Shift invariance
> Later signal --> later response

![[shift_invariance.png]]

Simply put, the system has shift invariance if the 2 signals sent at different time comes out with the same amount as the delay between the 2 signals.

## Convolution by hand

The steps in convolution by hand: 
- Decompose (additivity) - Break the signal into its component parts
- Scale (homogeneity) - Scale the impulse response
- Shift (shift invariance) - Shift the impulse response
- Synthesize (addition) - Add the components back together

### Decompose

![[convolution_decompose.png]]

### Scale

![[convolution_scale.png]]

### Shift

![[convolution_shift.png]]

### Synthesize

![[convolution_synthesize.png]]

