---
tags: [audio-feature, energy, spectrum, fundamental-frequency, loudness, brightness, pitch, rhythm, time-domain-feature, root-mean-squared-energy, zero-crossing-rate, amplitude-envelope, frequency-domain-feature, spectral-centroid, spectral-rolloff, spectral-flatness, spectral-flux, spectral-slope, spectral-spread, chroma, mel-frequency-ceptral-coefficients]
aliases: [ISP T4, Intelligent Signal Processing Topic 4]
---

# What is an Audio Feature?

> An audio feature is a measurement of a particular characteristic of an audio signal, and it gives us insight into what the signal contains.

Audio feature examples:

- Energy
- Spectrum
- Fundamental frequency
- Loudness
- Brightness
- Pitch
- Rhythm

# How can audio features be extracted?

- Audio features can be measured by running an algorithm on an audio signal that will return a number, or a set of numbers that quantify the characteristic that the specific algorithm is intended to measure
- Real-time / Non-real-time extraction

Some feature extraction toolboxes:

- Librosa - A python package for music and audio analysis
- YAAFE (Yet Another Audio Feature Extractor) - Low level feature extraction library written in C++ that can be used with Python and Matlab
- Meyda - Low-level feature extraction written in Javascript and so is aimed toward web-based and real time applications

# Applications of audio features

- Real-time audio visualisations
- Visual score generation
- Speech recognition
- Music recommendation
- Music genre classification
- Feature-based synthesis
- Feature extraction linked to audio effects

# Time-domain features

- Energy
- Root-Mean-Squared energy (RMSE) - loudness
	- calculates the RMS of a signal or a frame of a signal
	- indicator of loudness
	- less sensitive to outliers than amplitude envelope, also an indicator of loudness
	- used for: audio segmentation, music genre classification

$$ RMSE = \sqrt{\frac{1}{N}\sum_n |x(n)|^2} $$

![[rmse.png]]

- Zero crossing rate (ZCR)
	- the number of times that the signal crosses the zero value in a frame
	- recognition of percussive vs pitched sounds
	- used for: monophonic pitch estimation, in speech recognition for voice activity detection

![[zcr.png]]

- Amplitude envelope (AE)

# Frequency-domain features

- Spectral centroid - deduce brightness and then, can guess what kind of instruments are used
	- indicates where the spectral 'centre of mass', the spectral 'centre of gravity' for a sound is located
	- spectral centroid is an indicator of the 'brightness' of a given sound 
	- for example, spectral centroid can be used to classify a bass guitar (low spectral centroid) from a trumpet (high spectral centroid)

![[spectral_centroid_example.png]]

- Spectral rolloff
- Spectral flatness
- Spectral flux
- Spectral slope
- Spectral spread
- Mel-frequency cepstral coefficients (MFCC)
- Chroma

# Real-time audio visualisations with p5.js - time domain features

Video demonstration on coursera: 

[4.101 real-time audio visualisations with p5.js - time domain features](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/1yXSb/4-101-real-time-audio-visualisations-with-p5-js-time-domain-features)

# Real-time audio visualisations with p5.js - frequency domain features

Video demonstration on coursera:

[4.102 Real-time audio visualisations with p5.js – frequency domain features (1)](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/b2OJt/4-102-real-time-audio-visualisations-with-p5-js-frequency-domain-features-1)

[4.103 Real-time audio visualisations with p5.js – frequency domain features (2)](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/pURiw/4-103-real-time-audio-visualisations-with-p5-js-frequency-domain-features-2)

# Audio signal feature extraction with Python

[4.201 Audio signal feature extraction with Python – time-domain features (1)](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/5lrrF/4-201-audio-signal-feature-extraction-with-python-time-domain-features-1)

[4.203 Audio signal feature extraction with Python – time-domain features (2)](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/KOlI0/4-203-audio-signal-feature-extraction-with-python-time-domain-features-2)

[4.205 Audio signal feature extraction with Python - frequency-domain features](https://www.coursera.org/learn/uol-cm3065-intelligent-signal-processing/lecture/Lln9K/4-205-audio-signal-feature-extraction-with-python-frequency-domain-features)

