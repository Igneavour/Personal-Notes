---
tags: [intelligent-signal-processing, sound, wavelength, amplitude, crest, trough, frequency, period, period-frequency-formula, speed-wavelength-frequency-formula, sound-physical-property, time-period, duration, sound-perceptual-property, pitch, loudness, timbre, pure-tone, sampling-rate, nyquist-shannon-sampling-theorem, bit-depth, clipping, digital-audio-representation, spectrogram]
aliases: [ISP T1, Intelligent Signal Processing Topic 1]
---

# Sound
- Is a form of energy produced by vibrating matter.
- Is a form of mechanical energy that needs a medium to propagate.
- Cannot travel in vacuum, it travels in the form of mechanical waves through solid, liquid and gas.

# Wavelength and amplitude

![[soundwaves_graph.png]]

- The <b>wavelength</b> of a sound wave is the distance between two successive crests (or successive troughs) of the wave.
- The <b>amplitude</b> of a sound wave is the maximum change in pressure or density that the vibrating object produces in the surrounding air.

Pressure is measured in pascals (Pa). Although for practical reasons we usually use the dB SPL (sound pressure level in dB) scale for measuring sound amplitude.

# Frequency and period

![[soundwaves_graph_2.png]]

- The <b>frequency</b> of a wave refers to the number of compressions or rarefactions that pass a given point per unit of time.
- Frequency is measured in hertz (Hz).
- The <b>time period</b> is the time a sound wave takes to go through a compression-rarefaction cycle.

The period (T) is the inverse of the frequency (f):
$$
 T = \frac{1}{f}, f = \frac{1}{T}
$$

Example: T = 0.1s, then: $$ f = \frac{1}{0.1s} = 10Hz $$

# Sound speed, wavelength and frequency

There is also a direct relation between sound speed (v), wavelength (λ) and frequency (f):
$$ v = f * \lambda $$
# Physical and perceptual properties of sound

| Physical properties | Perceptual properties |
| ------------------- | --------------------- |
| Frequency           | Pitch                 |
| Amplitude           | Loudness              |
| Waveform            | Timbre                |
| Wavelength          |                       |
| Time period         |                       |
| Duration            |                       |

# Pure tone

- A pure tone is a sound with a sinusoidal waveform of any frequency, phase and amplitude. A pure tone is composed of a single frequency.
- Real-world sounds are much more complex with multiple frequencies.

# Perception of perceptual properties

## Pitch

![[perception_of_pitch.png]]

## Loudness

Loudness is a sensation related to the amplitude of sound waves.

To express sound amplitude in terms of pascal is inconvenient due to the wide range of values within the hearing threshold. For practical reasons, we use a logarithmic scale for measuring sound amplitude. The formula below converts sound pressure (p) into the dB SPL scale.
$$ SPL = 20log_{10}(\frac{p}{p_{ref}})dB $$

![[spl_frequency_graph.png]]

The graph shown above basically shows how different sounds at different SPL and frequency levels could be perceived to be equally loud. E.g: A sound with 20Hz and 90 dB SPL seems to be as loud as a sound at 1000hz and 20 dB SPL. 

## Timbre

- Its what differentiates two sounds of same frequencies and amplitude. 
- Its related to the physical property or the shape of the wave or waveform.

![[sound_different_timbre_graphs.png]]

# Sampling rate

- The number of measurements (samples) taken per second is called the sampling rate (Hz). 
- Each measurement of the waveform's amplitude is called a sample.

## Nyquist-Shannon sampling theorem

> Nyquist frequency = 1/2 * Sampling rate

- Signal above the Nyquist frequency is not recorded properly by ADCs (Analog-to-Digital Converter), introducing artificial frequencies in a process called <b>aliasing</b>.
- The sampling rate must be at least *twice* the frequency of the signal being sampled.
- To avoid aliasing, modern devices usually have an anti-aliasing filter that is a low-pass filter that eliminates frequencies above the Nyquist frequency before audio reaches the ADC.

# Bit depth

- Number of bits used to record the amplitude measurements.
- The more bits we use, the more accurately we can measure the analogue waveform but it also means more hard disk space or memory size (same as sampling rate).
- Common bit widths used for digital sound representation are 8,16,24 and 32 bits.

![[2pow1_bit_depth_graph.png]] ![[2pow2_bit_depth_graph.png]]
![[2pow3_bit_depth_graph.png]]

# Clipping

- Clipping occurs when the level of the input signal is too high and the ADC cannot assign to the signal to the right measurements that ADC assigns maximum or minimum amplitude values too many samples in a row.
- To avoid this, we need to watch the input level and ensure it doesn't reach zero.

# Simple calculation of file's byte size example

![[calculate_byte_size_file.png]]

- Since its stereo, there is two channels.
- 1 byte = 8 bits, therefore 84672000 bits = 10584000 B = 10.584 MB

# Digital audio representation

## Time domain representation 

![[time_domain_representation.png]]

## Frequency domain representation

![[frequency_domain_representation.png]]

## Spectrogram 
- A visual representation of the spectrum of frequencies of a signal as it varies with time.

## Decibels relative to full scale (dBFS) in Audacity

Normalised values [-1, 1] --> dBFS = 20 * log<sub>10</sub>(abs(value)) --> Amplitude in dBFS
Examples:
Normalised value = 0 --> 20 * log<sub>10</sub>(abs(0)) = -∞ dBFS
Normalised value = 1 --> 20 * log<sub>10</sub>(abs(1)) = 0 dBFS
Normalised value = -1 --> 20 * log<sub>10</sub>(abs(-1)) = 0 dBFS