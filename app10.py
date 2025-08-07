import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_nodes = 100
time_steps = 1000  # Number of time steps for signal generation
frequency = 1  # Frequency of the sinusoidal wave (Hz)
amplitude = 1.0  # Amplitude of the sinusoidal wave
sampling_rate = 1000  # Samples per second
infrared_voltage = 0.7  # Simulated infrared voltage for storage
pulse_width_modulation_frequency = 50  # Frequency of PWM in Hz

# SPWM Signal Generation
def generate_spwm_signal(time, frequency, amplitude):
    # Generate a sinusoidal signal
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    # Generate PWM signal based on the sinusoidal signal
    pwm_signal = np.where(sine_wave > np.random.rand(len(time)), 1, 0)
    return pwm_signal

# Infrared Energy Storage
def infrared_storage(pwm_signal, voltage):
    # Simulate storing data using infrared voltage energy
    stored_signal = pwm_signal * voltage
    return stored_signal

# Directional Transmission (simulating by a shift in phase)
def directional_transmission(stored_signal, phase_shift):
    # Apply a phase shift to simulate transmission towards a given direction
    transmitted_signal = np.roll(stored_signal, phase_shift)
    return transmitted_signal

# Create a time array
time = np.linspace(0, 1, time_steps)

# Generate SPWM Signal
spwm_signal = generate_spwm_signal(time, frequency, amplitude)

# Store the data using infrared voltage energy
infrared_stored_signal = infrared_storage(spwm_signal, infrared_voltage)

# Transmit the signal towards a given direction (simulate by shifting phase)
transmitted_signal = directional_transmission(infrared_stored_signal, phase_shift=100)

# Plot the SPWM signal, stored signal, and transmitted signal
plt.figure(figsize=(15, 8))

plt.subplot(3, 1, 1)
plt.plot(time, spwm_signal, color='blue', label='SPWM Signal')
plt.title('Sinusoidal Pulse Width Modulation (SPWM) Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, infrared_stored_signal, color='red', label='Infrared Stored Signal')
plt.title('Data Stored using Infrared Voltage Energy')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, transmitted_signal, color='green', label='Transmitted Signal')
plt.title('Transmitted Signal towards a Given Direction')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_nodes = 100
time_steps = 1000  # Number of time steps for signal generation
frequency = 1  # Frequency of the sinusoidal wave (Hz)
amplitude = 1.0  # Amplitude of the sinusoidal wave
sampling_rate = 1000  # Samples per second
infrared_voltage = 0.7  # Simulated infrared voltage for storage
pulse_width_modulation_frequency = 50  # Frequency of PWM in Hz
attenuation_factor = 0.5  # Attenuation factor for signal traveling through dense space
noise_intensity = 0.2  # Intensity of noise to simulate interference
multi_path_delay = 50  # Delay for multi-path effect in number of samples
multi_path_amplitude = 0.3  # Amplitude of the delayed multi-path signal

# SPWM Signal Generation
def generate_spwm_signal(time, frequency, amplitude):
    # Generate a sinusoidal signal
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    # Generate PWM signal based on the sinusoidal signal
    pwm_signal = np.where(sine_wave > np.random.rand(len(time)), 1, 0)
    return pwm_signal

# Infrared Energy Storage
def infrared_storage(pwm_signal, voltage):
    # Simulate storing data using infrared voltage energy
    stored_signal = pwm_signal * voltage
    return stored_signal

# Directional Transmission (simulating by a shift in phase)
def directional_transmission(stored_signal, phase_shift):
    # Apply a phase shift to simulate transmission towards a given direction
    transmitted_signal = np.roll(stored_signal, phase_shift)
    return transmitted_signal

# Signal Attenuation in Dense Space
def attenuate_signal(signal, attenuation_factor):
    # Apply exponential decay to simulate attenuation
    attenuation = np.exp(-attenuation_factor * np.arange(len(signal)) / len(signal))
    attenuated_signal = signal * attenuation
    return attenuated_signal

# Add Noise to Simulate Interference
def add_noise(signal, noise_intensity):
    noise = noise_intensity * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Apply Multi-Path Effects
def multi_path_effects(signal, delay, amplitude):
    delayed_signal = np.roll(signal, delay) * amplitude
    combined_signal = signal + delayed_signal
    return combined_signal

# Create a time array
time = np.linspace(0, 1, time_steps)

# Generate SPWM Signal
spwm_signal = generate_spwm_signal(time, frequency, amplitude)

# Store the data using infrared voltage energy
infrared_stored_signal = infrared_storage(spwm_signal, infrared_voltage)

# Transmit the signal towards a given direction (simulate by shifting phase)
transmitted_signal = directional_transmission(infrared_stored_signal, phase_shift=100)

# Attenuate the signal in a densely populated space
attenuated_signal = attenuate_signal(transmitted_signal, attenuation_factor)

# Add noise to the signal
noisy_signal = add_noise(attenuated_signal, noise_intensity)

# Apply multi-path effects
final_signal = multi_path_effects(noisy_signal, multi_path_delay, multi_path_amplitude)

# Plot the SPWM signal, stored signal, transmitted signal, and final signal
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.plot(time, spwm_signal, color='blue', label='SPWM Signal')
plt.title('Sinusoidal Pulse Width Modulation (SPWM) Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, infrared_stored_signal, color='red', label='Infrared Stored Signal')
plt.title('Data Stored using Infrared Voltage Energy')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time, transmitted_signal, color='green', label='Transmitted Signal')
plt.title('Transmitted Signal towards a Given Direction')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time, final_signal, color='purple', label='Final Signal with Attenuation, Noise, and Multi-Path Effects')
plt.title('Final Signal in Dense Space')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_nodes = 100
time_steps = 1000  # Number of time steps for signal generation
frequency = 1  # Frequency of the sinusoidal wave (Hz)
amplitude = 1.0  # Amplitude of the sinusoidal wave
sampling_rate = 1000  # Samples per second
infrared_voltage = 0.7  # Simulated infrared voltage for storage
pulse_width_modulation_frequency = 50  # Frequency of PWM in Hz
attenuation_factor = 0.5  # Attenuation factor for signal traveling through dense space
noise_intensity = 0.2  # Intensity of noise to simulate interference
multi_path_delay = 50  # Delay for multi-path effect in number of samples
multi_path_amplitude = 0.3  # Amplitude of the delayed multi-path signal

# SPWM Signal Generation
def generate_spwm_signal(time, frequency, amplitude):
    # Generate a sinusoidal signal
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    # Generate PWM signal based on the sinusoidal signal
    pwm_signal = np.where(sine_wave > np.random.rand(len(time)), 1, 0)
    return pwm_signal

# Infrared Energy Storage
def infrared_storage(pwm_signal, voltage):
    # Simulate storing data using infrared voltage energy
    stored_signal = pwm_signal * voltage
    return stored_signal

# Directional Transmission (simulating by a shift in phase)
def directional_transmission(stored_signal, phase_shift):
    # Apply a phase shift to simulate transmission towards a given direction
    transmitted_signal = np.roll(stored_signal, phase_shift)
    return transmitted_signal

# Signal Attenuation in Dense Space
def attenuate_signal(signal, attenuation_factor):
    # Apply exponential decay to simulate attenuation
    attenuation = np.exp(-attenuation_factor * np.arange(len(signal)) / len(signal))
    attenuated_signal = signal * attenuation
    return attenuated_signal

# Add Noise to Simulate Interference
def add_noise(signal, noise_intensity):
    noise = noise_intensity * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Apply Multi-Path Effects
def multi_path_effects(signal, delay, amplitude):
    delayed_signal = np.roll(signal, delay) * amplitude
    combined_signal = signal + delayed_signal
    return combined_signal

# Create a time array
time = np.linspace(0, 1, time_steps)

# Generate SPWM Signal
spwm_signal = generate_spwm_signal(time, frequency, amplitude)

# Store the data using infrared voltage energy
infrared_stored_signal = infrared_storage(spwm_signal, infrared_voltage)

# Transmit the signal towards a given direction (simulate by shifting phase)
transmitted_signal = directional_transmission(infrared_stored_signal, phase_shift=100)

# Attenuate the signal in a densely populated space
attenuated_signal = attenuate_signal(transmitted_signal, attenuation_factor)

# Add noise to the signal
noisy_signal = add_noise(attenuated_signal, noise_intensity)

# Apply multi-path effects
final_signal = multi_path_effects(noisy_signal, multi_path_delay, multi_path_amplitude)

# Plot the animated signal
fig, ax = plt.subplots(figsize=(15, 6))
line, = ax.plot([], [], color='purple')
ax.set_xlim(0, time_steps)
ax.set_ylim(-1.5, 1.5)
ax.set_title('Animated Signal Transmission')
ax.set_xlabel('Time Step')
ax.set_ylabel('Amplitude')
ax.grid(True)

# Animation function to update the frame
def animate(frame):
    # Update the signal to show propagation over time
    current_signal = np.roll(final_signal, frame)
    line.set_data(np.arange(len(current_signal)), current_signal)
    return line,

# Create the animation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=20, blit=True)

# Show the animation
plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_nodes = 100
time_steps = 1000  # Number of time steps for signal generation
frequency = 1  # Frequency of the sinusoidal wave (Hz)
amplitude = 1.0  # Amplitude of the sinusoidal wave
sampling_rate = 1000  # Samples per second
infrared_voltage = 0.7  # Simulated infrared voltage for storage
pulse_width_modulation_frequency = 50  # Frequency of PWM in Hz
attenuation_factor = 0.5  # Attenuation factor for signal traveling through dense space
noise_intensity = 0.2  # Intensity of noise to simulate interference
multi_path_delay = 50  # Delay for multi-path effect in number of samples
multi_path_amplitude = 0.3  # Amplitude of the delayed multi-path signal

# Encryption parameters
encryption_keys = [0.5, 1.2, 0.9]  # Different keys for multi-layered encryption

# SPWM Signal Generation
def generate_spwm_signal(time, frequency, amplitude):
    # Generate a sinusoidal signal
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    # Generate PWM signal based on the sinusoidal signal
    pwm_signal = np.where(sine_wave > np.random.rand(len(time)), 1, 0)
    return pwm_signal

# Infrared Energy Storage
def infrared_storage(pwm_signal, voltage):
    # Simulate storing data using infrared voltage energy
    stored_signal = pwm_signal * voltage
    return stored_signal

# Directional Transmission (simulating by a shift in phase)
def directional_transmission(stored_signal, phase_shift):
    # Apply a phase shift to simulate transmission towards a given direction
    transmitted_signal = np.roll(stored_signal, phase_shift)
    return transmitted_signal

# Signal Attenuation in Dense Space
def attenuate_signal(signal, attenuation_factor):
    # Apply exponential decay to simulate attenuation
    attenuation = np.exp(-attenuation_factor * np.arange(len(signal)) / len(signal))
    attenuated_signal = signal * attenuation
    return attenuated_signal

# Add Noise to Simulate Interference
def add_noise(signal, noise_intensity):
    noise = noise_intensity * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Apply Multi-Path Effects
def multi_path_effects(signal, delay, amplitude):
    delayed_signal = np.roll(signal, delay) * amplitude
    combined_signal = signal + delayed_signal
    return combined_signal

# Layered Encryption
def layered_encryption(signal, keys):
    encrypted_signal = signal.copy()
    for key in keys:
        encrypted_signal = np.sin(encrypted_signal * key)  # Encrypting layer
    return encrypted_signal

# Layered Decryption
def layered_decryption(encrypted_signal, keys):
    decrypted_signal = encrypted_signal.copy()
    for key in reversed(keys):
        decrypted_signal = np.arcsin(decrypted_signal) / key  # Decrypting layer
    return decrypted_signal

# Create a time array
time = np.linspace(0, 1, time_steps)

# Generate SPWM Signal
spwm_signal = generate_spwm_signal(time, frequency, amplitude)

# Store the data using infrared voltage energy
infrared_stored_signal = infrared_storage(spwm_signal, infrared_voltage)

# Transmit the signal towards a given direction (simulate by shifting phase)
transmitted_signal = directional_transmission(infrared_stored_signal, phase_shift=100)

# Attenuate the signal in a densely populated space
attenuated_signal = attenuate_signal(transmitted_signal, attenuation_factor)

# Add noise to the signal
noisy_signal = add_noise(attenuated_signal, noise_intensity)

# Apply multi-path effects
final_signal = multi_path_effects(noisy_signal, multi_path_delay, multi_path_amplitude)

# Encrypt the final signal with layered VPN-like encryption
encrypted_signal = layered_encryption(final_signal, encryption_keys)

# Decrypt the signal for verification
decrypted_signal = layered_decryption(encrypted_signal, encryption_keys)

# Plot the encrypted signal and decrypted signal
fig, ax = plt.subplots(2, 1, figsize=(15, 12))

# Plot Encrypted Signal
ax[0].plot(np.arange(len(encrypted_signal)), encrypted_signal, color='purple')
ax[0].set_title('Encrypted Signal w/Layered VPN Protection')
ax[0].set_xlabel('Time Step')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

# Plot Decrypted Signal
ax[1].plot(np.arange(len(decrypted_signal)), decrypted_signal, color='green')
ax[1].set_title('Decrypted Signal w/Layered VPN Decryption')
ax[1].set_xlabel('Time Step')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_nodes = 100
time_steps = 1000  # Number of time steps for signal generation
frequency = 1  # Frequency of the sinusoidal wave (Hz)
amplitude = 1.0  # Amplitude of the sinusoidal wave
sampling_rate = 1000  # Samples per second
infrared_voltage = 0.7  # Simulated infrared voltage for storage
pulse_width_modulation_frequency = 50  # Frequency of PWM in Hz
attenuation_factor = 0.5  # Attenuation factor for signal traveling through dense space
noise_intensity = 0.2  # Intensity of noise to simulate interference
multi_path_delay = 50  # Delay for multi-path effect in number of samples
multi_path_amplitude = 0.3  # Amplitude of the delayed multi-path signal

# Encryption parameters
encryption_keys = [0.5, 1.2, 0.9]  # Different keys for multi-layered encryption

# SPWM Signal Generation
def generate_spwm_signal(time, frequency, amplitude):
    # Generate a sinusoidal signal
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    # Generate PWM signal based on the sinusoidal signal
    pwm_signal = np.where(sine_wave > np.random.rand(len(time)), 1, 0)
    return pwm_signal

# Infrared Energy Storage
def infrared_storage(pwm_signal, voltage):
    # Simulate storing data using infrared voltage energy
    stored_signal = pwm_signal * voltage
    return stored_signal

# Directional Transmission (simulating by a shift in phase)
def directional_transmission(stored_signal, phase_shift):
    # Apply a phase shift to simulate transmission towards a given direction
    transmitted_signal = np.roll(stored_signal, phase_shift)
    return transmitted_signal

# Signal Attenuation in Dense Space
def attenuate_signal(signal, attenuation_factor):
    # Apply exponential decay to simulate attenuation
    attenuation = np.exp(-attenuation_factor * np.arange(len(signal)) / len(signal))
    attenuated_signal = signal * attenuation
    return attenuated_signal

# Add Noise to Simulate Interference
def add_noise(signal, noise_intensity):
    noise = noise_intensity * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Apply Multi-Path Effects
def multi_path_effects(signal, delay, amplitude):
    delayed_signal = np.roll(signal, delay) * amplitude
    combined_signal = signal + delayed_signal
    return combined_signal

# Layered Encryption
def layered_encryption(signal, keys):
    encrypted_signal = signal.copy()
    for key in keys:
        encrypted_signal = np.sin(encrypted_signal * key)  # Encrypting layer
    return encrypted_signal

# Layered Decryption
def layered_decryption(encrypted_signal, keys):
    decrypted_signal = encrypted_signal.copy()
    for key in reversed(keys):
        decrypted_signal = np.arcsin(decrypted_signal) / key  # Decrypting layer
    return decrypted_signal

# Create a time array
time = np.linspace(0, 1, time_steps)

# Generate SPWM Signal
def generate_spwm_signal(time, frequency, amplitude):
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    threshold = np.mean(sine_wave)  # Use mean of sine wave as threshold
    pwm_signal = np.where(sine_wave > threshold, 1, 0)
    return pwm_signal

# Store the data using infrared voltage energy
infrared_stored_signal = infrared_storage(spwm_signal, infrared_voltage)

# Transmit the signal towards a given direction (simulate by shifting phase)
transmitted_signal = directional_transmission(infrared_stored_signal, phase_shift=100)

# Attenuate the signal in a densely populated space
attenuated_signal = attenuate_signal(transmitted_signal, attenuation_factor)

# Add noise to the signal
noisy_signal = add_noise(attenuated_signal, noise_intensity)

# Apply multi-path effects
final_signal = multi_path_effects(noisy_signal, multi_path_delay, multi_path_amplitude)

# Encrypt the final signal with layered VPN-like encryption
encrypted_signal = layered_encryption(final_signal, encryption_keys)

# Decrypt the signal for verification
decrypted_signal = layered_decryption(encrypted_signal, encryption_keys)

# Plot the encrypted signal and decrypted signal
fig, ax = plt.subplots(2, 1, figsize=(15, 12))

# Plot Encrypted Signal
ax[0].plot(np.arange(len(encrypted_signal)), encrypted_signal, color='purple')
ax[0].set_title('Encrypted Signal w/Layered VPN Protection')
ax[0].set_xlabel('Time Step')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

# Plot Decrypted Signal
ax[1].plot(np.arange(len(decrypted_signal)), decrypted_signal, color='green')
ax[1].set_title('Decrypted Signal w/Layered VPN Decryption')
ax[1].set_xlabel('Time Step')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_nodes = 100
time_steps = 1000  # Number of time steps for signal generation
frequency = 1  # Frequency of the sinusoidal wave (Hz)
amplitude = 1.0  # Amplitude of the sinusoidal wave
sampling_rate = 1000  # Samples per second
infrared_voltage = 0.7  # Simulated infrared voltage for storage
pulse_width_modulation_frequency = 50  # Frequency of PWM in Hz
attenuation_factor = 0.5  # Attenuation factor for signal traveling through dense space
noise_intensity = 0.2  # Intensity of noise to simulate interference
multi_path_delay = 50  # Delay for multi-path effect in number of samples
multi_path_amplitude = 0.3  # Amplitude of the delayed multi-path signal

# Encryption parameters
encryption_keys = [0.5, 1.2, 0.9]  # Different keys for multi-layered encryption

# SPWM Signal Generation
def generate_spwm_signal(time, frequency, amplitude):
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    threshold = np.mean(sine_wave)  # Use mean of sine wave as threshold
    pwm_signal = np.where(sine_wave > threshold, 1, 0)
    return pwm_signal

# Infrared Energy Storage
def infrared_storage(pwm_signal, voltage):
    # Simulate storing data using infrared voltage energy
    stored_signal = pwm_signal * voltage
    return stored_signal

# Directional Transmission (simulating by a shift in phase)
def directional_transmission(stored_signal, phase_shift):
    # Apply a phase shift to simulate transmission towards a given direction
    transmitted_signal = np.roll(stored_signal, phase_shift)
    return transmitted_signal

# Signal Attenuation in Dense Space
def attenuate_signal(signal, attenuation_factor):
    # Use a more accurate model for attenuation
    attenuation = np.exp(-attenuation_factor * np.arange(len(signal)) / len(signal))
    attenuated_signal = signal * attenuation
    return attenuated_signal

# Add Noise to Simulate Interference
def add_noise(signal, noise_intensity):
    noise = noise_intensity * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Apply Multi-Path Effects
def multi_path_effects(signal, delay, amplitude):
    delayed_signal = np.roll(signal, delay) * amplitude
    combined_signal = signal + delayed_signal
    return combined_signal

# Layered Encryption
def layered_encryption(signal, keys):
    encrypted_signal = signal.copy()
    for key in keys:
        encrypted_signal = np.sin(encrypted_signal * key)  # Encrypting layer
    return encrypted_signal

# Layered Decryption
def layered_decryption(encrypted_signal, keys):
    decrypted_signal = encrypted_signal.copy()
    for key in reversed(keys):
        decrypted_signal = np.arcsin(decrypted_signal) / key  # Decrypting layer
    return decrypted_signal

# Validate encryption and decryption
def validate_encryption(original_signal, encrypted_signal, decrypted_signal):
    assert np.allclose(original_signal, decrypted_signal, atol=1e-2), "Decryption failed to recover the original signal."


# Create a time array
time = np.linspace(0, 1, time_steps)

# Generate SPWM Signal
spwm_signal = generate_spwm_signal(time, frequency, amplitude)

# Store the data using infrared voltage energy
infrared_stored_signal = infrared_storage(spwm_signal, infrared_voltage)

# Transmit the signal towards a given direction (simulate by shifting phase)
transmitted_signal = directional_transmission(infrared_stored_signal, phase_shift=100)

# Attenuate the signal in a densely populated space
attenuated_signal = attenuate_signal(transmitted_signal, attenuation_factor)

# Add noise to the signal
noisy_signal = add_noise(attenuated_signal, noise_intensity)

# Apply multi-path effects
final_signal = multi_path_effects(noisy_signal, multi_path_delay, multi_path_amplitude)

# Encrypt the final signal with layered VPN-like encryption
encrypted_signal = layered_encryption(final_signal, encryption_keys)

# Decrypt the signal for verification
decrypted_signal = layered_decryption(encrypted_signal, encryption_keys)

# Plot the encrypted signal and decrypted signal
fig, ax = plt.subplots(2, 1, figsize=(15, 12))

# Plot Encrypted Signal
ax[0].plot(np.arange(len(encrypted_signal)), encrypted_signal, color='purple')
ax[0].set_title('Encrypted Signal w/Layered VPN Protection')
ax[0].set_xlabel('Time Step')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

# Plot Decrypted Signal
ax[1].plot(np.arange(len(decrypted_signal)), decrypted_signal, color='green')
ax[1].set_title('Decrypted Signal w/Layered VPN Decryption')
ax[1].set_xlabel('Time Step')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Function to create a gradient color effect
def gradient_color(signal, cmap='viridis'):
    norm = plt.Normalize(signal.min(), signal.max())
    colors = plt.get_cmap(cmap)(norm(signal))
    return colors

# Create a time array
time = np.arange(len(final_signal))

# Generate gradient colors based on final signal
colors = gradient_color(final_signal)

# Plot the final signal with reflection effect
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the final signal
ax.plot(time, final_signal, color='blue', label='Final Signal')

# Add reflection effect
reflection_factor = 0.3
reflection = final_signal * reflection_factor
reflection_color = 'lightblue'

# Plot the reflection
ax.plot(time, -reflection - reflection.min(), color=reflection_color, linestyle='--', alpha=0.6, label='Signal Reflection')

# Add color gradient
for i in range(len(final_signal) - 1):
    ax.plot(time[i:i+2], final_signal[i:i+2], color=colors[i], lw=2)

# Enhance the plot
ax.set_title('Final Signal with Reflection and Color Gradient')
ax.set_xlabel('Time Step')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(True)

plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Function to create a gradient color effect
def gradient_color(signal, cmap='viridis'):
    norm = plt.Normalize(signal.min(), signal.max())
    colors = plt.get_cmap(cmap)(norm(signal))
    return colors

# Create a time array
time = np.arange(len(final_signal))

# Generate gradient colors based on final signal
colors = gradient_color(final_signal)

# Plot the final signal with reflection effect
fig, ax = plt.subplots(figsize=(15, 6))

# Create a smooth line plot with color transitions
for i in range(len(final_signal) - 1):
    ax.plot(time[i:i+2], final_signal[i:i+2], color=colors[i], lw=2)

# Add the final signal plot
ax.plot(time, final_signal, color='blue', alpha=0.5, label='Signal')

# Add reflection effect
reflection_factor = 0.3
reflection = final_signal * reflection_factor
reflection_color = 'lightblue'

# Plot the reflection
ax.plot(time, -reflection - reflection.min(), color=reflection_color, linestyle='--', alpha=0.6, label='Reflection')

# Enhance the plot
ax.set_title('PulseWavefront')
ax.set_xlabel('Time Step')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(True)

plt.show()