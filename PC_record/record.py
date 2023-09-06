import pyaudio
import wave

# Set the parameters
sample_rate = 44100  # 16 kHz
channels = 2         # Mono
sample_width = 2     # 16-bit audio
record_duration = 5  # Duration of recording in seconds
output_file = "s.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open an audio stream
stream = audio.open(
    format=audio.get_format_from_width(sample_width),
    channels=channels,
    rate=sample_rate,
    input=True,
    frames_per_buffer=1024
)

print("Recording...")

frames = []

# Record audio data
for _ in range(0, int(sample_rate / 1024 * record_duration)):
    data = stream.read(1024)
    frames.append(data)

print("Recording finished.")

# Close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded audio to a WAV file
with wave.open(output_file, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))

print(f"Audio saved as {output_file}")
