import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import time
import re
from datetime import datetime

# configs
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 # 44.1 kHz
INPUT_DEVICE_INDEX = 1   # shure mv7+, maybe change

CLIP_NEAR_FS = 32760


log_name = input("Enter a log name for this recording: ").strip()
if not log_name:
    log_name = "session"
# sanitize filename
log_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", log_name)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base = f"{log_name}_{timestamp}"
wav_path = f"{base}.wav"
csv_path = f"{base}_rms.csv"

print(f"\nOutput files:")
print(f"  Audio WAV: {wav_path}")
print(f"  RMS CSV  : {csv_path}\n")


p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

print("Available Input Devices:")
for i in range(0, numdevices):
    device_info = p.get_device_info_by_host_api_device_index(0, i)
    if device_info.get('maxInputChannels') > 0:
        print(f"  Input Device id {i} - {device_info.get('name')}")

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK, input_device_index=INPUT_DEVICE_INDEX)

wav_file = wave.open(wav_path, 'wb')
wav_file.setnchannels(CHANNELS)
wav_file.setsampwidth(p.get_sample_size(FORMAT))  # 2 bytes for int16
wav_file.setframerate(RATE)

csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["t_seconds", "rms_linear", "peak_abs", "clip_flag"])  # header

print("\nRecording...\n")


loudness_rms = []
elapsed_sec = 0.0
total_samples = 0
clip_count = 0
t0 = time.time()

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)


        wav_file.writeframes(data)


        x = np.frombuffer(data, dtype=np.int16).astype(np.float32)


        rms = np.sqrt(np.mean(x * x))
        peak = float(np.max(np.abs(x)))
        clip_flag = 1 if peak >= CLIP_NEAR_FS else 0
        if clip_flag:
            clip_count += 1


        total_samples += x.size
        elapsed_sec = total_samples / RATE


        csv_writer.writerow([f"{elapsed_sec:.6f}", f"{rms:.6f}", f"{peak:.1f}", clip_flag])
        loudness_rms.append(rms)

except KeyboardInterrupt:
    print("\nstopping because ctrl+c.")

finally:
    # cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()

    wav_file.close()
    csv_file.close()

    dur = time.time() - t0
    print(f"\nSaved .wav to {wav_path}")
    print(f"Saved RMS logs to {csv_path}")
    print(f"Duration: {dur:.1f}s, samples: {total_samples}, chunks clipped: {clip_count}")

    # plot rms
    if loudness_rms:
        plt.figure()
        plt.plot(loudness_rms)
        plt.xlabel("Time (chunks)")
        plt.ylabel("Loudness (RMS)")
        plt.title("Sound Intensity Over Time")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
