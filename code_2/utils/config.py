import pandas as pd
import os
sample_rate = 32000
clip_samples = sample_rate * 10

mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

df = pd.read_csv(r"C:\Users\path\ESC-50-master\meta\esc50.csv") # path to csv file
audio_folder = r"C:\Users\path\ESC-50-master\audio" #path to audio

labels = list(df['category'].unique())
lb_to_idx = dict(zip(df['category'], df['target']))
idx_to_lb = dict(zip(df['target'], df['category']))
classes_num = len(df['target'].unique())
file_names = df["filename"].tolist()

file_paths = [os.path.join(audio_folder, name) for name in file_names]

