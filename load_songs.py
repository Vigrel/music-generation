import os

import numpy as np

import midi
import util

all_samples = []
all_lens = []

for root, subdirs, files in os.walk("Dataset"):
	for file in files:
		path = root + "\\" + file
		
		if not (path.endswith('.mid') or path.endswith('.midi')):
			continue
		try:
			samples = midi.midi_to_samples(path)
		except:
			print("ERROR ", path)
			continue
		if len(samples) < 8:
			continue
			
		samples, lens = util.generate_add_centered_transpose(samples)
		all_samples += samples
		all_lens += lens
	
print(f"Saving {len(all_samples)} samples...")
all_samples = np.array(all_samples, dtype=np.uint8)
all_lens = np.array(all_lens, dtype=np.uint32)
np.save('samples.npy', all_samples)
np.save('lengths.npy', all_lens)
print("Done")