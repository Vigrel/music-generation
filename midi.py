from mido import Message, MidiFile, MidiTrack

num_notes = 96
samples_per_measure = 96

def samples_to_midi(samples, fname, ticks_per_sample, thresh=0.5):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	ticks_per_beat = mid.ticks_per_beat
	ticks_per_measure = 4 * ticks_per_beat
	ticks_per_sample = ticks_per_measure / samples_per_measure
	abs_time = 0
	last_time = 0
	for sample in samples:
		for y in range(sample.shape[0]):
			abs_time += ticks_per_sample
			for x in range(sample.shape[1]):
				note = x + (128 - num_notes)/2
				if sample[y,x] >= thresh and (y == 0 or sample[y-1,x] < thresh):
					delta_time = abs_time - last_time
					track.append(Message('note_on', note=int(note), velocity=127, time=int(delta_time + 8)))
					last_time = abs_time
				if sample[y,x] >= thresh and (y == sample.shape[0]-1 or sample[y+1,x] < thresh):
					delta_time = abs_time - last_time
					track.append(Message('note_off', note=int(note), velocity=127, time=int(delta_time + 8)))
					last_time = abs_time
	mid.save(fname)