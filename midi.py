import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pypianoroll

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

def midi2samples(mid, ticks_sample):
    multitrack = pypianoroll.read(mid).binarize()
    pitchs = np.asarray([i.pianoroll for i in multitrack.tracks])
    pitchs = pitchs.reshape(pitchs.shape[0] * pitchs.shape[1], 128)
    pitchs = pitchs[:, 24:96]

    all_samples = np.asarray(
        [
            pitchs[i * ticks_sample : i * ticks_sample + ticks_sample]
            for i in range(int(pitchs.shape[0] / ticks_sample))
        ],
        dtype=np.uint8,
    )

    return all_samples

def save_samples(dataset, samples_measure):
    samples = []

    for dirpath, _, filenames in os.walk(dataset):
        for File in filenames:
            path = os.path.join(dirpath, File)

            pianoroll = midi2samples(path, 512)

            for i in range(int(pianoroll.shape[0] / samples_measure)):
                samples.append(
                    pianoroll[
                        i * samples_measure : i * samples_measure + samples_measure
                    ]
                )
    np.save("pypianorollSamples.npy", np.array(samples, dtype=np.uint8))

def sample2midi(path, sample, resolution):
    music = sample.reshape(2048,72)
    
    all_notes = np.zeros((2048,128), dtype=np.uint8)
    all_notes[:, 24:96] = music

    pypianoroll.write(
        path=path, 
        multitrack=pypianoroll.Multitrack(
            resolution=resolution,
            tracks=[
                pypianoroll.BinaryTrack(
                    program=0, is_drum=False, pianoroll=all_notes
                    )
                ]
            )
        )

def erase_notes(sample, erase_pctg, num_notes):
    erased_music = sample.copy()
    num_played_notes = len(np.where(erased_music == 1)[0])
    all_notes = []
    mask = np.zeros((sample.shape[0] * sample.shape[1], num_notes))

    for i, l in zip(np.where(erased_music == 1)[0], np.where(erased_music == 1)[1]):
        all_notes.append((i,l))

    for i in range(int(num_played_notes*erase_pctg)):
        position = random.choice(all_notes)
        lista = list(np.where(sample[:,position[1]] == 1)[0])
        note_init = note_end = lista.index(position[0])

        try:
            while lista[note_init] - lista[note_init - 1] == 1:
                note_init = note_init - 1
        except:
            pass
        
        try:
            while lista[note_end + 1] - lista[note_end] == 1:
                note_end = note_end + 1
        except:
            pass
        
        erased_music[lista[note_init]:lista[note_end],position[1]] = 0
        mask[lista[note_init]:lista[note_end],position[1]] = 1

    return erased_music.swapaxes(1,0), mask.swapaxes(1,0)