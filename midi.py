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
    all_notes = np.stack([np.where(erased_music == 1)[0], np.where(erased_music == 1)[1]], axis=1)
    mask = np.zeros((erased_music.shape[0],num_notes))

    len_notes = 0
    start_note = 0

    dic = {}

    for i in range(1, len(all_notes[:,1])):
            if all_notes[:,1][i] != all_notes[:,1][i - 1]:
                    try: 
                            dic[all_notes[i-1][1]].append([start_note, all_notes[i][0]])
                    except:
                            dic[all_notes[i-1][1]] = []
                            dic[all_notes[i-1][1]].append([start_note, all_notes[i][0]])
                    start_note = all_notes[i][0]
                    len_notes+=1

    for i in range(int(len_notes * erase_pctg)):
            note = random.choice(list(dic.keys()))
            time = random.randint(0, len(dic[note]) - 1)
            erased_music[dic[note][time][0]: dic[note][time][1] - 1, :] = 0
            mask[dic[note][time][0]: dic[note][time][1] - 1, :] = 1
    
    return erased_music.swapaxes(1,0), mask.swapaxes(1,0)
