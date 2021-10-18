import numpy as np
import mido

TICKS_PER_BEAT = 32
BEAT_PER_COMPASS = 4
NUM_COMPASS = 4

def midi_to_pianoroll(mid):
    NUM_TICKS = TICKS_PER_BEAT * BEAT_PER_COMPASS * NUM_COMPASS

    track_num_ticks = mid.ticks_per_beat * BEAT_PER_COMPASS * NUM_COMPASS

    pianoroll = np.zeros((128, NUM_TICKS), dtype=np.uint8)
    state = np.zeros((128, 1), dtype=np.uint8)
    last_tick = 0
    current_tick = 0

    merged_track = mido.merge_tracks(mid.tracks)
    for msg in merged_track:
        new_tick = current_tick + msg.time
        if new_tick >= track_num_ticks:
            break
        current_tick = new_tick
        
        if msg.type not in ['note_on', 'note_off']:
            continue

        if current_tick != last_tick:
            pianoroll_last_tick = int(last_tick * TICKS_PER_BEAT /
                                      mid.ticks_per_beat)

            pianoroll_current_tick = int(current_tick * TICKS_PER_BEAT /
                                         mid.ticks_per_beat)

            pianoroll[:, pianoroll_last_tick:pianoroll_current_tick] = state

        last_tick = current_tick

        if msg.type == 'note_on' and msg.velocity > 0:
            state[msg.note] = msg.velocity
        elif msg.type == 'note_off':
            state[msg.note] = 0

    if current_tick < track_num_ticks:
        pianoroll_last_tick = int(last_tick * TICKS_PER_BEAT /
                                  mid.ticks_per_beat)
        pianoroll[:, pianoroll_last_tick:] = state

    return pianoroll

def pianoroll_to_midi(pianoroll):
    track = mido.MidiTrack()

    last_state = np.zeros(128, dtype=np.uint8)
    last_tick = 0
    for tick in range(0, pianoroll.shape[1]):
        current_state = pianoroll[:, tick]
        note_mismatches = current_state != last_state
        if not note_mismatches.any():
            continue
        changing_pos = np.where(note_mismatches)
        event_time = tick - last_tick
        last_tick = tick
        for note in changing_pos[0]:
            current_velocity = current_state[note]
            msg_type = 'note_off' if current_velocity == 0 else 'note_on'
            track.append(
                mido.Message(
                    msg_type,
                    note=note,
                    velocity=int(current_velocity),
                    time=16*event_time,
                ))

        last_state = current_state

    mid = mido.MidiFile()
    mid.tracks.append(track)
    return mid