import numpy as np
import math
import guitarpro


class Track(object):

    def __init__(self, track):
        self.track = track
        self.is_drum_track = self.track.isPercussionTrack
        
        self.__get_track_information__()

        self.num_measures = len(self.measure_list)
        self.max_measure_length = self.max_note_denominator*self.max_signature_ratio

    def __get_track_information__(self):
    	self.max_note_denominator = 1
    	self.max_signature_ratio = 1
    	self.measure_list = []
    	self.upper_note = 0

    	repeat = []
    	last_repeat = []

    	for measure in self.track.measures:
    		ts = measure.timeSignature
    		self.max_signature_ratio = max(self.max_signature_ratio, ts.numerator/ts.denominator.value)
    		self.measure_list += [measure]

    		if measure.isRepeatOpen:
    			repeat = []
    			last_repeat = []
    		repeat += [measure]
    		if measure.header.repeatAlternative == 0:
    			last_repeat += [measure]

    		num_repetition = measure.repeatClose
    		if num_repetition > -1:
    			self.measure_list += repeat*(num_repetition-2) + last_repeat

    		for voice in measure.voices:
    			for beat in voice.beats:
    				bd = beat.duration
    				self.max_note_denominator = max(self.max_note_denominator, bd.value*(1+1*bd.isDotted)*bd.tuplet.enters)
    				for note in beat.notes:
    					self.upper_note = max(self.upper_note, note.value+2)

    	self.max_signature_ratio = math.ceil(self.max_signature_ratio)

    def to_numpy(self):
    	numpy_track = np.zeros((self.max_measure_length, self.num_measures))

    	for (measure_index, measure) in enumerate(self.measure_list):
    		note_time = 0
    		ts = measure.timeSignature
    		measure_duration = ts.numerator*self.max_note_denominator/ts.denominator.value
    		for voice in measure.voices:
    			for beat in voice.beats:
    				bd = beat.duration
    				note_duration = self.max_note_denominator/(bd.value)*(1+0.5*bd.isDotted)*bd.tuplet.times/bd.tuplet.enters
    				note_value = 0

    				if self.is_drum_track:
    					ordered_note = []
    					for note in beat.notes:
    						ordered_note.append(note.value)
    					ordered_note.sort()
    					for note in ordered_note:
    						note_value = note_value*self.upper_note + (note+1)

    				else:
    					for note in beat.notes:
    						note_value += (note.value+1)*self.upper_note**note.string

    				if note_time < measure_duration:
    					numpy_track[note_time,measure_index] = note_value
    					note_time += int(note_duration)

    	empty_times = np.any(numpy_track > 0, axis=1)

    	return numpy_track[empty_times,:]


class Song(object):

	def __init__(self, song):
		self.song = song
		self.__get_reduced_numpy_list__()
		self.limit_size = 1e6

	def __get_reduced_numpy_list__(self):
		self.numpy_list = []
		is_first = True
		for track in self.song.tracks:
			tr = Track(track)
			numpy_track = tr.to_numpy()
			if is_first:
				is_first = False
				measure_count = np.sum(numpy_track > 0, axis=0)
			else:
				measure_count += np.sum(numpy_track > 0, axis=0)
			self.numpy_list.append(numpy_track)
		count_start = np.cumsum(measure_count > 0)
		count_end = np.cumsum(measure_count[::-1] > 0)[::-1]
		not_empty = count_start*count_end > 0
		self.numpy_list = [numpy_track[:,not_empty] for numpy_track in self.numpy_list]

		self.num_measures = tr.num_measures

	def pattern_matrix(self):
		is_first = True
		for numpy_track in self.numpy_list:
			(n_notes, n_measures) = np.shape(numpy_track)

			if n_notes*n_measures**2 <= self.limit_size:
				reshaped_1 = np.reshape(numpy_track, (n_notes, n_measures, 1))
				reshaped_1 = np.repeat(reshaped_1, n_measures, axis=2)
				reshaped_2 = np.reshape(numpy_track, (n_notes, 1, n_measures))
				reshaped_2 = np.repeat(reshaped_2, n_measures, axis=1)
				are_different = 1*(reshaped_1 != reshaped_2)
				non_zero_1 = 1*(reshaped_1 > 0)
				non_zero_2 = 1*(reshaped_2 > 0)
				diff = np.sum(are_different*(non_zero_1 + non_zero_2), axis=0)

			else:
				diff = np.zeros((n_measures, n_measures), dtype=int)
				for i in range(n_measures):
					for j in range(n_measures):
						are_different = 1*(numpy_track[:,i] != numpy_track[:,j])
						non_zero_1 = 1*(numpy_track[:,i] > 0)
						non_zero_2 = 1*(numpy_track[:,j] > 0)
						diff[i,j] = np.sum(are_different*(non_zero_1 + non_zero_2))

			if is_first:
				is_first = False
				pat_mat = diff
			else:
				pat_mat += diff

		return pat_mat
