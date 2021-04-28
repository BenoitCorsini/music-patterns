import math
import numpy as np


class Track(object):

    def __init__(self, track):
        self.track = track
        self.is_drum_track = self.track.isPercussionTrack
        
        self.__get_track_information__()

        self.num_measures = len(self.measure_list)
        self.max_measure_length = self.max_note_denominator*self.max_signature_ratio

    def __get_track_information__(self):
        '''
        This function adds some important parameters to a 'Track' object:
        - 'max_note_denominator' which is the smallest length of a note.
        - 'max_signature_ratio' which corresponds to the longest possible number of notes in a measure.
        - 'measure_list' which simply creates a list of the measures in order, applying repetitions when relevant.
        - 'upper_note' which corresponds to the highest value of a note. 
        '''
        self.max_note_denominator = 1
        self.max_signature_ratio = 1
        self.measure_list = []
        self.upper_note = 0

        repeat = []
        last_repeat = []

        for measure in self.track.measures:
            ts = measure.timeSignature
            self.max_signature_ratio = max(self.max_signature_ratio, ts.numerator/ts.denominator.value)
            self.measure_list.append(measure)

            if measure.isRepeatOpen:
                repeat = []
                last_repeat = []
            repeat.append(measure)
            if measure.header.repeatAlternative == 0:
                last_repeat.append(measure)

            num_repetition = measure.repeatClose
            if num_repetition > -1:
                self.measure_list += repeat*(num_repetition - 2) + last_repeat

            for voice in measure.voices:
                for beat in voice.beats:
                    bd = beat.duration
                    self.max_note_denominator = max(self.max_note_denominator, bd.value*(1 + 1*bd.isDotted)*bd.tuplet.enters)
                    for note in beat.notes:
                        self.upper_note = max(self.upper_note, note.value + 2)

        self.max_signature_ratio = math.ceil(self.max_signature_ratio)

    def notes_to_int(self, notes):
        '''
        This function transforms a sequence of notes into an integer value.
        If the track corresponds to a drum, the value corresponds to the sets being played.
        OPtherwise, the value corresponds to the position at which notes are being played on a guitar.
        '''
        note_value = 0
        if self.is_drum_track:
            ordered_notes = []
            for note in notes:
                ordered_notes.append(note.value)
            ordered_notes.sort()
            for note in ordered_notes:
                note_value = note_value*self.upper_note + (note + 1)

        else:
            for note in notes:
                note_value += (note.value + 1)*self.upper_note**note.string

        return note_value

    def to_numpy(self):
        '''
        This functions transforms a track into a numpy array.
        The array has size (maximal length of a measure)x(number of measures).
        Each time of the measure contains a single number which represents the note or collection of notes.
        '''
        numpy_track = np.zeros((self.max_measure_length, self.num_measures))

        for measure_index, measure in enumerate(self.measure_list):
            note_time = 0
            ts = measure.timeSignature
            measure_duration = ts.numerator*self.max_note_denominator/ts.denominator.value
            note_was_played = False

            for voice in measure.voices:
                for beat in voice.beats:
                    bd = beat.duration
                    note_duration = self.max_note_denominator/(bd.value)*(1 + 0.5*bd.isDotted)*bd.tuplet.times/bd.tuplet.enters

                    if note_time < measure_duration:
                        note_value = self.notes_to_int(beat.notes)

                        # If the measure is empty, silence are ignored
                        if not note_value:
                            if note_was_played:
                                note_value = 1 # This is a silence
                        else:
                            note_value += 1
                            note_was_played = True

                        numpy_track[note_time, measure_index] = note_value
                        note_time += int(note_duration)

        # In the end, the times never used thoughout all measures are deleted.
        not_empty_times = np.any(numpy_track > 0, axis=1)

        return numpy_track[not_empty_times,:]


class Song(object):

    def __init__(self, song):
        self.song = song
        self.limit_size = 1e6 # This corresponds to the limiting size of a numpy array.

        self.__numpy_list__()

    def __numpy_list__(self):
        '''
        This function gets the list of numpy representation of the tracks.
        It then simplifies them if there are empty measures at the beginning or at the end.
        '''
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

        # 'measure_count' counts for each measure the number of non-empty tracks on this specific measure.
        count_start = np.cumsum(measure_count > 0)
        count_end = np.cumsum(measure_count[::-1] > 0)[::-1]
        not_empty = count_start*count_end > 0

        self.numpy_list = [numpy_track[:,not_empty] for numpy_track in self.numpy_list]

    def pattern_matrix(self):
        '''
        This function returns the pattern matrix of a song.
        This pattern matrix corresponds to the distance between the measures of the song. 
        '''
        is_first = True
        for numpy_track in self.numpy_list:
            n_notes, n_measures = np.shape(numpy_track)

            # This method is faster than the next one but requires to keep a large array in memory.
            if n_notes*n_measures**2 <= self.limit_size:
                reshaped_1 = np.reshape(numpy_track, (n_notes, n_measures, 1))
                reshaped_1 = np.repeat(reshaped_1, n_measures, axis=2)
                reshaped_2 = np.reshape(numpy_track, (n_notes, 1, n_measures))
                reshaped_2 = np.repeat(reshaped_2, n_measures, axis=1)

                are_different = (reshaped_1 != reshaped_2).astype(int)
                non_zero_1 = (reshaped_1 > 0).astype(int)
                non_zero_2 = (reshaped_2 > 0).astype(int)
                diff = np.sum(are_different*(non_zero_1 + non_zero_2), axis=0)

            else:
                diff = np.zeros((n_measures, n_measures), dtype=int)
                for i in range(n_measures):
                    for j in range(n_measures):
                        are_different = (numpy_track[:,i] != numpy_track[:,j]).astype(int)
                        non_zero_1 = (numpy_track[:,i] > 0).astype(int)
                        non_zero_2 = (numpy_track[:,j] > 0).astype(int)
                        diff[i,j] = np.sum(are_different*(non_zero_1 + non_zero_2))

            if is_first:
                is_first = False
                pat_mat = diff
            else:
                pat_mat += diff

        return pat_mat