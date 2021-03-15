import json
import random
import numpy as np


def file_to_songs(file_name):
    '''
    This function takes a file and outputs a list of songs and a dictionary of ids.
    The list of songs should be of the form (artist, title).
    The dictionary should be of the form 'artist - title' : id.
    '''
    songs_dict = json.load(open(file_name, 'r'))
    songs = []
    ids = {}
    for song_info in songs_dict.values():
        artist = song_info.get('artist', '').lower()
        title = song_info.get('title', '').lower()
        songs.append((artist, title))
        if 'id' in song_info:
            ids[artist + ' - ' + title] = song_info['id']

    return songs, ids

def time_to_string(time):
    '''
    This function takes a time and transform it into a string.
    '''
    hours = int(time/3600)
    minutes = int((time - 3600*hours)/60)
    seconds = int(time - 3600*hours - 60*minutes)
    if hours:
        return '{}h{}m{}s'.format(hours, minutes, seconds)
    elif minutes:
        return '{}m{}s'.format(minutes, seconds)
    else:
        return '{}s'.format(seconds)

def process(pat_mat):
    '''
    This function processes a pattern matrix.
    '''
    processed = 0*pat_mat.copy()

    for p in range(1,101):
        processed += pat_mat >= np.percentile(pat_mat, p)

    processed = processed - np.min(processed)
    processed = processed / np.max(processed)

    return processed

def basic_stats():
    '''
    This function defines the basic statistics to be computed.
    For each type (feature, cluster, neighbour), it contains a list of stats to be plotted.
    The quadruplets in the list correspond to (feature, ordering, reverse, bars).
    '''
    basic_stats = {
        'feature' : [
            ('artist', 'mean', False, 'size'),
            ('year', 'mean', False, 'size'),
            ('decade', 'mean', False, 'size'),
            ('genre', 'mean', False, 'size'),
            ('types', 'mean', False, 'size')
        ],
        'cluster' : [
            ('artist', 'mean', False, 'size'),
            ('artist', 'std', False, 'size'),
            ('year', 'std', False, 'size'),
            ('decade', 'std', False, 'size'),
            ('genre', 'mean', False, 'size'),
            ('genre', 'std', False, 'size'),
            ('types', 'mean', False, 'size'),
            ('types', 'std', False, 'size')
        ],
        'neighbour' : [
            ('artist', 'mean', False, 'dist_mean'),
            ('artist', 'std', False, 'dist_mean'),
            ('year', 'std', False, 'dist_mean'),
            ('decade', 'std', False, 'dist_mean'),
            ('genre', 'mean', False, 'dist_mean'),
            ('genre', 'std', False, 'dist_mean'),
            ('types', 'mean', False, 'dist_mean'),
            ('types', 'std', False, 'dist_mean')
        ]
    }

    return basic_stats


#Some motifs and patterns, for esthetic purposes.
MOT1 = '\033[1;37;44m#\033[0;38;40m'
MOT2 = '\033[1;37;43m#\033[0;38;40m'
MOT3 = '\033[1;37;41m#\033[0;38;40m'
MOT4 = '\033[1;37;42m#\033[0;38;40m'
PAT1 = '\033[1;37;44m#\033[1;37;43m#\033[1;37;41m#\033[1;37;42m#\033[0;38;40m'
PAT2 = '\033[1;37;43m#\033[1;37;41m#\033[1;37;42m#\033[1;37;44m#\033[0;38;40m'
PAT3 = '\033[1;37;41m#\033[1;37;42m#\033[1;37;44m#\033[1;37;43m#\033[0;38;40m'
PAT4 = '\033[1;37;42m#\033[1;37;44m#\033[1;37;43m#\033[1;37;41m#\033[0;38;40m'
REP = 21

def intro():
    '''
    This function simply prints the headline of the algorithm
    '''
    print()
    print(PAT1*REP + MOT1)
    print(PAT2*REP + MOT2)
    print(PAT3 + MOT3 + '\033[1;38;40m                        _                    _   _                         ' + PAT3 + MOT3)
    print(PAT4 + MOT4 + '\033[1;38;40m    _ __ ___  _   _ ___(_) ___   _ __   __ _| |_| |_ ___ _ __ _ __  ___    ' + PAT4 + MOT4)
    print(PAT1 + MOT1 + '\033[1;38;40m   | \'_ ` _ \\| | | / __| |/ __| | \'_ \\ / _` | __| __/ _ \\ \'__| \'_ \\/ __|   ' + PAT1 + MOT1)
    print(PAT2 + MOT2 + '\033[1;38;40m   | | | | | | |_| \\__ \\ | (__  | |_) | (_| | |_| ||  __/ |  | | | \\__ \\   ' + PAT2 + MOT2)
    print(PAT3 + MOT3 + '\033[1;38;40m   |_| |_| |_|\\__,_|___/_|\\___| | .__/ \\__,_|\\__|\\__\\___|_|  |_| |_|___/   ' + PAT3 + MOT3)
    print(PAT4 + MOT4 + '\033[1;38;40m                                |_|                                        ' + PAT4 + MOT4)
    print(PAT1 + MOT1 + '\033[1;38;40m                                                                           ' + PAT1 + MOT1)
    print(PAT2*REP + MOT2)
    print(PAT3*REP + MOT3)
    print()

def chorus():
    '''
    This function prints a line between different algorithms.
    '''
    r = random.random()
    if r < .25:
        print()
        print(PAT1*REP + MOT1)
        print()
    elif r < .5:
        print()
        print(PAT2*REP + MOT2)
        print()
    elif r < .75:
        print()
        print(PAT3*REP + MOT3)
        print()
    else:
        print()
        print(PAT4*REP + MOT4)
        print()

def outro(time):
    '''
    This function prints the final state of the algorithm.
    '''
    print('MUSIC PATTERNS EXECUTED IN {}'.format(time_to_string(time)))
    print()
    print(PAT4*REP + MOT4)
    print(PAT1*REP + MOT1)
    print()