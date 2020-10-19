import numpy as numpy

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
    It is defined as such to have clearer representation of songs and better results.
    '''
    processed = 0*pat_mat.copy()

    for p in range(1,101):
        processed += pat_mat >= np.percentile(pat_mat, p)

    processed = processed - np.min(processed)
    processed = processed / np.max(processed)

    return processed

def link_sorter(artist_to_download, title_to_download, link_info):
    '''
    This function takes the info of a link for a tab and returns a number of information used to rank this tab.
    The tabs are going to be sorted as such:
    - First the tab with the right artist.
    - Then the tabs with the right title.
    - Then the tabs not corresponding to a full album.
    - Then the tabs not being only the solo of the song.
    - Then the tabs not being only the intro of the song.
    - Then the tabs not being a live version of the song.
    - Then the tabs not being an acoustic version of the song.
    Once the tabs are ordered according to these conditions, we order the tabs according to their rating and number of ratings.
    The last output is just the link of the tab that we need to keep.
    '''
    (artist, title, rating, nb_rating, hlink) = link_info
    is_good_artist = artist_to_download.lower() in artist.lower()
    is_good_title = title_to_download.lower() in title.lower()
    is_not_album = 'Album' not in title
    is_not_solo = 'Solo' not in title
    is_not_intro = 'Intro' not in title
    is_not_live = 'Live' not in title
    is_not_acoustic = 'Acoustic' not in title
    if nb_rating == '':
        nb_rate = 0
    else:
        nb_rate = int(nb_rating)
    return(is_good_artist,
           is_good_title,
           is_not_album,
           is_not_solo,
           is_not_intro,
           is_not_live,
           is_not_acoustic,
           rating,
           nb_rate,
           hlink)
