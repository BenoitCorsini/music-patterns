import numpy as np

TO_DELETE = ['/', '\\', '\'', '\"', '.', ',', '!', '?', '(', ')']
TO_REPLACE = ['-', ' ', ':']

TO_SPLIT_ARTIST = [' and ', ' & ', ' featuring ', ', ', ' (', ' with ', ' + ', ' X ', '/', ' feat ', ' or ']
TO_DELETE_ARTIST = [')', '\'', '\"', '-', '.', '\\']
TO_SPLIT_TITLE = [' (', '/', '\\', ' [']
TO_DELETE_TITLE = [')', '\'', '\"', '.', '?', '!', '-', '*', ']']


years_dict = {}
with open('Hot Stuff.csv', 'r') as hot_stuff:
	is_first = True
	for line in hot_stuff:
		if is_first:
			is_first = False
		else:
			split = line.split(',')
			title = split[3]
			artist = split[4]
			year = split[1].split('/')[-1]
			if (artist, title) in years_dict.keys():
				years_dict[artist, title] = min(years_dict[artist, title], int(year))
			else:
				years_dict[artist, title] = int(year)
	hot_stuff.close()

songs = ''
for (artist, title) in years_dict.keys():

	simple_artist = artist.lower()
	for delete_char in TO_DELETE:
		simple_artist = simple_artist.replace(delete_char, '')
	for replace_char in TO_REPLACE:
		simple_artist = simple_artist.replace(replace_char, '_')

	simple_title = title.lower()
	for delete_char in TO_DELETE:
		simple_title = simple_title.replace(delete_char, '')
	for replace_char in TO_REPLACE:
		simple_title = simple_title.replace(replace_char, '_')

	song_id = simple_artist + '__&&__' + simple_title + '__&&__' + str(years_dict[artist, title])

	artists = artist.lower()
	for delete_char in TO_DELETE_ARTIST:
		artists = artists.replace(delete_char, '')
	artists = [artists] * 2
	for split_char in TO_SPLIT_ARTIST:
		sub_artists = artists[1:]
		artists = [artists[0]]
		for sub_artist in sub_artists:
			artists += sub_artist.split(split_char)
	artists = [sub_artist.replace('(', '').replace(' & ', ' ') for sub_artist in artists if len(sub_artist) != 0]
	if len(artists) == 2:
		artists = artists[:1]

	titles = title.lower()
	for delete_char in TO_DELETE_TITLE:
		titles = titles.replace(delete_char, '')
	titles = [titles] * 2
	for split_char in TO_SPLIT_TITLE:
		sub_titles = titles[1:]
		titles = [titles[0]]
		for sub_title in sub_titles:
			titles += sub_title.split(split_char)
	titles = [sub_title.replace('(', '').replace('[', '') for sub_title in titles if len(sub_title) != 0]
	if len(titles) == 2:
		titles = titles[:1]

	for sub_artist in artists:
		for sub_title in titles:
			songs += sub_artist + '\t' + sub_title + '\t' + song_id + '\n'

with open('songs.txt', 'w') as songs_txt:
	songs_txt.write(songs)
	songs_txt.close()
