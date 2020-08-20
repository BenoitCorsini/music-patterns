from scroller import Scroller

NUM_SONGS = 1000
TAB_DIR = 'tablatures'

with open('songs.txt', 'r') as songs_txt:
	songs = songs_txt.readlines()
	songs_txt.close()

songs_to_download = songs[:NUM_SONGS]
songs_leftover = songs[NUM_SONGS:]

songs = []
ids = {}
for song in songs_to_download:
	(artist, title, song_id) = song.split('\n')[0].split('\t')
	songs.append((artist, title))
	ids[artist, title] = song_id

sc = Scroller(songs=songs, tab_dir=TAB_DIR, ids=ids)
songs_not_downloaded = sc.download_and_close()

songs = ''
for (artist, title) in songs_not_downloaded:
	songs += artist + '\t' + title + '\t' + ids[artist, title] + '\n'
for song_line in songs_leftover:
	songs += song_line

with open('songs.txt', 'w') as songs_txt:
	songs_txt.write(songs)
	songs_txt.close()
