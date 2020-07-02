from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions

import os
import os.path as osp
import shutil
from time import time


def link_sorter(artist_to_download, song_to_download, download_link):
    (artist, song, rating, nb_rating, hlink) = download_link
    is_good_artist = artist_to_download.lower() in artist.lower()
    is_good_song = song_to_download.lower() in song.lower()
    is_not_album = 'Album' not in song
    is_not_solo = 'Solo' not in song
    is_not_intro = 'Intro' not in song
    is_not_live = 'Live' not in song
    is_not_acoustic = 'Acoustic' not in song
    if nb_rating == '':
        nb_rate = 0
    else:
        nb_rate = int(nb_rating)
    return(is_good_artist,
    	     is_good_song,
           is_not_album,
           is_not_solo,
           is_not_intro,
           is_not_live,
           is_not_acoustic,
           rating,
           nb_rate,
           hlink)


class Scroller(object):

    def __init__(self,
                 songs=[],
                 tab_dir='data/tablatures',
                 time_limit=20,
                 ids={}):
        self.songs = songs
        self.tab_dir = tab_dir
        self.download_dir = osp.join(os.getcwd(), tab_dir)
        if not osp.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self.driver = Chrome()
        self.driver.maximize_window()
        self.time_limit = time_limit
        self.ids = ids

    def get_download_links(self, artist, song):
        search_title = artist.lower().replace(' ','+') + '+' + song.lower().replace(' ','+')
        song_url = 'https://www.ultimate-guitar.com/search.php?title={}&page=1&type=500'.format(search_title)
        self.driver.get(song_url)

        tab_choices = self.driver.find_elements_by_class_name('pZcWD')
        download_links = []
        current_band = ''

        for tab_choice in tab_choices:
            sub_tab_choices = tab_choice.find_elements_by_class_name('_3g0_K')

            if len(sub_tab_choices) < 4:
                print('ERROR')
                return []

            else:
                sub_band = sub_tab_choices[0].text
                sub_song = sub_tab_choices[1].text
                sub_nb_rating = sub_tab_choices[2].text
                sub_type = sub_tab_choices[3].text
                sub_rating_full = len(sub_tab_choices[2].find_elements_by_class_name('_3v82_'))
                sub_rating_half = len(sub_tab_choices[2].find_elements_by_class_name('_34xpF'))
                sub_rating_empty = len(sub_tab_choices[2].find_elements_by_class_name('_3YfNh'))
                sub_rating = sub_rating_full - 0.5*sub_rating_half - sub_rating_empty

                if sub_band != 'ARTIST':
                    if sub_band != '':
                        current_band = sub_band
                    if sub_type == 'Guitar Pro':
                        download_link = sub_tab_choices[1].find_element_by_css_selector('a._2KJtL').get_attribute('href')
                        download_links.append((current_band, sub_song, sub_rating, sub_nb_rating, download_link))

        download_links = [link_sorter(artist, song, link) for link in download_links]
        download_links = [link_score[-1] for link_score in sorted(download_links, reverse=True)]
        return download_links

    def downloader(self, download_link, song_id):
        link_artist = ''
        for c in download_link.split('/')[-2].split('-'):
            link_artist += c.upper() + ' '
        link_song = ''
        for c in download_link.split('/')[-1].split('-')[:-3]:
            link_song += ' ' + c.capitalize()
        link_file = link_artist + '-' + link_song

        download_dir = os.listdir(self.tab_dir)
        not_downloaded = all([link_file.lower() not in file.lower() for file in download_dir])

        if not_downloaded:
            chrome_options = ChromeOptions()
            prefs = {'download.default_directory' : self.download_dir}
            chrome_options.add_experimental_option('prefs', prefs)
            download_driver = Chrome(options=chrome_options)
            download_driver.get(download_link)

            file_type = download_driver.find_elements_by_class_name('_2EcLF')
            right_type = False
            for element in file_type:
                if 'File format' in element.text:
                    if any(gp in element.text for gp in ['gp3', 'gp4', 'gp5']):
                        right_type = True

            if right_type:
                download_button = download_driver.find_element_by_xpath('//form/div/button')
                download_button.click()

                start_time = time()
                while not_downloaded & (time() < start_time + self.time_limit):
                    download_dir = os.listdir(self.tab_dir)
                    not_downloaded = all([((link_file.lower() not in file.lower()) or (file.endswith('.crdownload'))) for file in download_dir])

            download_driver.close()

        if not not_downloaded:
            download_dir = os.listdir(self.tab_dir)
            download_dir.sort(key=lambda file : osp.getmtime(osp.join(self.tab_dir, file)), reverse=True)
            last_tab = download_dir[0]
            if (link_file.lower() in last_tab.lower()) & ('(id=' not in last_tab):
            	tab_type = last_tab[-4:]
            	os.rename(osp.join(self.tab_dir, last_tab), osp.join(self.tab_dir, link_file + song_id + tab_type))
            else:
            	last_good_tab = [file for file in download_dir if link_file.lower() in file.lower()][0]
            	tab_type = last_good_tab[-4:]
            	if last_good_tab != (link_file + song_id + tab_type):
            		shutil.copyfile(osp.join(self.tab_dir, last_good_tab), osp.join(self.tab_dir, link_file + song_id + tab_type))

        return not_downloaded

    def download_and_close(self):
        for (artist, song) in self.songs:
        	download_links = self.get_download_links(artist, song)
        	not_downloaded = True
        	if (artist, song) in self.ids:
	        	song_id = ' (id=' + self.ids[artist, song] + ')'
	        else:
	        	song_id = ''

       		for download_link in download_links:
       			if not_downloaded:
       				not_downloaded = self.downloader(download_link, song_id)
       	self.driver.close()


if __name__ == '__main__':
    TAB_DIR = 'data/tablatures'
    SCROLLER_DIR = 'scroller'

    songs = []
    for x in open(osp.join(SCROLLER_DIR, 'songs.txt')):
        artist, song = x.split('\n')[0].split('\t')
        songs.append((artist, song))

    scroll = Scroller(songs=songs, tab_dir=TAB_DIR)
    scroll.download_and_close()
