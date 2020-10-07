from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions

import argparse
import os
import os.path as osp
import shutil
import json
from time import time


# Main webpage properties
GLOBAL_DIV = 'js-page'
TABS_DIV = 'pZcWD'
SUB_TABS_DIV = '_3g0_K'
FULL_RATING_DIV = '_3v82_'
HALF_RATING_DIV = '_34xpF'
EMPTY_RATING_DIV = '_3YfNh'
LINK_CSS = 'a._2KJtL'

# Download page properties
FILE_TYPE_DIV = '_2EcLF'
BUTTON_PATH = '//form/div/button'


def link_sorter(artist_to_download, title_to_download, download_link):
    (artist, title, rating, nb_rating, hlink) = download_link
    is_good_artist = artist_to_download.lower() in artist.lower()
    is_good_title = title_to_download.lower() in title.lower()
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
           is_good_title,
           is_not_album,
           is_not_solo,
           is_not_intro,
           is_not_live,
           is_not_acoustic,
           rating,
           nb_rate,
           hlink)


class TabScroller(object):

    def __init__(self,
                 songs=[],
                 tab_dir='tablatures',
                 res_dir='results',
                 time_limit=20,
                 ids={}):
        self.songs = songs
        self.tab_dir = tab_dir
        self.res_dir = res_dir
        self.download_dir = osp.join(os.getcwd(), tab_dir)
        self.driver = Chrome()
        self.driver.maximize_window()
        self.time_limit = time_limit
        self.ids = ids
        self.outputs = {}

        if not osp.exists(self.tab_dir):
            os.makedirs(self.tab_dir)
        if not osp.exists(self.res_dir):
            os.makedirs(self.res_dir)

    def get_download_links(self, artist, title):
        search_song = artist.lower().replace(' ','+') + '+' + title.lower().replace(' ','+')
        song_url = 'https://www.ultimate-guitar.com/search.php?title={}&page=1&type=500'.format(search_song)
        self.driver.get(song_url)
        if not len(self.driver.find_elements_by_class_name(GLOBAL_DIV)):
            raise Exception('Empty page', song_url)

        tab_choices = self.driver.find_elements_by_class_name(TABS_DIV)
        download_links = []
        current_artist = ''

        for tab_choice in tab_choices:
            sub_tab_choices = tab_choice.find_elements_by_class_name(SUB_TABS_DIV)

            if len(sub_tab_choices) < 4:
                raise Exception('Error in the tab choices', song_url)

            else:
                sub_artist = sub_tab_choices[0].text
                sub_title = sub_tab_choices[1].text
                sub_nb_rating = sub_tab_choices[2].text
                sub_type = sub_tab_choices[3].text
                sub_rating_full = len(sub_tab_choices[2].find_elements_by_class_name(FULL_RATING_DIV))
                sub_rating_half = len(sub_tab_choices[2].find_elements_by_class_name(HALF_RATING_DIV))
                sub_rating_empty = len(sub_tab_choices[2].find_elements_by_class_name(EMPTY_RATING_DIV))
                sub_rating = sub_rating_full - 0.5*sub_rating_half - sub_rating_empty

                if sub_artist != 'ARTIST':
                    if sub_artist:
                        current_artist = sub_artist
                    if sub_type == 'Guitar Pro':
                        download_link = sub_tab_choices[1].find_elements_by_css_selector(LINK_CSS)
                        if not download_link:
                            raise Exception('No download link', song_url)
                        download_link = download_link[0].get_attribute('href')
                        download_links.append((current_artist, sub_title, sub_rating, sub_nb_rating, download_link))

        download_links = [link_sorter(artist, title, link) for link in download_links]
        download_links = [link_score[-1] for link_score in sorted(download_links, reverse=True)]

        return download_links

    def downloader(self, download_link, song_id):
        link_artist = ''
        for c in download_link.split('/')[-2].split('-'):
            link_artist += c.upper() + ' '
        link_title = ''
        for c in download_link.split('/')[-1].split('-')[:-3]:
            link_title += ' ' + c.capitalize()
        link_file = link_artist + '-' + link_title

        download_dir = os.listdir(self.tab_dir)
        not_downloaded = all([link_file.lower() not in file.lower() for file in download_dir])

        if not_downloaded:
            chrome_options = ChromeOptions()
            prefs = {'download.default_directory' : self.download_dir}
            chrome_options.add_experimental_option('prefs', prefs)
            download_driver = Chrome(options=chrome_options)
            download_driver.get(download_link)

            file_type = download_driver.find_elements_by_class_name(FILE_TYPE_DIV)
            right_type = False
            for element in file_type:
                if 'File format' in element.text:
                    if any(gp in element.text for gp in ['gp3', 'gp4', 'gp5']):
                        right_type = True

            if right_type:
                download_button = download_driver.find_elements_by_xpath(BUTTON_PATH)
                if not download_button:
                    raise Exception('No download button: {}'.format(download_link))
                download_button = download_button[0]
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
        errors = []
        for (artist, title) in self.songs:
            try:
                download_links = self.get_download_links(artist, title)
            except Exception as exc:
                self.outputs[artist + ' - ' + title] = 'EXCEPTION: ' + exc.args[0]
                download_links = []
            not_downloaded = True
            if (artist + ' - ' + title) in self.ids:
                song_id = ' (id=' + self.ids[artist + ' - ' + title] + ')'
            else:
                song_id = ''

            for download_link in download_links:
                if not_downloaded:
                    not_downloaded = self.downloader(download_link, song_id)

            if (artist + ' - ' + title) not in self.outputs:
                if not download_links:
                    self.outputs[artist + ' - ' + title] = 'No tablature to download'
                elif not_downloaded:
                    self.outputs[artist + ' - ' + title] = 'Tablature not downloaded'
                else:
                    self.outputs[artist + ' - ' + title] = 'Success!'

        self.driver.close()
        with open(osp.join(self.res_dir, 'songs.json'), 'w') as song_output:
            json.dump(self.outputs, song_output, indent=2)
            song_output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--song_file', type=str, default='songs.txt')
    parser.add_argument('--tab_dir', type=str, default='tablatures')
    parser.add_argument('--res_dir', type=str, default='results')
    parser.add_argument('--time_limit', type=float, default=20)
    parser.add_argument('--ids_file', type=str, default='songs_id.json')
    cmd = vars(parser.parse_args())

    songs = []
    with open(cmd['song_file'], 'r') as song_file:
        for line in song_file:
            artist, title = line.split('\n')[0].split('\t')
            songs.append((artist, title))

    if osp.exists(cmd['ids_file']):
        ids = json.load(open(cmd['ids_file']))
    else:
        ids = {}

    ts = TabScroller(
        songs=songs,
        tab_dir=cmd['tab_dir'],
        res_dir=cmd['res_dir'],
        time_limit=cmd['time_limit'],
        ids=ids
    )
    ts.download_and_close()
