import os
import os.path as osp
from time import time

from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions

from utils import link_sorter



class scroller:
    
    
    
    def __init__(self,
                 songs=[('Tool', 'Lateralus'), ('Nirvana', 'Smells Like Teen Spirit')],
                 tab_dir='data\\tablatures',
                 driver_file='chromedriver.exe',
                 time_limit=10):
        
        self.songs = [(x.lower(),y.lower()) for (x,y) in songs]
        self.tab_dir = tab_dir
        self.download_dir = osp.join(os.getcwd(), tab_dir)
        self.driver_file = driver_file
        self.driver = Chrome(driver_file)
        self.time_limit = time_limit
        
        if not osp.exists(tab_dir):
            os.makedirs(tab_dir)
    
    
    
    def get_download_links(self, artist, song):
        
        self.driver.maximize_window()
        
        search_title = artist.replace(' ','+') + '+' + song.replace(' ','+')
        song_url = 'https://www.ultimate-guitar.com/search.php?title={}&page=1&type=500'.format(search_title)
        self.driver.get(song_url)
        
        tab_choices = self.driver.find_elements_by_class_name('pZcWD')
        
        download_links = []
        current_band = u''
        
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
                    
                    if sub_band != u'':
                        current_band = sub_band
                    
                    if sub_type == u'Guitar Pro':
                        
                        download_link = sub_tab_choices[1].find_element_by_css_selector('a._2KJtL').get_attribute('href')
                        download_links.append((current_band, sub_song, sub_rating, sub_nb_rating, download_link))
            
        download_links = [link_sorter(artist, song, x) for x in download_links]
        download_links = [x[-1] for x in sorted(download_links, reverse=True)]
        
        return download_links
    
    
    
    def downloader(self, download_link):
        
        link_artist = ''
        for c in download_link.split(u'/')[-2].split('-'):
            link_artist += c.capitalize() + ' '
        
        link_song = ''
        for c in download_link.split(u'/')[-1].split('-')[:-3]:
            link_song += ' ' + c.capitalize()
        
        link_file = link_artist + '-' + link_song
        
        download_dir = os.listdir(self.tab_dir)
        downloaded = any([(link_file in x) for x in download_dir])
        
        if not downloaded:
            
            #chrome_options = ChromeOptions()
            #prefs = {'download.default_directory' : self.download_dir}
            #chrome_options.add_experimental_option('prefs', prefs)
            #download_driver = Chrome(executable_path=self.driver_file, chrome_options=chrome_options)
            download_driver = Chrome(self.driver_file)
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
                
                download_complete = False
                start_time = time()
                
                while not download_complete:
                    download_dir = os.listdir(self.tab_dir)
                    downloaded = [x for x in download_dir if link_file in x]
                    if len(downloaded) > 0:
                        downloaded = all([(not x.endswith('.crdownload')) for x in downloaded])
                    else:
                        downloaded = False
                    download_complete = downloaded or (time() > start_time + self.time_limit)
            
            download_driver.close()
        
        return download_complete #downloaded
    
    
    
    def scroll_songs(self):
        
        for (artist, song) in self.songs:
            download_links = self.get_download_links(artist, song)
            
            song_downloaded = False
            
            for download_link in download_links:
                if not song_downloaded:
                    song_downloaded = self.downloader(download_link)
        
        self.driver.close()

scroller().scroll_songs()
