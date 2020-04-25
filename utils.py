import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt



def list_all_tabs(tab_path):
    
    list_tabs = [(tab_path, x) for x in os.listdir(tab_path)]
    
    for (file_path, tab_name) in list_tabs:
        if osp.isdir(osp.join(file_path, tab_name)):
            list_tabs += [(osp.join(file_path, tab_name), x) for x in os.listdir(osp.join(file_path, tab_name))]
    
    list_tabs = [(x,y) for (x,y) in list_tabs if osp.isfile(osp.join(x, y))]
    
    return list_tabs



def save_image(pat_mat, song_name, im_path, overwrite_im):
    
    if overwrite_im or (not osp.exists(osp.join(im_path, song_name + '.png'))):
    
        (n1,n2) = np.shape(pat_mat)
        color_mat = np.zeros((n1,n2,3))
        
        processed = process(pat_mat)**.5
        
        color_mat[:,:,0] = 2*(0.5 - processed)*(processed < 0.5)
        color_mat[:,:,1] = 1 - processed
        color_mat[:,:,2] = 1 - 2*(processed - 0.5)*(processed > 0.5)
        
        if not osp.exists(im_path):
            os.makedirs(im_path)
        
        plt.figure(figsize=(5,5))
        plt.imshow(color_mat, interpolation='nearest')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.savefig(osp.join(im_path, song_name + '.png'), dpi=2*n1)
        plt.close()



def process(pat_mat):
    
    processed = 0*pat_mat.copy()
    
    for p in range(1,101):
        processed += pat_mat >= np.percentile(pat_mat, p)
    
    processed = processed - np.min(processed)
    processed /= np.max(processed)
    
    return processed



def matrix_distance(normed_mat1, normed_mat2):
    
    return 100*np.mean(np.abs(normed_mat1 - normed_mat2))


def link_sorter(artist_to_download, song_to_download, download_link):
    (artist, song, rating, nb_rating, hlink) = download_link
    is_good_song = (artist_to_download in artist.lower()) & (song_to_download in song.lower())
    is_not_album = 'Album' not in song
    is_not_solo = 'Solo' not in song
    is_not_intro = 'Intro' not in song
    is_not_live = 'Live' not in song
    is_not_acoustic = 'Acoustic' not in song
    if nb_rating == u'':
        nb_r = 0
    else:
        nb_r = int(nb_rating)
    return(is_good_song,
           is_not_album,
           is_not_solo,
           is_not_intro,
           is_not_live,
           is_not_acoustic,
           rating,
           nb_r,
           hlink)



def start():
    
    start = '#'*59 + '\n'
    
    start += '#'*5 + '  '
    start += 'START COMPUTING THE \'MUSIC PATTERN\' ALGORITHM'
    start += '  ' + '#'*5 + '\n'
    
    start += '#'*5 + ' '*5 + '-'*39 + ' '*5 + '#'*5 + '\n'
    
    start += '#'*5 + ' '*11
    
    start += '\033[1;30;42mDo\033[0;38;40m  '
    start += '\033[1;31;43mRe\033[0;38;40m  '
    start += '\033[1;32;44mMi\033[0;38;40m  '
    start += '\033[1;33;45mFa\033[0;38;40m  '
    start += '\033[1;34;46mSol\033[0;38;40m  '
    start += '\033[1;35;40mLa\033[0;38;40m  '
    start += '\033[1;36;41mSi\033[0;38;40m'
    
    start += ' '*11 + '#'*5 + '\n'
    
    start += '#'*59
    
    print(start)
