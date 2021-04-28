# Music patterns

This project is able to download songs online, transform them into similarity matrices and computes statistics on the resulting images.

The code present here was used in the article [Similarity of structure in popular music](https://www.math.mcgill.ca/bcorsini/math/articles/similarity_of_structures_in_popular_music.pdf).

## Getting started

### Running the code

The library requirements for this code can be found in `requirements.txt`. To install them, run the following command line in the terminal:
```sh
pip install -r requirements.txt
```
Once this is done, or if the required libraries are already installed, run the following:
```sh
python main.py
```
This line will run the music pattern algorithm on the tablatures available in `data/tablatures/`.

If you want to run the code on the whole dataset and obtain the results from the article, run the following:
```sh
python main.py --tab_dir dataset/tablatures --song_file dataset/songs.json --colour blue --batch_size 500 --n_batch 10 --clusters_size_threshold 15 --n_neighbours 20 --min_feat_size 5 --min_clus_size 5
```
__Careful__, it will take about XXX hours for the whole algorithm to be done.

### Organization of the code

The code of this project is able to execute several tasks:
* It can scroll through [Ultimate Guitar](https://www.ultimate-guitar.com/) and download the songs specified (when available). The code for this task can be found in `scroller.py`.
* It can transform songs in _GuitarPro_ format into pattern matrices and images. The code for this task uses `song.py`, which transforms a song into a pattern matrix, and `patterns.py`, which saves these pattern matrices and transform them into images.
* It can compute the distance between songs using their pattern matrices. The code for this task can be found in `measures.py`.
* It can use the distance between songs to group them into either clusters or neighbourhoods. The code for this task can be found in `grouping.py`.
* It can find relation between pattern structures and features. The code for this task can be found in `statistics.py`

On top of these files, `utils.py` contains a few useful functions used in different places, and `main.py` combines all this code.

### Running the different files

On top of `main.py` which combines all algorithms together, each file ending with `.py` (apart from `utils.py` and `song.py`) can be run independently. For example, if you are only interested in transforming your favorite tablatures into their corresponding images, you can place them in a new folder `my_tablatures/` and then run the following command line:
```sh
python patterns.py --tab_dir my_tablatures --im_dir my_images --colour blue
```
This will transforms the songs in `my_tablatures/` into images and save these images in `my_images/`. The parameter `--colour` can be used to choose the colour of the images and the set of choices can be found in `patterns.py`.

### Website scroller

The code in `scroller.py` is used to scroll through [Ultimate Guitar](https://www.ultimate-guitar.com/) and to download the corresponding files. This code requires you to set up a web driver for Chrome. To download the Chrome driver, go to [this webpage](https://chromedriver.chromium.org/downloads) and follow the instructions. Once you have downloaded the chromedriver, you can either put it in the main directory, or specify its path by using the argument `--chromedriver` in `scroller.py`.

__Disclaimer__: the code in `scroller.py` is very dependent on the architecture of the website it scrolls through. It might not be up-to-date with the current organization of the website and might need to be slightly modified. If it does not work, you can also download the tablatures from the website and put them into your tablature folder (`data/tablatures` by default).

## Results

This project produces two main types of results: image representation of songs, and statistics on a set of songs.

### Representation of songs

This project transforms songs into corresponding _pattern similarity matrices_.

### Statistics on patterns of songs

Using the previously represented matrices, this project then studies statistical properties of the songs based on their pattern structures.

## Contact and information

If you have any questions regarding the code, feel free to contact me at <benoitcorsini@gmail.com>.

If you found this code useful or used it for your own study, please cite the following paper:
...