# Music patterns

This project is able to download songs online, transform them into similarity matrices and computes statistics on the resulting images.

The code present here was used in the article [Similarity of structure in popular music](https://arxiv.org/abs/2007.13728).

If you are using this code, please cite this paper: ...

## Getting started

### Running the code

The library requirements for this code can be found in `requirements.txt`. To install them, run the following command line in the terminal:
```sh
pip install -r requirements.txt
```
Once this is done or if the required libraries are already installed, run the following:
```sh
python main.py
```
This line will run the music pattern algorithm on the tablatures available in `data`. It is possible to add

### Organization of the code

The code of this project is able to execute several tasks:
* It scrolls through [Ultimate Guitar](https://www.ultimate-guitar.com/) and downloads the songs specified (when available). The code for this task can be found `scroller.py`.
* It transforms songs from their _GuitarPro_ files into pattern matrices and images. The code for this task uses `song.py`, which transforms a song into a pattern matrix, and `patterns.py`, which saves these pattern matrices and transform them into images.
* It computes the distance between songs using their pattern matrices. The code for this task can be found in `measures.py`.
* It uses the distance between songs to group them into either clusters or neighbourhoods. The code for this task can be found in `grouping.py`.
* It represents the statistics of the relation between the features of the songs and their distance. The code for this task can be found in `statistics.py`

On top of these files, `utils.py` contains a few useful functions used in different places, and `main.py` combines all this code. `main.py` combines all possible parameters and is organized according to different parameters. For more details into the roles of the parameters, I invite to look into the corresponding file and the definition of the class.

### Running the different files

On top of `main.py` which combines all algorithms together, each file ending with `.py` (apart from `utils.py` and `song.py`) can be run independetly. For example, if you are only interested in transforming your favorite tablatures into their corresponding images, you can create a new folder `my_tablatures` in which you place them, and then run the following command line:
```sh
python patterns.py --tab_dir my_tablatures --im_dir my_images --colour my_colour
```
This will transforms the songs in `my_tablatures` into images and save these images in `my_images`. The parameter `my_colour` can be used to choose the colour of the images and the set of choices can be found in `patterns.py`.

### Website scroller

As explained earlier, the code in `scroller.py` is used to scroll though [Ultimate Guitar](https://www.ultimate-guitar.com/) and to download the corresponding files. This code requires you to set up a web driver for Chrome. To download the Chrome driver, go to [this](https://chromedriver.chromium.org/downloads) webpage and follows the instructions. Once you have downloaded the chromedriver, you can either put it in the main directory, or specify its path by using the argument `--chromedriver` in `scroller.py`.