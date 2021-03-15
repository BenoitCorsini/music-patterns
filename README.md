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

### Organization of the code

The code of this project is able to execute several tasks:
* It scrolls through [Ultimate Guitar](https://www.ultimate-guitar.com/) and download the songs specified (when available). The code for this task can be found `scroller.py`.
* It transforms songs from their _GuitarPro_ files into pattern matrices and images. The code for this task uses `song.py`, which transforms a song into a pattern matrix, and `patterns.py`, which saves these pattern matrices and transform them into images.
* It computes the distance between songs using their pattern matrices. The code for this task can be found in `measures.py`.
* It uses the distance between songs to group them into either clusters or neighbourhoods. The code for this task can be found in `grouping.py`.
* It represents the statistics of the relation between the features of the songs and their distance. The code for this task can be found in `statistics.py`