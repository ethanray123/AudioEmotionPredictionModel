MIREX-like Mood Dataset - 2013/07
http://mir.dei.uc.pt


Dataset following the MIREX Mood Classification dataset.
More details at the website.

Important notes:
- The dataset consists of:
 *) 903 30 second clips. The files are organized in five
    folders (clusters) and subfolders
(representing their labels/subcategories).
 *) 764 lyric files in txt format
 *) 196 midi files

- The three formats follow the same file naming scheme:
 *) Eg., file 004.mp3, 004.txt and 004.mid correspond to
the same song.

- Due to size and bandwidth constraints the files are in
mp3, they should be converted to the same format used in
MIREX:
22 KHz
Sample size: 16 bit
Number of channels: 1 (mono)
Encoding: WAV 

- 7 additional files are included representing:
 * clusters.txt - the cluster of each clip represented by
one line in dataset.txt.
 * categories.txt - similar to clusters.txt but using the
respective subcategories.
 * dataset info (csv & html) - information about each file
 * 3 split-by-categories(...) - scripts to move/sort files
by category.

