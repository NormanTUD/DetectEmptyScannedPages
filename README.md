# DetectEmptyScannedPages

Keras NN that detects whether scanned pages are empty or not

## Problem description

I have a scanner that scans 2 sided but does not determine whether the two-sided
scans are empty or not. 

This script was trained on about 600 images of pages that were scanned,
roughly half of which were empty, and the other half was not.


# How it works

Call it like this:

```
python3 is_empty_scan.py path/to/img/file.jpg
```

It will print out "Nonempty" or "Empty" and exit with 0 for
empty files and 1 for non-empty files.

This way you can use this script in a bash-script:

```
python3 is_empty_scan.py file.jpg && rm file.jpg
```

Although I personally recommend NOT deleting the files directly,
but moving them in a seperate directory to check before deleting
them manually. I am in no way responsible if you delete stuff that
you regret deleting afterwards!
