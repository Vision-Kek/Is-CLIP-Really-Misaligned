* Notebooks that produce paper figures
* Figures themselves
* cvpr supplementary submission (code edited afterwards)


If you try to run the notebooks, you'll need to fix path errors that occured because of restructuring project.
Mainly:
* rename challenging -> paperfigs
* rename intramodalmisalignment -> experiments

Further, in feature caches paths:
* data/ is merely a symbolic link to local /mnt/hdd/datasets/CoOp. data/ is not actually contained in ICLR24 or Cross-the-Gap
* ICLR24 is a private repo forked from the lda baseline
* Cross-the-Gap is a private repo forked from the oti baseline
