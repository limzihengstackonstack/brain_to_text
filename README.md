# Data for: An accurate and rapidly calibrating speech neuroprosthesis

*The New England Journal of Medicine* (2024)

Nicholas S. Card, Maitreyee Wairagkar, Carrina Iacobacci, Xianda Hou, Tyler Singer-Clark, Francis R. Willett, Erin M. Kunz, Chaofei Fan, Maryam Vahdati Nia, Darrel R. Deo, Aparna Srinivasan, Eun Young Choi, Matthew F. Glasser, Leigh R. Hochberg, Jaimie M. Henderson, Kiarash Shahlaie, Sergey D. Stavisky*, and David M. Brandman*.

* "*" denotes co-senior authors

## Overview

This repository contains the data necessary to reproduce the results of the paper "*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), *N Eng J Med*.

The code is written in Python and is hosted on GitHub (link in the Related Works section).

The data can be downloaded from this Dryad repository. Please download this data and place it in the `data` directory of the GitHub code.

Data is currently limited to what is necessary to reproduce the results in the paper. We intend to share additional data, including neural data, in the coming months. All included data has been anonymized and does not include any identifiable information.

## Version 1 release files:

* `t15_copyTask.pkl`
  * Data from Copy Task trials during evaluation blocks (1,718 total trials) necessary for reproducing the online decoding performance plots (Figure 2).
  * Copy Task data includes, for each trial: cue sentence, decoded phonemes and words, trial duration, and RNN-predicted logits.
* `t15_personalUse.pkl`
  * Data from Conversation Mode (22,126 total sentences) necessary for reproducing Figure 4.
  * Conversation Mode data includes, for each trial: the number of decoded words, the sentence duration, and the participant's rating of how correct the decoded sentence was.
  * Specific decoded sentences from Conversation Mode are not included to protect the participant's privacy.