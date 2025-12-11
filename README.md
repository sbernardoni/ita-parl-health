This is the code repository for "From Lockdown to Lexicon: How COVID-19 Changed Italy’s Health Parliamentary Discourse", a brief research project by Bernardoni Sara, Janssen Franz, Loreto Filippo, Monaco Gianmaria, Muriano Francesca for the course "20886 - Foundations of Social Sciences - Module I (Empirical Research Methods and Data Analysis)", at Bocconi University.

To replicate the results, please follow these steps:
1. Import the libraries from the ````requirements.py```` file. If they are not already installed on your machine, please run ````pip install```` followed by the name of the package in your terminal.
2. Download the _ItaParlCorpus_ data from http://doi.org/10.1017/ipo.2025.6, limited to the ````camera_2006-2022.csv```` file
3. Run the files in the folders, following the order in which they are presented in the paper. The order in which to run the files is specified in the file names. NB: Please remember to modify the file path included in the files to reflect the directory where you are working.
4. The file ````final_prep.py````, for the harmonisation of the subset of speeches obtained with BERTopic and LDA, relies on the output of the BERT model and LDA and therefore must be run last within that folder (i.e. after all the other ````.py```` files in the folder ````health speech identification````).

The file `utils.py` contains a helper function for saving plots used across multiple scripts in the `numeric analysis` folder. It is imported automatically; no separate execution is required.

The present repository includes the output of the robustness checks not shown in the paper. As mentioned there, they support the findings of the study, albeit with a less clear distribution.

For further information, questions or clarifications, please contact sara.bernardoni@studbocconi.it.
