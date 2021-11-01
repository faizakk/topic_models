# topic_models

pip install -r requirements.txt


This code generates topic modelling results for the text documents.
Current code is using tfidf and LDA model.
Saves the results in the csv file.
Genrates word clouds for each topic and saves them.

Usage:
    python3 main.py

Other files:
    doc_preprocessor.py: Loads text data from a csv files. Cleans and preprocesses the documents.
    
    vectorize_doc.py: vecotrizes(tfidf) the preprocessed documents.
    
    topic_model.py: Applies topic modelling to vectorized documents.
    
    visualization.py: Creates vitialization of topics (word clouds).
