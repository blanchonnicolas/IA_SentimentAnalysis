# IA_Project7_Openclassrooms_IA_SentimentAnalysis
Sentiment Analysis of Tweets, for AirParadis (Educational project)

## Air Paradis

Repository of OpenClassrooms project 7' [AI Engineer path](https://openclassrooms.com/fr/paths/188)

Goal : "Air Paradis" is an airline company who's marketing department wants to be able to detect quickly "bad buzz" on social networks, to be able to anticipate and address issues as fast as possible. They need an AI API that can detect "bad buzz" and predict the reason for it.
The goal here is to evaluate different approaches to detect "bad buzz" 

You can see the results here :

-   [Presentation](https://github.com/blanchonnicolas/IA_Project7_Openclassrooms_IA_SentimentAnalysis/blob/main/p7_04_presentation.pdf)

-   [Article de Blog](https://github.com/blanchonnicolas/IA_Project7_Openclassrooms_IA_SentimentAnalysis/blob/main/Blog.pdf)

-   [Vidéo](xx)

-   [Notebook 1 : Simple Models](https://github.com/blanchonnicolas/IA_Project7_Openclassrooms_IA_SentimentAnalysis/blob/main/IABadBuzz_AirParadis_Mod%C3%A8leSimple.ipynb)
    - NLP pre-processing (Bag Of Words, TF-IDF, Tokenization, Lemmatization, Stemmatisation, Pipeline)
    - Tweet preprocessing ([See Library](https://pypi.org/project/tweet-preprocessor/))
    - Supervised Machine Learning : SKLearn Logistic Regression, Decision Tree, Gradient Boosting, GridSearchCV, Threshold optmization
    - Confusion Matrix and Performance Metrics of classification : F1 score, Recall, Precision, Accuracy, ROC_AUC
    - Model Storage with Joblib

-   [Notebook 2 : Advanced Models](https://github.com/blanchonnicolas/IA_Project7_Openclassrooms_IA_SentimentAnalysis/blob/main/IABadBuzz_AirParadis_Mod%C3%A8leAvanc%C3%A9.ipynb)
    - Deep Learning and Neural Networks : Keras Tensorflow, CNN, Convolution, Max Pooling, LSTM
    - Transfer Learning : [Word Embeddings](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/), [Glove](https://nlp.stanford.edu/projects/glove/), [Fasttext](https://fasttext.cc/), [Word2Vec](https://fr.wikipedia.org/wiki/Word2vec)
    - Loss and Accuracy evolution : Epoch, re-train embeddings, Gradient descent and [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
 
-   [Notebook 3 : BERT Models](https://github.com/blanchonnicolas/IA_Project7_Openclassrooms_IA_SentimentAnalysis/blob/main/IABadBuzz_AirParadis_Mod%C3%A8leBERT.ipynb)
    - Deep Learning and [BERT Models](https://huggingface.co/docs/transformers/tasks/sequence_classification) : Datasets, Transformers, BERT, RoBERTa, CamemBERT, [Pipeline](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/pipelines#transformers.TextClassificationPipeline)
    - Fine Tuning and sequence classification: [See Blog](https://lesdieuxducode.com/blog/2019/4/bert--le-transformer-model-qui-sentraine-et-qui-represente), dataset, tokenizer, preprocess_function, data_collator, trainer, compute_metrics
    
-   [FastAPI Deployment - Prototype](https://github.com/blanchonnicolas/IA_Project7_Openclassrooms_IA_SentimentAnalysis/blob/main/IABadBuzz_AirParadis_FastAPI.ipynb)
    - [FastAPI Framework](https://fastapi.tiangolo.com/): uvicorn, Server, Swagger Interactive interface
    - Load pre-trained BERT Model or BERT Pipeline, Load Tokenizer
    - Get and Post Queries HTTP: Process Text and Predict sentiment
