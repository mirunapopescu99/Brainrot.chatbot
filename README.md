# Brainrot.chatbot


## BrainrotBot: A GenZ/Alpha Mood-Translating Music Chatbot

##Project Description
This project implements a chatbot that translates user moods into GenZ/Alpha slang and recommends music based on those moods.
Project Overview
BrainrotBot is designed to enhance user engagement by providing a fun and relatable music recommendation experience. The chatbot interprets user-inputted moods, translates them into slang terms popular among Gen Z and Alpha generations, and then recommends songs with matching emotional tones from a curated dataset. This project utilizes various Natural Language Processing (NLP) techniques to achieve this functionality.  The core innovation lies in combining mood translation with music recommendation, creating a unique user experience.
Data Description
The project uses three primary datasets:
Song Dataset: A curated dataset of 125 songs, each manually annotated with a mood label (e.g., happy, sad, chill, angry, neutral).  The dataset includes columns for Song Title, Artist, Album, Lyrics, and Mood.
Slang Dataset: A dictionary containing Gen Z and Alpha slang terms mapped to corresponding moods.  This was compiled from various online sources, including TikTok and a GitHub repository Kaspercools. (2023) 'GenZ Dataset'. Available at https://github.com/kaspercools/genz-dataset/blob/main/genz_slang.csv (Accessed: 16/11/24)..
TikTok Data (Supplementary): Additional slang terms were collected from a TikTok video to enrich the slang datasetKaspercools. Louis Wong. (2024) Brainrot Gen Alpha Terms. 12 July. Available at: https://vm.tiktok.com/ZGdjrDKBD/ (Accessed: 20/11/24)


## Methodology
The chatbot's functionality is divided into these main components:
Mood Detection: User input is processed to detect the expressed mood using keyword detection and regular expressions.
Mood Translation: The detected mood is translated into randomly selected slang terms based on the slang dataset.
Music Recommendation: A song is selected from the curated song dataset, with its mood aligning with the user’s expressed mood (and slang translation).
NLP Techniques: The project employs several NLP techniques:
Sentiment Analysis: VADER sentiment analysis is used to verify the mood assigned to songs.
TF-IDF: Term Frequency-Inverse Document Frequency is used to vectorize song lyrics for enhanced similarity comparisons in mood detection and topic modeling.
Topic Modeling: Latent Dirichlet Allocation (LDA) is applied to identify themes in song lyrics, allowing for a richer understanding of song mood and improved recommendations.
Lemmatization: SpaCy library is used to reduce words to their root forms, improving the accuracy of mood detection.
Regular Expressions: Used in processing user input and generating responses.
Results
The chatbot was tested with four participants who found it helpful, friendly, and entertaining.  It accurately identified and translated moods like "happy" and "sad," but struggled with more nuanced moods like "hungry" or "lonely" due to limitations in the slang dataset. This highlights an area for future improvement.  


## Technologies Used
Python
NLTK (or other NLP libraries used)
SpaCy
TensorFlow/PyTorch (if used for sentiment analysis or topic modeling)
VADER
Pandas
Regular Expressions




## Future Work
Expand the slang dataset to include a wider range of moods and slang terms from diverse sources.
Implement a feedback loop allowing the chatbot to learn from user interactions.
Integrate additional NLP techniques such as contextual understanding to improve mood detection accuracy.
Explore using more sophisticated topic modeling techniques to refine mood classification.
