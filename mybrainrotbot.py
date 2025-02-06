import csv
import os 
import re
import random as random
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer 
from chatbot_base import ChatbotBase 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Imports for local data retrieval (part 1)
import random 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity



"""
Retrieval based chatbot that translates your mood into generational brainrot and recommends song based off 
user's mood

""" 


class BrainrotBot(ChatbotBase):
    def __init__(self, name ='BrainrotBot', dataset_path='simplified_song_moods.csv'): 
        super().__init__(name) 
        self.dataset_path = dataset_path
        self.column_names = ['Song', 'Artist', 'Lyrics', 'Mood', 'Album']
        self.dataset_separator = ','

        # Initialize the VADER SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()

        self.nlp = spacy.load('en_core_web_sm') 

        
        self.df = self.load_lyrics(self.dataset_path, self.dataset_separator, self.column_names)
        
        
        if self.df is None or 'Mood' not in self.df.columns or self.df['Mood'].isnull().all():
            print("Error: 'Mood' column not found in dataset.") 
            exit(1) 

        self.vectorizer, self.lyrics_matrix = self.lyrics_to_tfidf(self.df)
        
        if self.vectorizer is None or self.lyrics_matrix is None or self.lyrics_matrix.shape[0] == 0:
            print("Error: TF-IDF vectorization failed or resulted in an empty matrix. Exiting.")
            exit(1) 

        self.songs_dict = self.load_songs_from_csv(self.dataset_path) 

        self.genz_brainrot_dict = self.initialize_brainrot_slang()

         # Call greeting method to show a greeting on initialization
        self.greeting() 


        self.perform_topic_modeling(num_topics=5)

    def detect_keywords(self, user_input):
        """Detect keywords in user input and return related mood."""
        # Define a list of keywords related to moods
        keywords = {
            'happy': ['happy', 'joy','enjoy', 'excited', 'cheerful', 'elated', 'good'],
            'romantic': ['romantic', 'in love', 'love', 'mushy', 'cute', 'kiss', 'hug'],
            'sad': ['sad', 'cry', 'down', 'blue', 'upset', 'depressed', 'low', 'moody', 'emotional', 'tearful'],
            'angry': ['angry', 'mad', 'crazy', 'frustrated', 'annoyed', 'pissed', 'scream', 'bad', 'evil', 'stressed'],
            'cool': ['chill', 'relax', 'calm', 'peaceful', 'peace', 'slow', 'tame', 'okay', 'neutral'],
            'confident': ['energetic', 'lively', 'excited', 'fire', 'positive', 'amazing', 'upbeat'],
            'silly': ['party','foolish', 'dumb', 'stupid', 'mindless', 'dummy', 'brainless'],
            'mysterious': ['magical', 'weird', 'strange', 'curious'],
            'confused': ['unsure', 'help', 'dunno', 'hmmm', 'thinking'],
            'surprised': ['shocked', 'amazed', 'astonished'],
            'funny': ['odd', 'cheesy'],
            'disappointed': ['beaten', 'shot-down', 'defeated'],
            'sleepy': ['tired', 'heavy', 'bed'],
            'crazy': ['party', 'high', 'intoxicated'],
            'creative': ['arty', 'create', 'inspiration'],
            'unwell': ['sick', 'unhealthy', 'droopy'],
            'quirky': ['wacky', 'different', 'unique', 'peculiar', 'wacky'],
            'introspective': ['reflective', 'thoughtful'],
            'rebellious': ['unruly', 'anarchistic','restless', 'rebel', 'difficult', 'turbulent'],


        }

        # Normalize the user input to lower case
        user_input = user_input.lower()

        # Check for keywords and return the corresponding mood
        for mood, words in keywords.items():
            if any(word in user_input for word in words):
                return mood  # Return the detected mood
        return None  # Return None if no keyword is found 


    def perform_topic_modeling(self, num_topics=5):
        """ Perform topic modeling on the song lyrics using LDA. """
        
        # Create a CountVectorizer for the lyrics to produce a document-term matrix
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2)) 
        doc_term_matrix = vectorizer.fit_transform(self.df['Lyrics'])

        # Create the LDA model
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix)

        # Print the topics found
        for idx, topic in enumerate(lda_model.components_):
            print(f"Topic {idx + 1}:")
            top_feature_indices = topic.argsort()[-10:]  # Get the top 10 words for this topic
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_feature_indices]
            print(" ".join(top_words))


    def initialize_brainrot_slang(self):
        """
        Initialize a dictionary with Gen Z slang mappings.
        """
        brainrot_dict = { 
            'happy': [
                 'W',                   # Shorthand for win
                 'Dank',
                 'This slaps',
                 'Straight gas',                
                 'Bop',                 
                 'Smol',                
                 'Bussin',            
                 'Sheesh',              
                 'Im weak',
                 '+ 10,000 aura points'             
            ],
            'sad': [
                  'What an L',
                  'Left on read',                   
                  'Ghosting',            
                  'Salty',
                  'Womp womp',
                  'Rip', 
                  'Big yikes',
                  'Its giving...'                
            ],
            'angry': [
                    'What an L',             
                    'Ffs',
                    'Ur cooked',
                    '@ me',
                    'This aint it',                
                    'Catch these hands',    
                    'Take several seats',
                    'Its giving...'             
            ],
            'rebellious': [
                'Thats cap',                 
                'Ur getting canceled',      
                'Finesse',
                'Lit',
                'Yes diva',                
                'Snack',
                'Its giving...'             
               
            ],
            'introspective': [
                'TFW',                 # that feeling when
                'Hits different',      
                'Rent free',           
                'Big yikes',
                'Its giving ...'            
                             
            ],
            'mysterious': [
            'E-boy',
            'Who is this diva?',               
            'E-girl',
            'Fierce',              
            'Sus',
            'Let him cook'                  
            ],
           'romantic': [
                'Simp',
                'Its giving situationship',
                'Okay soft launch',
                'Rizzler',
                'Gyatt',                
                'Snack',
                'Its giving talking stage'                
            ],
            'quirky': [
                'Who is the diva',
                '@ me',               
                'Spill the tea',             
                'Iykyk',
                'Cringe',
                'Its giving ...' 
            ],
            'cool': [
                'Lit',
                'Slay',
                'With rizz',
                'Left no crumbs',
                'Infinite rizz',
                'Are u mewing?',
                'Woke',
                'We stan',                
                'Valid' 
            ], 
            'silly': [
                'Skibidi',
                'The math is not mathing',
                'Delulu', 
                'What the sigma',
                'Are u from Ohio?',
                'Are u fanum taxing me?'               # when someone takes your food          
                        
            ],
            'unwell': [ 
                'Mid',
                'The math is not mathing',
                'Are u a pick-me?',
                '- 10,000 aura points',
                'Its giving ...'                  
            ],
            'creative': [
                'Beat your face',
                'Who is this diva?',
                'Looksmaxxing',
                'Who cooked in here?'
                'Its giving ...'  
            ], 
            'crazy':  [
                'Out of pocket',
                'How many aura points did u lose?',
                'You yapper',
                'Its giving...'
            ],
            'sleepy': [
                'Ok boomer',
                'Cringe',
                'Smh',
                'Its giving...'
            ],                
            'disappointed': [ 
                'Smh',
                'Pick-me girl', 
                'Cheugy',
                'Not very demure not very mindful', 
                'Its giving...'     
            ],
            'funny': [
             'Ded',
             'Do u do weddings?',
             'Very demure very mindful',
             'Its giving...'               
            ],
            'surprised': [
              'I oop',
              'Math is not mathing',
              'Shookth',
              'Its giving...' 
             ],
            'confident': [
                'Drip',              
                'Bet',               
                'Periodt',           
                'Boujee',            
                'Main character',    
                'Based',
                'Thats facts'
             ],
            'confused': [ 
                'TFW',                
                'Hits different',      
                'Rent free',           
                'Big yikes'                            
            ],              
                                         
            
        }
        return brainrot_dict 


    def translate_mood_to_brainrot(self, mood):
        """
        Translate the given mood to Gen Z slang using the brainrot dictionary.
        Randomly selects one or two brainrot terms for chosen mood of user.
        """
        mood = mood.lower().strip()  # Normalize the input 
        slang_terms = self.genz_brainrot_dict.get(mood, []) 
        
        if slang_terms:
            selected_slang = random.sample(slang_terms, k=min(4, len(slang_terms)))  # Select up to two slang terms
            return f"{', '.join(selected_slang)} ({mood})" 
        
        return f"aww what youre fr feeling {mood}. Heres a bussin bop recommendation for that lowkey mood!" 

        
    def assign_mood_based_on_sentiment(self, lyrics):
        """
        Assigns a mood based on the sentiment of the lyrics using VADER.
        """
        sentiment = self.analyzer.polarity_scores(lyrics)  # Returns a dictionary
        polarity = sentiment['compound']  # Get the compound score from VADER
        
        if polarity > 0.6:
            return 'happy'
        elif polarity < -0.6:
            return 'sad'
        elif -0.5 <= polarity <= -0.2:
            return 'disappointed' 
        elif 0.1 <= polarity <= 0.6:
            return 'chill'
        else:
            return 'neutral'
    
    def load_songs_from_csv(self, file_path): 
        """
        Load the songs dataset and create a dictionary mapping moods to songs.
        """
        songs_dict = {}
        try:
            with open(file_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    mood = row['Mood'].strip().lower()  # Normalize mood to lowercase
                    song_info = {
                        'song': row['Song'],
                        'artist': row['Artist'],
                        'lyrics': row['Lyrics'],
                        'album': row['Album']
                    }
                    if mood not in songs_dict:
                        songs_dict[mood] = []
                    songs_dict[mood].append(song_info)
            
            print(f"Loaded moods: {list(songs_dict.keys())}")

        except FileNotFoundError:
            print(f"Error: File {file_path} not found. Songs dataset not loaded.")
            return songs_dict
        except Exception as e:
            print(f"Error loading songs dataset: {e}") 
        
        return songs_dict 


    
    def get_brainrot_meaning(self, term):
        return self.genz_brainrot_dict.get(term, "Bruh wdym.")


    def greeting(self):
        """
        Display a random greeting.
        """

        greetings = [
            f"Ayyooo its ur boi {self.name}, here to yeet you some absolute bops which are fire no cap!!!",
            f"Wyu2 hun im {self.name}, here to recommend u some fire tunez",
            f"Fancy seein u here I am {self.name}, hope ur feelin hella demure whats the vibe ur after!!!",
            f"Very mindful, very demure, I am {self.name}, here to get u goin with some bops!!!",
            f"I'm working lateee cuz im a {self.name}, take a hit of ur espresso and tell me whats the vibe u want!!!",
            f"Lets relive brat summer im {self.name}, here to get u on some club classics u 365 party gurl xo"
        ]
        print(random.choice(greetings)) 


    def process_input(self, user_input): 
        """
        Processes the user's input to prepare it for further analysis.
        """

        processed_input = re.sub(r'[^\x00-\x7f]',r'', user_input) 
        processed_input = processed_input.lower() 
        return processed_input
    

    def preprocess_text(self, text):
        """Process the input text to lemmatize words."""
        # Process the text with SpaCy
        doc = self.nlp(text)  # Use SpaCy to create a Doc object
        # Lemmatize each word, removing punctuation, and return them as a single string
        return ' '.join([token.lemma_ for token in doc if not token.is_punct])

    

    def load_lyrics(self, file_path, separator, column_names): 
        """
        Load the lyrics dataset, ensuring all columns are present.
        """
        try:
            df = pd.read_csv(file_path, sep=separator, encoding='utf-8', on_bad_lines='warn')
            print("DataFrame Loaded:", df.head())


            df.columns = df.columns.str.strip()
            print(f"Original Columns Loaded: {df.columns.tolist()}") 
            print(f"Cleaned Columns: {df.columns.tolist()}") 

            for col in column_names:
                if col not in df.columns:
                    df[col]= None

            df = df[column_names] 
            df = df.fillna('')  

            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 
            
            df = df[df['Lyrics'].str.strip().notna() & (df['Lyrics'].str.strip() != '')]

            if df['Lyrics'].isnull().all(): 
                print("Error: Lyrics column is empty or invalid.")
                return pd.DataFrame(columns=column_names)  # Return an empty DataFrame

            # Assign mood based on sentiment
            df['Mood'] = df['Mood'].where(df['Mood'].str.strip() != '', 
                                           df['Lyrics'].apply(self.assign_mood_based_on_sentiment))
            
            return df

        except FileNotFoundError:
           print(f"Error: File {file_path} not found. Please ensure the dataset is available.")
           return pd.DataFrame(columns=column_names)  # Return empty DataFrame
        except Exception as e:
           print(f"Error loading dataset: {e}")
           return pd.DataFrame(columns=column_names) 



    def lyrics_to_tfidf(self, df): 
        """ 
        Convert the lyrics from the dataframe into a TF-IDF matrix.
        """
        if df.empty or df['Lyrics'].isnull().all():
         return None, None

        df = df[df['Lyrics'].str.strip().notna()]

        
        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
        lyrics_matrix = vectorizer.fit_transform(df['Lyrics'])
        
        # Debugging: Check the number of unique terms in the vocabulary
        print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
        print("TF-IDF matrix shape:", lyrics_matrix.shape)

        return vectorizer, lyrics_matrix

    
    def find_song_based_on_mood(self, mood):
        """
        Filter songs through mood of user.
        """
        mood = mood.strip().lower()

        if mood in self.songs_dict:
            song = random.choice(self.songs_dict[mood])  # Randomly choose a song for that mood
            track_name = song['song']  # Corrected to match the dictionary key
            artist_name = song['artist'] 
            lyrics = song['lyrics']

            return f"I recommend '{track_name}' by '{artist_name}'. Here's a snippet of the lyrics:\n\n{lyrics[:250]}..."
        else:
            return "Sorry, I don't have a song for that mood right now." 
        
    def generate_response(self, processed_input): 
        """
        Generate a response based on the user's input. 
        """

         # Check if the user's input is a simple goodbye
        if processed_input in ["bye", "goodbye", "see you", "later", "peace"]:
           return "Byeee King! You got caught in 4k, laters - pop off girly pop!" 
        
        detected_mood = self.detect_keywords(processed_input)  # Use the new keyword detection method
        print(f"Detected mood for input '{processed_input}': {detected_mood}") 

        if detected_mood:
        # Respond with the detected mood
            slang_response = self.translate_mood_to_brainrot(detected_mood)
            response = self.find_song_based_on_mood(detected_mood)
            return f"{slang_response} I recommend a song for that vibe: {response}"

       
        # Define regex sequence for mood and genre 
        song_regex = r'(give|recommend) me a (.+?) song(?: about (.+?))?(?: in a (.+?) mood)?'
        match = re.match(song_regex, processed_input)
        
        moods = list(self.songs_dict.keys())  # Get available moods

        if match: 
            action, genre, mood = match.groups()
            mood = mood or input("What's the vibe? (e.g., happy, sad, in-love, angry, mysterious): ").lower()

            # Respond with slang translation and song recommendation
            slang_response = self.translate_mood_to_brainrot(mood)  
            response = self.find_song_based_on_mood(mood)
            return f"{slang_response} I recommend a song for that vibe: {response}"

        if any(mood in processed_input for mood in moods):  # Check if any mood is present in user input
            mood = next((mood for mood in moods if mood in processed_input), None)  # Find the first matching mood
            slang_response = self.translate_mood_to_brainrot(mood)  # Get slang translation
            response = self.find_song_based_on_mood(mood)  # Get song recommendation
            return f"{slang_response} I recommend a song for that vibe: {response}"
        
        uncertain_phrases = [
            "Aww, no cap! On god, no stress, homie, I gotcha!", 
            "U good, broski? Thats lowkey skibidi, I got you fam!",
            "I oop, but lets vibe anyways! Here's a tune!",
            "Ehh okay u NPC, no need to make that ur personality trait, anyways heres somthin idk",
            "Cringe! Haha jkjk let me hook you up with a banger!",
            "Whatever u say short king? No problem! Ill help you catch a more slay vibe! Here's a bop!",
            "Is that vibe in the room with us? Here's what I got for ya, king!",
            "Is that vibe in the room with us? No cap, just pop off to this, diva!" 
        ] 
    
        # Randomly select a phrase from random_phrases
        random_response = random.choice(uncertain_phrases) 
        random_song = self.find_song_based_on_mood("happy")  # Default to happy if no mood specified
    
        return f"{random_response} Here's a song I recommend: {random_song}" 


    def recommend_song_by_mood(self, mood): 
        """
        Recommend a song based on the detected mood.
        """
        if mood in self.songs_dict:
            song = random.choice(self.songs_dict[mood])
            return f"Song: {song['song']}, Artist: {song['artist']}, Album: {song['album']}"
        else:
            return "Sorry, I don't have a song for that mood right now." 
        

if __name__ == "__main__": 
    music_bot = BrainrotBot() 
     
    print("Sample dataset with placeholder 'mood' column:")
    print(music_bot.df.head())  
    music_bot.greeting() 

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":  
            print("Goodbye!")
            break  # End the conversation
        processed_input = music_bot.process_input(user_input)
        response = music_bot.generate_response(processed_input) 
        print(response) 

