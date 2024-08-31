import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import math
import string

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load('en_core_web_sm')
from wordcloud import WordCloud


############################ plot func ####################
def plot_sentiment_distribution(df,save_path):
    """
    Plot a bar plot and a donut chart showing the distribution of sentiments.

    :param df: pandas DataFrame, the input DataFrame containing a 'Sentiment' column
    :return: None
    """
    sentiment_counts = df['labels'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Distribution',fontsize=20)

    ax1 = axes[0]
    sentiment_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightgreen', 'salmon','red'])
    ax1.set_title('Distribution of Sentiments')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')

  
    for i in ax1.patches:
        ax1.text(i.get_x() + i.get_width() / 2, i.get_height() + 5,
                 str(int(i.get_height())), ha='center', va='bottom')

 
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                                       startangle=140, colors=['skyblue', 'lightgreen', 'salmon','red'],
                                       wedgeprops=dict(width=0.3))

    for text in autotexts:
        text.set_color('black')

    ax2.set_title('Sentiment Distribution (Percentage)')

 
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

  
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

## word_clouds
def plot_word_clouds(df, text_column='tweets', class_column='labels'):
    # Get unique classes
    classes = df[class_column].unique()
    num_classes = len(classes)
    
    # Set up the plot dimensions
    fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 6), squeeze=False)
    fig.suptitle('Words Clouds Dataset', fontsize=25)
    
    for i, class_label in enumerate(classes):
        # Combine all text for the current class
        class_text = ' '.join(df[df[class_column] == class_label][text_column])

        # Create the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(class_text)

        # Plot the word cloud
        ax = axes[0, i]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{class_label} Word Cloud', fontsize=15)
    
    plt.tight_layout()
    plt.show()


############################ func ####################
def balance_by_removing(df, class_column='labels'):
    # Find the size of the smallest class
    class_counts = df[class_column].value_counts()
    min_class_size = class_counts.min()

   
    balanced_df = pd.DataFrame()

    
    for cls in class_counts.index:
        if class_counts[cls] > min_class_size:
            class_subset = df[df[class_column] == cls].drop_duplicates()
            balanced_class_subset = class_subset.sample(n=min_class_size)
            balanced_df = pd.concat([balanced_df, balanced_class_subset], ignore_index=True)
        else:
            small_class_data = df[df[class_column] == cls]
            balanced_df = pd.concat([balanced_df, small_class_data], ignore_index=True)
            
    return balanced_df

def remove_duplicate(df):   #remove duplicate
    """
    Cleans the tweet dataset by removing retweets and duplicates.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the tweet data.
    

    Returns:
    pandas.DataFrame, str: The cleaned DataFrame and the dataset name.
    """
    # Retweet removing
    print(df.shape[0])
    retweets = df['tweets'].str.startswith('RT')
    if retweets.sum() > 0:
        print(f"Number of retweets: {retweets.sum()} rows")
        df = df[~retweets]
    
    # Check for duplicates
    duplicates = df.duplicated(subset='tweets', keep='first')
    print(f"Number of duplicates: {duplicates.sum()} rows")
    
    # Display the duplicate rows, if any
    if duplicates.sum() > 0:
        print("Duplicate rows:")
        print(df[duplicates])

    # Handle duplicates by removing them
    df_cleaned = df.drop_duplicates(subset='tweets', keep='first')

    # Verify that duplicates are removed
    print(f"Number of rows after removing duplicates: {df_cleaned.shape[0]}")

  
    return df_cleaned


# Function To Remove the URLs from text 

# Define the function
def remove_url_and_domains(text):
    """
    Remove URLs and domain-like patterns from the text.

    :param text: str, the text from which URLs and domains will be removed
    :return: str, the cleaned text
    """
    if not isinstance(text, str):
        return text  # or handle non-string types appropriately

    # Remove URLs
    re_url = re.compile(r'https?://\S+|www\.\S+')
    text = re_url.sub('', text)
    
    # Define the regex pattern to match domain-like patterns
    pattern = r'\b(?:\w+@\w+\.\w+|\w+\.\w+|\w+\.(?:com|tv|org|net|edu|gov|mil|int))\b'
    
    # Remove domain-like patterns
    cleaned_text = re.sub(pattern, '', text)
    
    # Remove multiple spaces resulting from removal and trim leading/trailing spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


# Function To Remove Emojies

def remove_all_emojis(text):
    """
    Removes all emojis from the given text using regular expressions.
    """
    # Define a regex pattern to match emojis
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # Emoticons
        u'\U0001F300-\U0001F5FF'  # Symbols & Pictographs
        u'\U0001F680-\U0001F6FF'  # Transport & Map Symbols
        u'\U0001F700-\U0001F77F'  # Alchemical Symbols
        u'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
        u'\U0001F800-\U0001F8FF'  # Supplemental Arrows Extended-A
        u'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        u'\U0001FA00-\U0001FA6F'  # Chess Symbols
        u'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
        u'\U00002702-\U000027B0'  # Dingbats
        u'\U000024C2-\U0001F251'  # Enclosed Characters
        u'\U0001F004-\U0001F0CF'  # Playing Cards
        u'\U0001F18E'              # Specific Symbols
        u'\U0001F191-\U0001F251'  # Additional Symbols
        u'\U0001F600-\U0001F64F'  # Emoticons
        u'\U0001F300-\U0001F5FF'  # Symbols & Pictographs
        u'\U0001F680-\U0001F6FF'  # Transport & Map Symbols
        u'\U0001F700-\U0001F77F'  # Alchemical Symbols
        u'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
        u'\U0001F800-\U0001F8FF'  # Supplemental Arrows Extended-A
        u'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        u'\U0001FA00-\U0001FA6F'  # Chess Symbols
        u'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
        u'\U00002702-\U000027B0'  # Dingbats
        u'\U000024C2-\U0001F251'  # Enclosed Characters
        ']+', 
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', text)



# Replacing some common patterns the exists in many samples 

# Collectd Manually
replacement_dict = {
    r'\br\b'         : 'are'           ,          r'\bwkly\b'    : 'weekly'   ,
    r'\bk\b'         : 'ok'            ,          r'\bu\b'       : 'you'      ,
    r'\btkts\b'      : 'tickets'       ,          r'\bb\b'       : 'be'       ,
    r'\baft\b'       : 'after'         ,          r'&amp;'       : ''         ,
    r'â€™'           : "'"             ,          r'\bur\b'      : 'your'     ,
    r'\bv\b'         : 'very'          ,          r'\bpls\b'     : 'please'   ,
    r'\bc\b'         : 'see'           ,          r'\blar\b'     : 'later'    ,          
    r'\bda\b'        : 'the'           ,          r'frnd'        : 'friend'   ,          
    r'\bwat\b'       : 'what'          ,          r'\babt\b'     : 'about'    ,
    r'\bwen\b'       : 'when'          ,          r'\benuff\b'   : 'enough'   ,          
    r'\bn\b'         : 'in'            ,          r'\brply\b'    : 'reply'    ,
    r'\bthk\b'       : 'think'         ,          r'\btot\b'     : 'thought'  ,
    r'\bnite\b'      : 'night'         ,          r'\bnvm\b'     : 'never mind',
    r'\btxt\b'       : 'text'          ,          r'\btxting\b'  : 'texting'  ,
    r'\bgr8\b'       : 'great'         ,          r'\bim\b'      : 'i am'     ,
    r'\b<unk>\b'     : ''              ,          r'\bfav\b'     : 'favorite' ,
    r'\bdlvr\b'      : 'deliver'       ,          r"(?<=\s)'m(?=\s)|^'m(?=\s)|(?<=\s)'m$" : 'am',
    r'\b\w*\d\w*\b'  : ''              ,          r"(?<=\s)<unk>(?=\s)|^<unk>(?=\s)|(?<=\s)<unk>$": ''           
    
}

# Define the cleaning function
def clean_tweets(df, replacement_dict):
    def replace_patterns(text):
        for pattern, replacement in replacement_dict.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    df['tweets'] = df['tweets'].apply(replace_patterns)
    return df



# Function To Remove the punctiuation from the text

def remove_punctuation_and_special_characters(text):
    exclude = string.punctuation.replace("'",'')
    text = text.translate(str.maketrans('', '', exclude))
    
    # Define a regex pattern to match punctuation and special characters, including underscores
    pattern = r"[^\w\s']"
    
    # Replace matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

# Function To Remove Stop and common Words From the Text
stop_words = stopwords.words('english') + ["'s", "'m", "'t","chatgpt", "openai"]
def remove_stopwords(text):
    filtered_text = []
    text = text.split() 
    for word in text:
        if word not in stop_words:
            filtered_text.append(word)
    return ' '.join(filtered_text)



# Function To Lemmatize the Text
def lemmatize_text(text):
    if isinstance(text, list):
        # If text is a list of tokens
        doc = nlp(' '.join(text))
        lemmas = [token.lemma_ for token in doc]
    else:
        # If text is a string
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)




def preprocess_text(df):
    # Copy From the Dataset
    processed_df = df.copy()
    processed_df= remove_duplicate(processed_df)
    # Remove Emojis
    processed_df['tweets'] = processed_df['tweets'].apply(remove_all_emojis)
    
    # Remove URLs and Domain Names
    processed_df['tweets'] = processed_df['tweets'].apply(remove_url_and_domains)
    
    # Lowercase the Dataset
    processed_df['tweets'] = processed_df['tweets'].str.lower()
    
    # Replace and Remove Some Common Special Patterns
    processed_df = clean_tweets(processed_df, replacement_dict)
    
    # Remove Special Characters and Punctuation
    processed_df['tweets'] = processed_df['tweets'].apply(remove_punctuation_and_special_characters)
    
    # Drop any rows where 'Tweet' is NaN
    processed_df.dropna(subset=['tweets'], inplace=True)
    
 
    # Stop-words Removal     
    processed_df['tweets'] =  processed_df['tweets'].apply(remove_stopwords)
    
    # Lemmatization
    processed_df['tweets'] =  processed_df['tweets'].apply(lemmatize_text)
    
    return  processed_df



def preprocess_text_DL(df, output_csv_path):
    # Copy From the Dataset
    processed_df = df.copy()
    processed_df= remove_duplicate(processed_df)
    # Remove Emojis
    processed_df['tweets'] = processed_df['tweets'].apply(remove_all_emojis)
    
    # Remove URLs and Domain Names
    processed_df['tweets'] = processed_df['tweets'].apply(remove_url_and_domains)
    
    # Lowercase the Dataset
    #processed_df['tweets'] = processed_df['tweets'].str.lower()
    
    # Replace and Remove Some Common Special Patterns
    processed_df = clean_tweets(processed_df, replacement_dict)
    
    # Remove Special Characters and Punctuation
    processed_df['tweets'] = processed_df['tweets'].apply(remove_punctuation_and_special_characters)
    
    # Drop any rows where 'Tweet' is NaN
    processed_df.dropna(subset=['tweets'], inplace=True)
    
    processed_df.to_csv(output_csv_path, index=False)
    # Stop-words Removal     
   # processed_df['tweets'] =  processed_df['tweets'].apply(remove_stopwords)
    
    # Lemmatization
   # processed_df['tweets'] =  processed_df['tweets'].apply(lemmatize_text)
    
    return  processed_df