import json
import re
import os
import re
import nltk
import pytube
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
from urllib.parse import urlparse, parse_qs
import textwrap
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Download necessary NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_video_id(youtube_url):
    """
    Extract the YouTube video ID from various YouTube URL formats
    """
    # Handle embedded URLs
    if 'embed' in youtube_url:
        # Pattern for embedded URLs like https://www.youtube.com/embed/tXQFokoeUmw?feature=oembed
        pattern = r'embed/([a-zA-Z0-9_-]+)'
    else:
        # Pattern for standard URLs like https://www.youtube.com/watch?v=tXQFokoeUmw
        pattern = r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)'
    
    match = re.search(pattern, youtube_url)
    if match:
        return match.group(1)
    return None

def get_transcript(video_id):
    """Get the transcript of a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}."

def summarize_text_nltk(text, num_sentences=5):
    """Summarize text using frequency-based extractive summarization with NLTK."""
    if not text or text.startswith("Error") or text.startswith("Transcript not available"):
        return text
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    # If there are fewer sentences than requested, return all sentences
    if len(sentences) <= num_sentences:
        return text
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('spanish'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    # Calculate word frequencies
    freq = FreqDist(words)
    # Score sentences based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if i in sentence_scores:
                    sentence_scores[i] += freq[word]
                else:
                    sentence_scores[i] = freq[word]
    # Get the top N sentences with highest scores
    summary_sentences_indices = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences_indices.sort()  # Sort to maintain original order
    # Construct the summary
    summary = ' '.join([sentences[i] for i in summary_sentences_indices])
    return summary

def main():
    # Load the JSON data
    try:
        with open('esbieta-recipes.json', 'r', encoding='utf-8') as file:
            recipes = json.load(file)
    except FileNotFoundError:
        print("Error: JSON file not found. Please specify the correct path.")
        return
    
    # Counter for tracking progress
    total = len(recipes)
    processed = 0
    
    # Process each recipe
    for recipe in recipes:
        youtube_url = recipe.get('youtube_url', '')
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                print(f"Processing video: {recipe['title']} - {video_id}")
                transcript = get_transcript(video_id)
                recipe['youtube_transcript'] = transcript
            else:
                recipe['youtube_transcript'] = "Invalid YouTube URL format"
        else:
            recipe['youtube_transcript'] = "No YouTube URL provided"
        
        processed += 1
        print(f"Progress: {processed}/{total} recipes processed")
    
    # Save the updated data
    with open('esbieta_recipes_with_transcripts.json', 'w', encoding='utf-8') as file:
        json.dump(recipes, file, ensure_ascii=False, indent=2)
    
    print(f"Complete! Transcripts added to {processed} recipes.")

if __name__ == "__main__":
    main()