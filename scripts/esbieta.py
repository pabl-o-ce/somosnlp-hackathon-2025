import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging
import argparse
import os
from urllib.parse import urljoin
import random
from concurrent.futures import ThreadPoolExecutor
import urllib.parse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recipe_scraper.log"),
        logging.StreamHandler()
    ]
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Scrape recipes from recetasdesbieta.com')
parser.add_argument('--output', type=str, default='recipes.json', help='Output JSON file')
parser.add_argument('--start-index', type=int, default=0, help='Start from specific recipe index')
parser.add_argument('--delay', type=float, default=2.0, help='Delay between requests in seconds')
parser.add_argument('--max-recipes', type=int, default=None, help='Maximum number of recipes to scrape')
parser.add_argument('--threads', type=int, default=1, help='Number of threads to use')
args = parser.parse_args()

# Setup session with headers for requests
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
})

def get_recipe_links():
    """Extract all recipe links from the alphabetical index page"""
    url = "https://www.recetasdesbieta.com/todas-las-recetas-por-orden-alfabetico/"
    
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all recipe links
        links = []
        # In the website, recipe links appear directly in the content
        content = soup.find('div', class_='entry-content')
        
        if content:
            # Get all <a> tags that are direct recipe links
            for a_tag in content.find_all('a'):
                href = a_tag.get('href')
                if href and 'recetasdesbieta.com' in href and not href.endswith('#comments'):
                    links.append(href)
                    logging.debug(f"Found recipe link: {href}")
        
        return links
    except Exception as e:
        logging.error(f"Error getting recipe links: {str(e)}")
        return []

def extract_youtube_url(soup):
    """Extract YouTube URL from the recipe page if available"""
    try:
        # Find iframe with YouTube embed
        iframe = soup.find('iframe', src=lambda s: s and 'youtube.com/embed' in s)
        if iframe and 'src' in iframe.attrs:
            return iframe['src']
        return ""
    except Exception as e:
        logging.error(f"Error extracting YouTube URL: {str(e)}")
        return ""

def extract_ingredients(soup):
    """Extract ingredients list from the recipe page"""
    try:
        ingredients = []
        # Try to find ingredients in different formats
        
        # Look for ingredient paragraphs - often they appear after bold text that says "Ingredientes"
        ingredient_section = None
        for h2 in soup.find_all(['h2', 'h3', 'h4', 'strong', 'b']):
            if 'ingrediente' in h2.text.lower():
                ingredient_section = h2.find_next('ul')
                break
        
        # If we found a list of ingredients
        if ingredient_section and ingredient_section.find_all('li'):
            for li in ingredient_section.find_all('li'):
                ingredients.append(li.text.strip())
        else:
            # Sometimes ingredients are in paragraphs
            content = soup.find('div', class_='entry-content')
            if content:
                # Try to identify the ingredients section
                paragraphs = content.find_all('p')
                for i, p in enumerate(paragraphs):
                    text = p.text.strip().lower()
                    if ('ingrediente' in text or 'necesita' in text) and i + 1 < len(paragraphs):
                        # The next paragraph might contain ingredients
                        ingredient_text = paragraphs[i + 1].text.strip()
                        # Split by commas or line breaks
                        for item in re.split(r'[,\n]', ingredient_text):
                            if item.strip() and not item.strip().startswith('http'):
                                ingredients.append(item.strip())
        
        return ingredients
    except Exception as e:
        logging.error(f"Error extracting ingredients: {str(e)}")
        return []

def extract_instructions(soup):
    """Extract cooking instructions from the recipe page"""
    try:
        instructions = ""
        content = soup.find('div', class_='entry-content')
        
        if not content:
            return instructions
            
        # Instructions typically start after ingredients
        # Look for common headers that indicate the start of instructions
        instruction_markers = ['preparación', 'elaboración', 'cómo hacer', 'procedimiento', 
                              'vamos', 'paso a paso', 'instrucciones']
        
        # First try to find the instructions section by headers
        found = False
        for header in content.find_all(['h2', 'h3', 'h4', 'strong', 'b']):
            header_text = header.text.lower()
            
            if any(marker in header_text for marker in instruction_markers):
                # Found the start of instructions
                instructions_elem = header.find_next('ol')
                if instructions_elem:
                    # If we found an ordered list, use that
                    steps = []
                    for li in instructions_elem.find_all('li'):
                        steps.append(li.text.strip())
                    instructions = "\n".join(steps)
                    found = True
                    break
                else:
                    # Otherwise, collect paragraphs that follow
                    steps = []
                    for elem in header.next_siblings:
                        if elem.name == 'p':
                            paragraph_text = elem.text.strip()
                            if paragraph_text:
                                steps.append(paragraph_text)
                        elif elem.name in ['h2', 'h3', 'h4', 'div', 'section']:
                            # Stop when we hit another header or div
                            break
                    
                    instructions = "\n".join(steps)
                    found = True
                    break
        
        # If we didn't find instructions by headers, try to extract from paragraphs
        if not found:
            # Get all paragraphs
            paragraphs = content.find_all('p')
            
            # Skip early paragraphs (typically intro and ingredients)
            start_idx = min(2, len(paragraphs) // 3)
            
            # Collect middle paragraphs as instructions
            end_idx = max(start_idx + 1, len(paragraphs) - 2)
            instruction_paragraphs = paragraphs[start_idx:end_idx]
            
            instructions = "\n".join(p.text.strip() for p in instruction_paragraphs)
        
        return instructions
    except Exception as e:
        logging.error(f"Error extracting instructions: {str(e)}")
        return ""

def extract_main_image(soup, url):
    """Extract the main image of the recipe"""
    try:
        # First try to find the featured image
        featured_img = soup.find('div', class_='featured-image')
        if featured_img and featured_img.find('img') and featured_img.find('img').get('src'):
            return featured_img.find('img')['src']
        
        # Second, try to find the first image in the content
        content = soup.find('div', class_='entry-content')
        if content and content.find('img') and content.find('img').get('src'):
            img_src = content.find('img')['src']
            # Make sure it's an absolute URL
            if not img_src.startswith('http'):
                img_src = urljoin(url, img_src)
            return img_src
        
        # Try meta image
        meta_img = soup.find('meta', property='og:image')
        if meta_img and meta_img.get('content'):
            return meta_img['content']
            
        return ""
    except Exception as e:
        logging.error(f"Error extracting main image: {str(e)}")
        return ""

def extract_youtube_transcript(youtube_url):
    """Try to extract YouTube transcript - simplified version"""
    try:
        if not youtube_url or 'youtube.com/embed/' not in youtube_url:
            return ""
            
        # Extract video ID from the URL
        video_id = youtube_url.split('/')[-1].split('?')[0]
        
        # Normally you would use youtube_transcript_api here
        # This is a placeholder since we don't want to add external dependencies
        # You would replace this with actual transcript extraction code
        # from youtube_transcript_api import YouTubeTranscriptApi
        # transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # return " ".join([entry['text'] for entry in transcript])
        
        # For now just return a placeholder
        return f"Transcript for YouTube video {video_id} would be extracted here."
    except Exception as e:
        logging.error(f"Error extracting YouTube transcript: {str(e)}")
        return ""

def scrape_recipe(url):
    """Scrape a single recipe page"""
    try:
        logging.info(f"Scraping recipe: {url}")
        
        # Add a delay to avoid hammering the server
        time.sleep(args.delay + random.uniform(0, 1))
        
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all required information
        title = soup.find('h1', class_='entry-title').text.strip() if soup.find('h1', class_='entry-title') else "Unknown Title"
        image_url = extract_main_image(soup, url)
        youtube_url = extract_youtube_url(soup)
        ingredients = extract_ingredients(soup)
        instructions = extract_instructions(soup)
        
        # Get full content 
        content_div = soup.find('div', class_='entry-content')
        full_content = content_div.text.strip() if content_div else ""
        
        # Get YouTube transcript (if available)
        youtube_transcript = extract_youtube_transcript(youtube_url) if youtube_url else ""
        
        # Return recipe data
        return {
            "title": title,
            "url": url,
            "image_url": image_url,
            "youtube_url": youtube_url,
            "ingredients": ingredients,
            "instructions": instructions,
            "full_content": full_content,
            "youtube_transcript": youtube_transcript
        }
    except Exception as e:
        logging.error(f"Error scraping recipe {url}: {str(e)}")
        return None

def load_existing_recipes(filename):
    """Load existing recipes from a JSON file if it exists"""
    if not os.path.exists(filename):
        return []
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading existing recipes: {str(e)}")
        return []

def save_recipes(recipes, filename):
    """Save recipes data to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(recipes)} recipes to {filename}")
    except Exception as e:
        logging.error(f"Error saving recipes: {str(e)}")

def main():
    """Main function to orchestrate the scraping process"""
    logging.info("Starting recipe scraper")
    
    # Get all recipe links
    all_links = get_recipe_links()
    logging.info(f"Found {len(all_links)} recipe links")
    
    # Limit the number of recipes if requested
    if args.max_recipes:
        all_links = all_links[:args.max_recipes]
        logging.info(f"Limited to {len(all_links)} recipes as requested")
    
    # Load existing data if resuming
    recipes = load_existing_recipes(args.output)
    logging.info(f"Loaded {len(recipes)} existing recipes")
    
    # Create a set of already scraped URLs for quick lookup
    scraped_urls = {recipe['url'] for recipe in recipes}
    
    # Filter out already scraped recipes
    links_to_scrape = [link for link in all_links[args.start_index:] if link not in scraped_urls]
    logging.info(f"Remaining recipes to scrape: {len(links_to_scrape)}")
    
    if not links_to_scrape:
        logging.info("No new recipes to scrape. Exiting.")
        return
    
    # Scrape recipes
    if args.threads > 1:
        # Use ThreadPoolExecutor for parallel scraping
        logging.info(f"Scraping with {args.threads} threads")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            new_recipes = list(executor.map(scrape_recipe, links_to_scrape))
        
        # Filter out None values (failed scrapes)
        new_recipes = [r for r in new_recipes if r]
        recipes.extend(new_recipes)
    else:
        # Scrape sequentially
        for i, link in enumerate(links_to_scrape):
            logging.info(f"Scraping recipe {i+1}/{len(links_to_scrape)}: {link}")
            
            # Scrape the recipe
            recipe_data = scrape_recipe(link)
            
            if recipe_data:
                recipes.append(recipe_data)
                
                # Save incrementally every 10 recipes
                if (i + 1) % 10 == 0:
                    save_recipes(recipes, args.output)
    
    # Final save
    save_recipes(recipes, args.output)
    logging.info("Recipe scraping completed")

if __name__ == "__main__":
    main()