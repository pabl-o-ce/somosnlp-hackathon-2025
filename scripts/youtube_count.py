import requests
import re
import json
import time

# Add User-Agent to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def getStats(youtube_url):
    try:
        # Convert embed URL to watch URL
        video_id_match = re.search(r'embed/([^?]+)', youtube_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            watch_url = youtube_url

        print(f"Fetching data for: {watch_url}")
        page = requests.get(watch_url, headers=headers)
        page_content = page.text

        # Try different patterns to find like count
        like_match = re.search(r'"label":"([\d,]+) likes"', page_content) or \
                    re.search(r'"likeCount":"(\d+)"', page_content) or \
                    re.search(r'"likesText":"([\d,]+)"', page_content) or \
                    re.search(r'"likes":(\d+)', page_content)

        if like_match:
            likes = like_match.group(1).replace(',', '')
            return int(likes)
        else:
            # Backup approach - try to find it in a different format
            like_text_match = re.search(r'like this video along with ([\d,]+)', page_content)
            if like_text_match:
                likes = like_text_match.group(1).replace(',', '')
                return int(likes)
            return 0

    except Exception as e:
        print(f"Error processing {youtube_url}: {str(e)}")
        return 0

# Function to load the JSON data
def load_json_data(content_string):
    try:
        # Try to load the JSON directly
        data = json.loads(content_string)
        return data
    except json.JSONDecodeError:
        # If it fails, try to repair it by checking if it's an incomplete array
        if content_string.strip().startswith("[") and not content_string.strip().endswith("]"):
            try:
                repaired_content = content_string.strip() + "]"
                data = json.loads(repaired_content)
                print("Fixed incomplete JSON array")
                return data
            except json.JSONDecodeError:
                print("Could not repair JSON")
                return None
        print("Invalid JSON format")
        return None

# Main function
def main():
    # Load JSON data from the provided paste.txt
    try:
        with open('es-food-recipes-with-images.json', 'r', encoding='utf-8') as file:
            content = file.read()
            data = load_json_data(content)

        if not data:
            print("Failed to parse JSON data")
            return

        # Process each recipe
        for recipe in data:
            youtube_url = recipe.get('youtube_url')
            if youtube_url:
                print(f"Processing: {recipe['nombre']}")
                likes = getStats(youtube_url)
                recipe['votos'] = likes
                print(f"Updated '{recipe['nombre']}' with {likes} likes")
                # Add a small delay to avoid rate limiting
                time.sleep(1.5)

        # Save the updated JSON
        with open('updated_recipes.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        print("Success! Updated JSON saved to 'updated_recipes.json'")

    except FileNotFoundError:
        print("Error: paste.txt file not found")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
