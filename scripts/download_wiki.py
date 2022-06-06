import requests
import json

# Define the URL for the Wikipedia API
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
LIMIT=None

# Define the parameters for the API query
params = {
    "action": "query",
    "generator": "allpages",
    "prop": "extracts",
    "exintro": True,
    "explaintext": True,
    "format": "json",
    "gaplimit": 100,
    "gapfrom": "A" # Starting letter of Wikipedia pages to download
}

pages_count = 0

# Open the output file in JSON Lines format
with open("wikipedia_articles.jsonl", "w") as f:

    # Loop through the API results in batches of 10 pages at a time
    while True:

        # Send the API request and get the response
        response = requests.get(WIKIPEDIA_API_URL, params=params)
        data = response.json()

        # Loop through the pages in the response and write each one to the output file
        for page_id, page_data in data["query"]["pages"].items():
            # Extract relevant fields
            title = page_data["title"]
            body = page_data["extract"]
            wiki_id = page_data["pageid"]
            
            # Write the fields to the output file in JSON Lines format
            f.write(json.dumps({"title": title, "body": body, "wikipedia_id": wiki_id}) + "\n")
            pages_count += 1

            if pages_count % 100 == 0:
                print(f"Downloaded {pages_count} pages")

            if LIMIT and pages_count == LIMIT:
                break

        # Check if there are more pages to download
        if "continue" in data:
            params["gapcontinue"] = data["continue"]["excontinue"]
        else:
            break