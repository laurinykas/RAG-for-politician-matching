import json
import os
import glob


def combine_json_files(directory_path, output_file):
    # Dictionary to store combined data
    combined_data = {}
    # Dictionary to track matches
    matches = {}

    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "lrt_*_articles.json"))

    print(f"Found {len(json_files)} JSON files to process")

    # Process each file
    for file_path in json_files:
        # Extract the name from the filename
        filename = os.path.basename(file_path)
        name = filename.replace("lrt_", "").replace("_articles.json", "")

        print(f"Processing file: {filename}, extracted name: {name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Process each article in the JSON file
                for article in data:
                    # Use URL as unique identifier
                    url = article.get("url", "")

                    if url:
                        # Ensure keywords exists and is a list
                        if "keywords" not in article:
                            article["keywords"] = []

                        # Add the name to keywords if it's not already there
                        if name not in article["keywords"]:
                            article["keywords"].append(name)

                        if url in combined_data:
                            # If URL already exists, record the match
                            if url not in matches:
                                matches[url] = [name]
                            else:
                                matches[url].append(name)
                        else:
                            # Add new article
                            combined_data[url] = article
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    # Add matches to keywords
    for url, match_names in matches.items():
        if url in combined_data:
            for match_name in match_names:
                if match_name not in combined_data[url]["keywords"]:
                    combined_data[url]["keywords"].append(match_name)

    # Convert dictionary to list
    combined_list = list(combined_data.values())

    # Write combined data to output file
    output_path = os.path.join(directory_path, output_file)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(combined_list, outfile, ensure_ascii=False, indent=4)

    print(f"Combined {len(combined_data)} unique articles with {len(matches)} matches")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    # Directory containing the JSON files - using raw string to avoid escape issues
    directory = "C:/Users/Laurynas/PycharmProjects/ttp/articles"

    # Output file name
    output_file = "combined_lrt_articles.json"

    combine_json_files(directory, output_file)