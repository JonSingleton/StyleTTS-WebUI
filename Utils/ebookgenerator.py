import os
import shutil
import subprocess
import re
from pydub import AudioSegment
import tempfile
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import csv

def is_folder_empty(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List directory contents
        if not os.listdir(folder_path):
            return True  # The folder is empty
        else:
            return False  # The folder is not empty
    else:
        print(f"The path {folder_path} is not a valid folder.")
        return None  # The path is not a valid folder

def remove_folder_with_contents(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully removed {folder_path} and all of its contents.")
    except Exception as e:
        print(f"Error removing {folder_path}: {e}")

def wipe_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all the items in the given folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # If it's a file, remove it and print a message
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Removed file: {item_path}")
        # If it's a directory, remove it recursively and print a message
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Removed directory and its contents: {item_path}")
    
    print(f"All contents wiped from {folder_path}.")

def create_m4b_from_chapters(input_dir, ebook_file, output_dir):
    # Function to sort chapters based on their numeric order
    def sort_key(chapter_file):
        numbers = re.findall(r'\d+', chapter_file)
        return int(numbers[0]) if numbers else 0

    # Extract metadata and cover image from the eBook file
    def extract_metadata_and_cover(ebook_path):
        try:
            cover_path = ebook_path.rsplit('.', 1)[0] + '.jpg'
            subprocess.run(['ebook-meta', ebook_path, '--get-cover', cover_path], check=True)
            if os.path.exists(cover_path):
                return cover_path
        except Exception as e:
            print(f"Error extracting eBook metadata or cover: {e}")
        return None
    # Combine mp3 files into a single file
    def combine_mp3_files(chapter_files, output_path):
        # Initialize an empty audio segment
        combined_audio = AudioSegment.empty()

        # Sequentially append each file to the combined_audio
        for chapter_file in chapter_files:
            print(f'Chapter file: {chapter_file}')
            audio_segment = AudioSegment.from_mp3(chapter_file)
            audio_segment + 6 # boost volume a bit - will need to make user-facing option eventually.
            combined_audio += audio_segment
        # Export the combined audio to the output file path
        combined_audio.export(output_path, format="mp3", bitrate="192k")
        print(f"Combined audio saved to {output_path}")
    
    # Function to generate metadata for M4B chapters
    def generate_ffmpeg_metadata(chapter_files, metadata_file):
        with open(metadata_file, 'w') as file:
            file.write(';FFMETADATA1\n')
            start_time = 0
            for index, chapter_file in enumerate(chapter_files):
                duration_ms = len(AudioSegment.from_mp3(chapter_file))
                file.write(f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_time}\n')
                file.write(f'END={start_time + duration_ms}\ntitle=Chapter {index + 1}\n')
                start_time += duration_ms

    # Generate the final M4B file using ffmpeg
    def create_m4b(combined_wav, metadata_file, cover_image, output_m4b):
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_m4b), exist_ok=True)
        
        ffmpeg_cmd = ['ffmpeg', '-i', combined_wav, '-i', metadata_file]
        if cover_image:
            ffmpeg_cmd += ['-i', cover_image, '-map', '0:a', '-map', '2:v']
        else:
            ffmpeg_cmd += ['-map', '0:a']
        
        ffmpeg_cmd += ['-map_metadata', '1', '-c:a', 'aac', '-b:a', '192k']
        if cover_image:
            ffmpeg_cmd += ['-c:v', 'png', '-disposition:v', 'attached_pic']
        ffmpeg_cmd += [output_m4b]

        subprocess.run(ffmpeg_cmd, check=True)



    # Main logic
    chapter_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp3')], key=sort_key)
    temp_dir = os.path.join(".", "audiobooks", "working", "temp_ebook", "combined")
    temp_combined_mp3 = os.path.join(temp_dir, 'combined.mp3')
    metadata_file = os.path.join(temp_dir, 'metadata.txt')
    cover_image = extract_metadata_and_cover(ebook_file)
    output_m4b = os.path.join(output_dir, os.path.splitext(os.path.basename(ebook_file))[0] + '.m4b')

    # only try to combine the chapters if the temp combined file doesn't exist. Trivial check, mostly for debugging
    if not os.path.isfile(temp_combined_mp3): combine_mp3_files(chapter_files, temp_combined_mp3)

    # only generate metadata and convert to m4b if it doesn't exist yet. Also a trivial check.
    if not os.path.isfile(output_m4b):
        generate_ffmpeg_metadata(chapter_files, metadata_file)
        create_m4b(temp_combined_mp3, metadata_file, cover_image, output_m4b)

    # Cleanup
    # if os.path.exists(temp_combined_mp3):
    #     os.remove(temp_combined_mp3)
    # if os.path.exists(metadata_file):
    #     os.remove(metadata_file)
    # if cover_image and os.path.exists(cover_image):
    #     os.remove(cover_image)

def create_chapter_labeled_book(ebook_file_path):
    # Function to ensure the existence of a directory
    def ensure_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")

    ensure_directory(os.path.join(".", "audiobooks", 'working', 'Book'))

    def convert_to_epub(input_path, output_path):
        # Convert the ebook to EPUB format using Calibre's ebook-convert
        if os.path.isfile(output_path):
            print(f'{output_path} already created, skipping epub conversion.')
        else:
            try:
                subprocess.run(['ebook-convert', input_path, output_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while converting the eBook: {e}")
                return False
            return True

    def save_chapters_as_text(epub_path):
        # Create the directory if it doesn't exist
        directory = os.path.join(".", "audiobooks", "working", "temp_ebook")
        ensure_directory(directory)

        # Open the EPUB file
        book = epub.read_epub(epub_path)

        previous_chapter_text = ''
        previous_filename = ''
        chapter_counter = 0
        chapterTexts = {}

        from string import printable

        # Iterate through the items in the EPUB file
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:

                # don't regenerate existing chapter files
                if os.path.isfile(os.path.join(directory, f"chapter_{chapter_counter}.txt")):
                    print(f'chapter_{chapter_counter}.txt already exists, skipping')
                    continue
                else:
                    # Use BeautifulSoup to parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()

                    from cleantext import clean
                    text = clean(text,
                        fix_unicode=True,               # fix various unicode errors
                        to_ascii=True,                  # transliterate to closest ASCII representation
                        lower=False,                     # lowercase text
                        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
                        no_urls=False,                  # replace all URLs with a special token
                        no_emails=False,                # replace all email addresses with a special token
                        no_phone_numbers=False,         # replace all phone numbers with a special token
                        no_numbers=False,               # replace all numbers with a special token
                        no_digits=False,                # replace all digits with a special token
                        no_currency_symbols=False,      # replace all currency symbols with a special token
                        no_punct=False,                 # remove punctuations
                        replace_with_punct="",          # instead of removing punctuations you may replace them
                        replace_with_url="<URL>",
                        replace_with_email="<EMAIL>",
                        replace_with_phone_number="<PHONE>",
                        replace_with_number="<NUMBER>",
                        replace_with_digit="0",
                        replace_with_currency_symbol="<CUR>",
                        lang="en"                       # set to 'de' for German special handling
                    )
                    
                    if text.strip():
                        if len(text) < 2300 and previous_filename:
                            # Append text to the previous chapter if it's short
                            # chapterTexts[chapter_counter] = f'{chapterTexts[chapter_counter]}\n{text}'
                            with open(previous_filename, 'a', encoding='utf-8') as filetext:
                                filetext.write('\n' + text)
                        else:
                            # Create a new chapter file and increment the counter
                            # chapterTexts[chapter_counter] = text
                            previous_filename = os.path.join(directory, f"chapter_{chapter_counter}.txt")
                            chapter_counter += 1
                            with open(previous_filename, 'w', encoding='utf-8') as filetext:
                                filetext.write(text)
                                print(f"Saved chapter: {previous_filename}")

    input_ebook = ebook_file_path 
    output_epub = os.path.join(".", "audiobooks", "working", "temp.epub")


    if os.path.exists(output_epub):
        os.remove(output_epub)
        print(f"File {output_epub} has been removed.")
    else:
        print(f"The file {output_epub} does not exist.")

    if convert_to_epub(input_ebook, output_epub):
        save_chapters_as_text(output_epub)

    def process_chapter_files(folder_path, output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(['Text', 'Start Location', 'End Location', 'Is Quote', 'Speaker', 'Chapter'])

            # Process each chapter file
            os.path.splitext(os.path.basename("chapter_file"))[0]
            chapter_files = sorted(os.listdir(folder_path))
            for filename in chapter_files:
                filename = os.path.basename(filename)
                if filename.startswith('chapter_') and filename.endswith('.txt'):
                    chapter_number = int(filename.split('_')[1].split('.')[0])
                    file_path = os.path.join(folder_path, filename)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            from string import printable
                            text = re.sub('[^a-zA-Z0-9\n\.]', ' ', re.sub("[^{}]+".format(printable), "", file.read()).strip()).rstrip()
                            # text = file.read()
                            # Insert "NEWCHAPTERABC" at the beginning of each chapter's text
                            if text:
                                text = "NEWCHAPTERABC" + text
                            sentences = nltk.tokenize.sent_tokenize(text)
                            for sentence in sentences:
                                start_location = text.find(sentence)
                                end_location = start_location + len(sentence)
                                writer.writerow([sentence, start_location, end_location, 'True', 'Narrator', chapter_number])
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

    # Example usage
    folder_path = os.path.join(".", "audiobooks", "working", "temp_ebook")
    output_csv = os.path.join(".", "audiobooks", "working", "Book", "Other_book.csv")
    if not os.path.isfile(output_csv):
        process_chapter_files(folder_path, output_csv)

    def sort_key(filename):
        """Extract chapter number for sorting."""
        match = re.search(r'chapter_(\d+)\.txt', filename)
        return int(match.group(1)) if match else 0

    def combine_chapters(input_folder, output_file):
        # Create the output folder if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # List all txt files and sort them by chapter number
        files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        sorted_files = sorted(files, key=sort_key)

        with open(output_file, 'w', encoding='utf-8') as outfile:  # Specify UTF-8 encoding here
            for i, filename in enumerate(sorted_files):
                with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:  # And here
                    outfile.write(infile.read())
                    # Add the marker unless it's the last file
                    if i < len(sorted_files) - 1:
                        outfile.write("\nNEWCHAPTERABC\n")

    # Paths
    input_folder = os.path.join(".", "audiobooks", 'working', 'temp_ebook')
    output_file = os.path.join(".", "audiobooks", 'working', 'Book', 'Chapter_Book.txt')


    # Combine the chapters
    combine_chapters(input_folder, output_file)

    ensure_directory(os.path.join(".", "audiobooks", "working", "Book"))

def combine_wav_files(input_directory, output_directory, file_name):
    # Ensure that the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Specify the output file path
    output_file_path = os.path.join(output_directory, file_name)
    output_mp3_path = os.path.join(output_directory,f'{os.path.splitext(os.path.basename(output_file_path))[0]}.mp3')

    # Initialize an empty audio segment
    combined_audio = AudioSegment.empty()

    # Get a list of all .wav files in the specified input directory and sort them
    input_file_paths = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".wav")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    # Sequentially append each file to the combined_audio
    for input_file_path in input_file_paths:
        audio_segment = AudioSegment.from_wav(input_file_path)
        audio_segment = audio_segment + 2 # make it a bit louder
        combined_audio += audio_segment

    # Export the combined audio to the output file path
    combined_audio.export(output_mp3_path, format="mp3", bitrate="320k")

    print(f"Combined audio saved to {output_file_path}")

def combine_mp3_files(input_directory, output_directory, file_name):
    # Ensure that the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Specify the output file path
    output_file_path = os.path.join(output_directory, file_name)

    # Initialize an empty audio segment
    combined_audio = AudioSegment.empty()

    # Get a list of all .wav files in the specified input directory and sort them
    input_file_paths = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".mp3")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    # Sequentially append each file to the combined_audio
    for input_file_path in input_file_paths:
        audio_segment = AudioSegment.from_mp3(input_file_path)
        combined_audio += audio_segment

    # Export the combined audio to the output file path
    combined_audio.export(output_file_path, format='mp3', bitrate="320k")

    print(f"Combined audio saved to {output_file_path}")

def list_audiobook_files(audiobook_folder):
    # List all files in the audiobook folder
    files = []
    for filename in os.listdir(audiobook_folder):
        if filename.endswith('.m4b'):  # Adjust the file extension as needed
            files.append(os.path.join(audiobook_folder, filename))
    return files

def download_audiobooks():
    audiobook_output_path = os.path.join(".", "audiobooks")
    return list_audiobook_files(audiobook_output_path)