import os
import shutil
import subprocess
import re
from pydub import AudioSegment
from tqdm import tqdm
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import num2words
from anyascii import anyascii

def remove_folder_with_contents(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully removed {folder_path} and all of its contents.")
    except Exception as e:
        print(f"Error removing {folder_path}: {e}")

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

# Function to sort chapters based on their numeric order
def sort_key(chapter_file):
    numbers = re.findall(r'\d+', chapter_file)
    return int(numbers[0]) if numbers else 0

def sort_key_texts(filename):
    """Extract chapter number for sorting."""
    match = re.search(r'chapter_(\d+)\.txt', filename)
    return int(match.group(1)) if match else 0


def create_m4b_from_chapters(input_dir, ebook_file, output_dir, genData, isResume):
    
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
    temp_combined_mp3 = os.path.join(genData['genDirectories']['combinedAudioOutput'], 'combined.mp3')
    metadata_file = os.path.join(genData['genDirectories']['combinedAudioOutput'], 'metadata.txt')
    cover_image = extract_metadata_and_cover(genData['genPaths']['ebookFilePath'])
    output_m4b = genData['genPaths']['m4bOutput']

    # only try to combine the chapters if the temp combined file doesn't exist. Trivial check, mostly for debugging
    if not os.path.isfile(temp_combined_mp3): 
        if isResume:
            combine_mp3_files(chapter_files, temp_combined_mp3)
        else:
            # create empty audiosegment to load each chapter into
            combined_audio = AudioSegment.empty()

            # merge each chapter's audiosegment 
            for i in genData['genPaths']['chapterAudios']:
                combined_audio += genData['genPaths']['chapterAudios'][i]['audioSegment']

            # export the combined audio as mp3
            combined_audio.export(temp_combined_mp3, format="mp3", bitrate="192k")
            print(f"Combined audio saved to {temp_combined_mp3}")

    # only generate metadata and convert to m4b if it doesn't exist yet. Also a trivial check.
    if not os.path.isfile(output_m4b):
        generate_ffmpeg_metadata(chapter_files, metadata_file)
        try:
            create_m4b(temp_combined_mp3, metadata_file, cover_image, output_m4b)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while created the m4b file: {e}")
            return False
        else:
            remove_folder_with_contents(genData['genDirectories']['audiobooksWorkingDir'])

def prepareTexts(txtPath,genData):
    # basic cleaning definitions and functions
    _whitespace_re = re.compile(r"\s+")

    def remove_aux_symbols(text):
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        return text
    
    def collapse_whitespace(text):
        return re.sub(_whitespace_re, " ", text).strip()
    
    def replace_symbols(text):
        text = text.replace(";", ",")
        text = text.replace("-", " ")
        text = text.replace(":", ",")
        text = text.replace("&", " and ")
        text = text.replace(". . .", "")
        return text

    # convert each chapter into generated audio
    from string import printable
    import shlex

    for chapterFile in sorted(os.listdir(txtPath), key=sort_key_texts):
        if chapterFile.endswith('.txt'):
            
            match = re.search(r"chapter_(\d+).txt", chapterFile)
            if match:
                chapterNum = int(match.group(1))
            else:
                print(f"Skipping file {chapterFile} as it does not match the expected format.")
                continue
            
            print(f'Prepping chapter {chapterNum} text')

            genData['genPaths']['chapterTexts'][chapterNum] = {
                'fileName':chapterFile,
                'filePath':os.path.join(genData['genDirectories']['chapterTexts'], chapterFile),
                'content':{},
                'inferences':{}
            }

            genData['genPaths']['chapterAudios'][chapterNum] = {
                'fileName':f'audio_chapter_{chapterNum}.mp3',
                'filePath':os.path.join(genData['genDirectories']['chapterAudioOutput'], f"audio_chapter_{chapterNum}.mp3"),
            }

            

            print(f'reading lines from {genData["genPaths"]["chapterTexts"][chapterNum]["filePath"]}')
            lineList = []
            holdOver = ''
            file1 = open(genData['genPaths']['chapterTexts'][chapterNum]['filePath'], 'r',encoding='utf-8')
            Lines = file1.readlines()
            for line in Lines:
                linetext = anyascii(line)
                linetext = re.sub("[^{}]+".format(printable), "", linetext)
                linetext = shlex.split(linetext,posix=False)
                
                cleanLine = ''

                for t in linetext:
                    # if the element isn't a quote, keep concatenating to rebuild the sentence.
                    if not t.startswith("\"") and not t.endswith("\""):
                        cleanLine = f'{cleanLine} {t}' if cleanLine else t
                    else: # otherwise remove the quotes and prepend the word "quote" to be spoken
                        cleanLine = f'{cleanLine} quote {t}'.replace("\"", "")

                # hopefully fix issues with expanded tensor sizes going over 512 by expanding numbers to full words in text
                cleanLine = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), cleanLine)
                
                # make sure every section ends with a period.
                cleanLine = f'{cleanLine}.' if not cleanLine.endswith(".") or cleanLine.endswith("?") or cleanLine.endswith("!") else cleanLine

                # remove symbols like () and [], etc.
                cleanLine = remove_aux_symbols(cleanLine)

                # obvious
                cleanLine = replace_symbols(cleanLine)

                # obvious
                cleanLine = collapse_whitespace(cleanLine)

                if holdOver:
                    cleanLine = f'{holdOver} {cleanLine}'
                    holdOver = ''

                if len(re.sub(r'\W+', '', cleanLine)) < 25:
                    holdOver = cleanLine
                else:
                    lineList.append(cleanLine)
                    holdOver = ''
                
            
            #in case last line was < 25 chars, throw it on the end of the last element
            if holdOver:
                lineList[-1] = f'{lineList[-1]} {holdOver}'
                holdOver = ''

            genData['genPaths']['chapterTexts'][chapterNum]['cleanLines'] = lineList

            # for each line item (usually paragraphs from the book) split and recombine for inference
            # keeping each line/paragraph separate to add short silences in between each
            srtIter = 0
            for t in lineList:
                splitList = split_and_recombine_text(t)
                genData['genPaths']['chapterTexts'][chapterNum]['inferences'][srtIter] = splitList
                srtIter += 1

    for i in genData['genPaths']['chapterTexts']:
        for i2 in genData['genPaths']['chapterTexts'][i]['inferences']:
            for i3 in genData['genPaths']['chapterTexts'][i]['inferences'][i2]:
                print(i3)
            
    return genData

def create_chapter_labeled_book(genData):
    # Function to ensure the existence of a directory

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
        directory = genData['genDirectories']['chapterTexts']

        # Open the EPUB file
        book = epub.read_epub(epub_path)

        previous_filename = ''
        chapter_counter = 0

        # Iterate through the items in the EPUB file
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                genData['genContents'][chapter_counter] = {
                    'filePath':os.path.join(directory, f"chapter_{chapter_counter}.txt"),
                    'fileName':f"chapter_{chapter_counter}.txt",
                    'fileText':''
                }
                # Use BeautifulSoup to parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text().lstrip().rstrip()
                
                if text:
                    if len(text.strip()) < 2300:
                        continue
                    else:
                        previous_filename = genData['genContents'][chapter_counter]['filePath']
                        chapter_counter += 1
                        with open(previous_filename, 'w', encoding='utf-8') as filetext:
                            filetext.write(text)
                            print(f"Saved chapter: {previous_filename}")

    if os.path.exists(genData['genPaths']['tmpEpub'] ):
        os.remove(genData['genPaths']['tmpEpub'] )
        print(f"File {genData['genPaths']['tmpEpub'] } has been removed.")
    else:
        print(f"The file {genData['genPaths']['tmpEpub'] } does not exist.")

    if convert_to_epub(genData['genPaths']['ebookFilePath'], genData['genPaths']['tmpEpub'] ):
        save_chapters_as_text(genData['genPaths']['tmpEpub'] )

    genData = prepareTexts(genData['genDirectories']['chapterTexts'],genData)

    if os.path.isfile(genData['genPaths']['tmpEpub'] ):
        os.remove(genData['genPaths']['tmpEpub'] )
    return genData

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

import gradio as gr
import random
from webui import *

def load_audiobook_generation_settings(configPath):
    with open(configPath, "r") as f:
        return yaml.safe_load(f)

def convert_ebook_to_audio(ebook_file, progress=gr.Progress()):
    currentSettings = load_settings()
    loadedModel = load_all_models(currentSettings['voice_model'])

    global_phonemizer = loadedModel['global_phonemizer']
    model = loadedModel['model']
    model_params = loadedModel['model_params']
    sampler = loadedModel['sampler']
    textcleaner = loadedModel['textcleaner']
    to_mel = loadedModel['to_mel']

    ebook_file_path = ebook_file.name

    # Define unique working folder name for ebook
    filenameNoExtension = os.path.splitext(os.path.basename(ebook_file_path))[0]
    # remove any non-alphanumeric characters from name (just in case?) and spaces
    filenameNoExtensionMin =  re.sub(r'\W+', '', filenameNoExtension)
    print(filenameNoExtensionMin)

    # define working directory
    audiobooksDir = os.path.join(".", "audiobooks")
    audiobooksWorkingDir = os.path.join(audiobooksDir, "working")
    conversionDir = os.path.join(audiobooksWorkingDir, filenameNoExtensionMin)
    inferenceGenerationSettingsFile = os.path.join(conversionDir,f'{filenameNoExtensionMin}.yaml')
    chapterTexts = os.path.join(conversionDir, "tmpAudiobook", "chapterText")
    chapterAudioOutput = os.path.join(conversionDir, "tmpAudiobook", "audio", "chapterAudio")
    combinedAudioOutput = os.path.join(conversionDir, "tmpAudiobook", "audio", "combinedChapterAudio")
    epubConversion = os.path.join(conversionDir, "tmpAudiobook", 'tmpEpub')

    genData = {
        'genDirectories':{
            'audiobooksDir':audiobooksDir,
            'audiobooksWorkingDir':audiobooksWorkingDir,
            'conversionDir':conversionDir,
            'chapterTexts':chapterTexts,
            'chapterAudioOutput':chapterAudioOutput,
            'combinedAudioOutput':combinedAudioOutput,
            'epubConversion':epubConversion,
            'm4bOutput':os.path.join(audiobooksDir, "audiobooks")
        },
        'genPaths':{
            'inferenceGenerationSettingsFile':inferenceGenerationSettingsFile,
            'ebookFilePath':ebook_file_path,
            'tmpEpub':os.path.join(epubConversion, 'temp.epub'),
            'epubCSV':os.path.join(epubConversion, 'Other_book.csv'),
            'chapterBookText':os.path.join(epubConversion, 'Chapter_Book.txt'),
            'chapterTexts':{},
            'chapterAudios':{},
            'm4bOutput':os.path.join(audiobooksDir, f'{filenameNoExtension}.m4b')
        },
        'genContents':{

        }
    }

    # cleanliness is next to godliness
    del audiobooksDir,audiobooksWorkingDir,conversionDir,inferenceGenerationSettingsFile,chapterTexts,chapterAudioOutput,combinedAudioOutput,epubConversion

    os.makedirs(genData['genDirectories']['audiobooksDir'], exist_ok=True)
    os.makedirs(genData['genDirectories']['audiobooksWorkingDir'], exist_ok=True)
    os.makedirs(genData['genDirectories']['conversionDir'], exist_ok=True)
    os.makedirs(genData['genDirectories']['chapterTexts'], exist_ok=True)
    os.makedirs(genData['genDirectories']['chapterAudioOutput'], exist_ok=True)
    os.makedirs(genData['genDirectories']['combinedAudioOutput'], exist_ok=True)
    os.makedirs(genData['genDirectories']['epubConversion'], exist_ok=True)

    # check for inference settings for this specific book. Helps ensure consistency on later resumes
    if os.path.isfile(genData['genPaths']['inferenceGenerationSettingsFile']): # if it exists, this is probably a resume, load it
        currentSettings = load_audiobook_generation_settings(genData['genPaths']['inferenceGenerationSettingsFile'])
    else: # otherwise this is probably a new book, use current settings to create a new one.
        # ensure consistent seed across full audiobook
        currentSettings["seed"] = random.randint(0, 2**32 - 1) if currentSettings["seed"]==-1 else currentSettings["seed"]
        with open(genData['genPaths']['inferenceGenerationSettingsFile'], "w") as f:
            yaml.safe_dump(currentSettings, f)
    
    set_seeds(currentSettings["seed"])

    try:
        progress(0, desc="Starting conversion")
    except Exception as e:
        print(f"Error updating progress: {e}")

    try: # Check that calibre is installed
        subprocess.run(['ebook-convert', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Calibre is not installed. Please install Calibre for this functionality.")
    
    try:
        progress(0, desc="Creating chapter-labeled book")
    except Exception as e:
        print(f"Error updating progress: {e}")
    
    genData = create_chapter_labeled_book(genData)

    # debug
    # import json
    # print(json.dumps(genData, indent=4))
    
    try:
        progress(0, desc="Converting chapters to audio")
    except Exception as e:
        print(f"Error updating progress: {e}")

    # convert each chapter into generated audio

    totalChapterCount = len(genData['genPaths']['chapterTexts'])
    currentInference = 1
    totalInferences = 0
    for i in genData['genPaths']['chapterTexts']:
        for i2 in genData['genPaths']['chapterTexts'][i]['inferences']:
            totalInferences += len(genData['genPaths']['chapterTexts'][i]['inferences'][i2])

    # print(f'Total inferences to process: {totalInferences}')

    # create silence object to be inserted on occasion (prevents having to recreate it every inference)
    silence = AudioSegment.silent(duration=75) # between paragraph-like structures
    isResume = False

    reference_audio_path = os.path.join("voices", currentSettings["voice"], currentSettings["reference_audio_file"])

    # try to give a reasonable ETA after a bit.
    start = time.time()
    allInferencesETAList = []

    def getETA(start,inferenceCounter,totalInferences,currentInference):
        # example: if in 30 seconds it infers 22 fragments
        # 30 / 22 = 1.36 secPerInference
        # so calculate totalInferences - currentInference to get remainingInferences
        # mutliply remainingInferences * secPerInference to get inferencesETA
        # appnd inferencesETA to allInferencesETAList
        # add all items in allInferencesETAList and divide by len(allInferencesETAList) to get avgAllInferencesETAList
        # avgAllInferencesETAList should be relatively consistent after so long. maybe. Possibly. ...probably

        secPerInference = ((time.time() - start) / inferenceCounter)
        remainingInferences = totalInferences - currentInference
        inferencesETA = remainingInferences * secPerInference
        allInferencesETAList.append(inferencesETA)
        currentAverageETA = round(sum(allInferencesETAList) / len(allInferencesETAList),2)
        currentAverageETA = str(datetime.timedelta(currentAverageETA=666))

        return currentAverageETA

    # compute reference voice data
    mean, std = -4, 4
    print(f'types: model: {model}, to_mel: {to_mel}')
    print(f'reference_audio_path: {reference_audio_path}, device: {device}')
    ref_s = compute_style(reference_audio_path, model, to_mel, mean, std, device)

    # kick off background thread for exporting audio from the queue.
    queueThreadManager('start')

    expectedETA = 0

    for i in genData['genPaths']['chapterTexts']:
        chapterNum = i
        
        audio_opt_dir = os.path.dirname(genData['genPaths']['chapterAudios'][chapterNum]['filePath'])
        audio_opt_filename = os.path.basename(genData['genPaths']['chapterAudios'][chapterNum]['filePath'])
        output_file_path = os.path.join(audio_opt_dir, audio_opt_filename)

        if os.path.isfile(output_file_path):
            print(f"Chapter {chapterNum} audio file already exists, skipping")
            isResume = True

            # get number of inferences that are considered complete now
            for section in genData['genPaths']['chapterTexts'][i]['inferences']:
                currentInference =+ len(genData['genPaths']['chapterTexts'][chapterNum]['inferences'][section])
            continue

        print(f'Starting chapter {chapterNum} narration')

        # create new AudioSegment obj for the new chapter
        combined_audio = AudioSegment.empty()

        # loop over texts to be inferenced 
        for i2 in genData['genPaths']['chapterTexts'][i]['inferences']:
            inferenceCounter = 1
            currentProgressPercent = currentInference / totalInferences
            for t in genData['genPaths']['chapterTexts'][i]['inferences'][i2]:
                currentAverageETA = getETA(start,inferenceCounter,totalInferences,currentInference)
                progress(currentProgressPercent, desc=f"Inferencing: Chapter {chapterNum} of {totalChapterCount} | Section {i2} of {len(genData['genPaths']['chapterTexts'][i]['inferences'])} | Fragment {inferenceCounter} of {len(genData['genPaths']['chapterTexts'][i]['inferences'][i2])}. Estimated ETA: {currentAverageETA}")
            
            

                bytes_wav = bytes()
                byte_io = BytesIO(bytes_wav)
                # bytes_wav.seek(0)
                write(byte_io, 24000,   inference(
                                            t, 
                                            ref_s, 
                                            model, 
                                            sampler, 
                                            textcleaner, 
                                            to_mel, 
                                            device, 
                                            model_params, 
                                            global_phonemizer=global_phonemizer, 
                                            alpha=currentSettings["alpha"], 
                                            beta=currentSettings["beta"], 
                                            diffusion_steps=currentSettings["diffusion_steps"], 
                                            embedding_scale=currentSettings["embedding_scale"]
                                        )
                )
                
                audio_segment = AudioSegment.from_wav(byte_io)
                audio_segment = audio_segment[:-80]
                audio_segment = audio_segment + 1 # make it a bit louder
                combined_audio += audio_segment
                combined_audio += silence

                

                inferenceCounter += 1
                currentInference += 1

        # keep audiosegment of chapter in memory - if whole book is generated in same session there will be no need to reload them from disk
        genData['genPaths']['chapterAudios'][chapterNum]['audioSegment'] = combined_audio

        # send to background export queue to be able to resume in a later session.
        exportQueue.put({'action':'export','audioDirectory':audio_opt_dir,'audioFilename':audio_opt_filename,'audioPath':output_file_path,'combinedAudio':combined_audio})

    # stop the background thread - kind of trivial at this point but why not.
    queueThreadManager('start')
                
    print(f"All chapters converted to audio ")

    try:
        progress(0, desc="Creating M4B from chapters")
    except Exception as e:
        print(f"Error updating progress: {e}")
    
    create_m4b_from_chapters(
            genData['genDirectories']['chapterAudioOutput'], 
            genData['genPaths']['ebookFilePath'],
            genData['genDirectories']['audiobooksDir'],
            genData,
            isResume
        )

    try:
        progress(0.99, desc="Conversion complete")
    except Exception as e:
        print(f"Error updating progress: {e}")
    print(f"Audiobook created at {genData['genPaths']['m4bOutput']}")
    return f"Audiobook created at {genData['genPaths']['m4bOutput']}"


import threading, queue

# efficient audio exports in case a resume is necessary
exportQueue = queue.Queue()

def exportAudio():
    if not exportQueue.empty():
        args = exportQueue.get()
    # do the exporting here
    if args['action'] == 'export':
        try:
            args['combinedAudio'].export(os.path.join(args["audioDirectory"],f'{os.path.splitext(os.path.basename(args["audioPath"]))[0]}.mp3'), format="mp3", bitrate="320k")
        except Exception as er:
            print(f"Error combining and saving: {er}")
    if args['action'] == 'inference':
        pass


def audioExportThreadLoop(e):
    global exportQueue
    '''
    callback for dequeue threaded loop to handle pausing and resuming.
    Pausing and resuming is necessary to prevent exceptions related to either local strategy available balance not being updated or balance on exchange not being updated before executing the next buy.
    Manage the queue using:
        queueThreadManager('start') | initializes queue thread. Smart enough not to start the loop if already running and will instead act the same as "run", so I suppose run is redundant ¯\_(ツ)_/¯
        queueThreadManager('wait') | sets a wait event in effect pausing the queue thread's loop - since everything is in the loop it's basically a pause button.
        queueThreadManager('run') | resumes the paused queue thread.
    '''
    global threadCaller

    while True:
        if not threadCaller['status']:
            #print(f"THREAD: paused while waiting for the cancel to process")
            event_is_set = e.wait()
            #print(f"Looks like cancel order is done")
            threadCaller['status'] = True
            e.clear()
        if exportQueue.empty(): time.sleep(300/1000)
        else: exportAudio()

threadCaller = {
    'status': True
}
e = threading.Event()
qThread = threading.Thread(name='pausable_thread',
                    target=audioExportThreadLoop,
                    args=(e,))

def queueThreadManager(action):
    '''
    Communicate with the looping queue thread. See explanation in DequeueOrWait() head.
    '''
    global threadCaller

    if action == 'start':
        threadCaller['status'] = True if not threadCaller['status'] else False
        e.set()
        qThread.start() if not qThread.is_alive() else False

    if action == 'run':
        threadCaller['status'] = True
        e.set()

    if action == 'wait':
        threadCaller['status'] = False