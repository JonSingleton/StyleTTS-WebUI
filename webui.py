'''
Things that I need to work on:
    - Need to unload previous model before loading new one
    - Model loading needs work

'''

import os
import sys

if os.path.exists("runtime"):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Add this directory to sys.path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    espeak_path = os.path.join(os.path.dirname(__file__), 'espeak NG')
    espeak_library = os.path.join(os.path.dirname(__file__), 'espeak NG', 'libespeak-ng.dll')
    espeak_data_path = os.path.join(espeak_path, 'espeak-ng-data')
    os.environ['PHONEMIZER_ESPEAK_PATH'] = espeak_path
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = espeak_library
    os.environ['ESPEAK_DATA_PATH'] = espeak_data_path

import gradio as gr
import torch
import time
import yaml
import multiprocessing
import shutil
import datetime
import glob
import webbrowser
import socket
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from io import BytesIO
import re
import nltk.data

from mutagen import wave
from mutagen.wave import WAVE
from mutagen.id3 import TXXX

from styletts2.utils import *
from modules.tortoise_dataset_tools.dataset_whisper_tools.dataset_maker_large_files import *
from modules.tortoise_dataset_tools.dataset_whisper_tools.combine_folders import *
from Utils.splitcombine import split_and_recombine_text

# Path to the settings file
SETTINGS_FILE_PATH = "Configs/generate_settings.yaml"
APP_SETTINGS_FILE_PATH = "Configs/app_settings.yaml"
GENERATE_SETTINGS = {}
TRAINING_DIR = "training"
BASE_CONFIG_FILE_PATH = r"Configs\template_config_ft.yml"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]
VALID_AUDIO_EXT = [
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".opus"
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_phonemizer = None
model = None
model_params = None
sampler = None
textcleaner = None
to_mel = None
params_whole = None
loaded = False
ref_s = None
appSettings = {}
generateSettings = {}
genData = {}
genHistory = {}
historyFileList = []

def load_all_models(model_path):
    global global_phonemizer, model, model_params, sampler, textcleaner, to_mel, params_whole
    
    model_config = (get_model_configuration(model_path))
    if not model_config:
        return None
    
    config = load_configurations(model_config)
    
    
    sigma_value = config['model_params']['diffusion']['dist']['sigma_data']
    
    model, model_params = load_models_webui(sigma_value, device)
    global_phonemizer = load_phonemizer()
    
    sampler = create_sampler(model)
    textcleaner = TextCleaner()
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    
    params_whole = load_pretrained_model(model, model_path=model_path)

    return {'global_phonemizer':global_phonemizer,'model':model,'model_params':model_params,'sampler':sampler,'textcleaner':textcleaner,'to_mel':to_mel,'params_whole':params_whole}

def unload_all_models():
    global global_phonemizer, model, model_params, sampler, textcleaner, to_mel, params_whole, ref_s

    if global_phonemizer:
        del global_phonemizer
        global_phonemizer = None
        print("Unloaded phonemizer")

    if model:
        del model
        model = None
        print("Unloaded model")

    if model_params:
        del model_params
        model_params = None
        print("Unloaded model params")

    if sampler:
        del sampler
        sampler = None
        print("Unloaded sampler")

    if textcleaner:
        del textcleaner
        textcleaner = None
        print("Unloaded textcleaner")

    if to_mel:
        del to_mel
        to_mel = None
        print("Unloaded to_mel")

    if params_whole:
        del params_whole
        params_whole = None
        print("Unloaded params_whole")

    if ref_s:
        # del ref_s
        ref_s = None
        print("Unloaded ref_s")

    do_gc()
    torch.cuda.empty_cache()

    gr.Info("All models unloaded.")

def do_gc():
    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        pass
    
def get_file_path(root_path, voice, file_extension, error_message):
    model_path = os.path.join(root_path, voice)
    if not os.path.exists(model_path):
        raise gr.Error(f'No {file_extension} located in "{root_path}" folder')

    for file in os.listdir(model_path):
        if file.endswith(file_extension):
            return os.path.join(model_path, file)
    
    raise gr.Error(error_message)

def get_model_configuration(model_path):
    base_directory, _ = os.path.split(model_path)
    for file in os.listdir(base_directory):
        if file.endswith(".yml"):
            configuration_path = os.path.join(base_directory, file)
            return configuration_path
    
    raise gr.Error("No configuration file found in the model folder")
    
def load_voice_model(voice):
    return get_file_path(root_path="models", voice=voice, file_extension=".pth", error_message="No TTS model found in specified location")


def generate_audio(text, voice, reference_audio_file, seed, alpha, beta, diffusion_steps, embedding_scale, voice_model, audio_opt_path=None, voices_root="voices",):
    global generateSettings, ref_s, loaded
    
    original_seed = int(seed)
    reference_audio_path = os.path.join(voices_root, voice, reference_audio_file)
    # noise = torch.randn(1, 1, 256).to(device)
    start = time.time()
    if original_seed==-1:
        seed_value = random.randint(0, 2**32 - 1)
    else:
        seed_value = original_seed
    set_seeds(seed_value)
    
    mean, std = -4, 4
    if ref_s == None or reference_audio_file != generateSettings['reference_audio_file'] or voice != generateSettings['voice']:
        ref_s = compute_style(reference_audio_path, model, to_mel, mean, std, device)
    # ref_s = compute_style(path, model, to_mel, mean, std, device)

    texts = split_and_recombine_text(text)
    audios = []

    print(f'seed: {seed_value}\nalpha: {alpha}\nbeta: {beta}\nembedding_scale: {embedding_scale}\ndiffusion_steps: {diffusion_steps}\nvoice: {voice}\nvoice_model: {voice_model}')

    combined_audio = AudioSegment.empty()
    
    for t in texts:
        # print(f'Generating: {t}')
        bytes_wav = bytes()
        byte_io = BytesIO(bytes_wav)
        # bytes_wav.seek(0)
        write(byte_io, 24000, inference(t, ref_s, model, sampler, textcleaner, to_mel, device, model_params, global_phonemizer=global_phonemizer, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale))
        audio_segment = AudioSegment.from_wav(byte_io)
        audio_segment = audio_segment[:-80]
        audio_segment = audio_segment + 1 # make it a bit louder
        combined_audio += audio_segment
        # audios.append(inference(t, ref_s, model, sampler, textcleaner, to_mel, device, model_params, global_phonemizer=global_phonemizer, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale))

    rtf = (time.time() - start)
    print(f"RTF = {rtf:5f}")
    
    print(f"{voice} Synthesized:")
    
    genDate = datetime.datetime.now()
    genDateLocal = datetime.datetime.strptime(genDate.strftime("%c"), '%a %b %d %H:%M:%S %Y')
    genDateReadable = genDateLocal.strftime("%m-%d-%y_%H-%M-%S")
    
    os.makedirs(os.path.join(".","results",f"{voice}"), exist_ok=True)
    audio_opt_path = os.path.join("results", f"{voice}", f"{voice}-{genDateReadable}.wav")

    generateSettings = {
        "text": text,
        "voice": voice,
        "reference_audio_file": reference_audio_file,
        "seed": original_seed if original_seed == -1 else seed_value,
        "alpha": alpha,
        "beta": beta,
        "diffusion_steps": diffusion_steps,
        "embedding_scale": embedding_scale,
        "voice_model" : voice_model
    }
        
    save_settings(generateSettings)
    
    # write(audio_opt_path, 24000, np.concatenate(audios))
    

    audio_opt_dir = os.path.dirname(audio_opt_path)
    audio_opt_filename = os.path.basename(audio_opt_path)
    output_file_path = os.path.join(audio_opt_dir, audio_opt_filename)
    output_wav_path = os.path.join(audio_opt_dir,f'{os.path.splitext(os.path.basename(output_file_path))[0]}.wav')
    output_mp3_path = os.path.join(audio_opt_dir,f'{os.path.splitext(os.path.basename(output_file_path))[0]}.mp3')

    combined_audio.export(output_wav_path, format='wav')

    if appSettings['enableID3tagging']:
        ID3Tags = {
            'seed':seed_value,
            'original_seed':original_seed,
            'alpha':alpha,
            'beta':beta,
            'diffusion_steps':diffusion_steps,
            'embedding_scale':embedding_scale,
            'reference_audio_path':reference_audio_path,
            'voice':voice,
            'voice_model':voice_model,
            'rtf':f'{rtf:5f}',
            'text':text,
            'date_generated': genDateLocal.strftime("%Y-%m-%d, %H:%M:%S")
        }

        tagWAV(output_wav_path,ID3Tags)

        fileList,genHistoryArray = getGenHistory()

    return audio_opt_path, [[seed_value]], genHistoryArray if appSettings['enableID3tagging'] else None
    

def train_model(data):
    return f"Model trained with data: {data}"

def update_settings(setting_value):
    return f"Settings updated to: {setting_value}"

def get_folder_list(root):
    folder_list = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
    return folder_list

def get_reference_audio_list(voice_name, root="voices"):
    reference_directory_list = os.listdir(os.path.join(root, voice_name))
    return reference_directory_list

def get_voice_models():
    folders_to_browse = ["training", "models"]
    
    model_list = []

    for folder in folders_to_browse:
        # Construct the search pattern
        search_pattern = os.path.join(folder, '**', '*.pth')
        # Use glob to find all matching files, recursively search in subfolders
        matching_files = glob.glob(search_pattern, recursive=True)
        # Extend the model_list with the found files
        model_list.extend(matching_files)
        
    return model_list
        
    
def update_reference_audio(voice):
    return gr.Dropdown(choices=get_reference_audio_list(voice), value=get_reference_audio_list(voice)[0])

def update_voice_model(model_path):
    gr.Info("Wait for models to load...")
    # model_path = get_models_path(voice, model_name)
    path_components = model_path.split(os.path.sep)
    voice = path_components[1]
    loaded_check = load_all_models(model_path=model_path)
    # if loaded_check:
    #     raise gr.Warning("No model or model configuration loaded, check model config file is present")
    gr.Info("Models finished loading")

def get_models_path(voice, model_name, root="models"):
    return os.path.join(root, voice, model_name)

def update_voice_settings(voice):
    try:
        # gr.Info("Wait for models to load...")
        # model_name = get_voice_models(voice)    
        # model_path = get_models_path(voice, model_name[0])   
        # loaded_check = load_all_models(model_path=model_path)
        # if loaded_check == None:
        #     gr.Warning("No model or model configuration loaded, check model config file is present")
        ref_aud_path = update_reference_audio(voice)
        
        # gr.Info("Models finished loading")
        return ref_aud_path #gr.Dropdown(choices=model_name, value=model_name[0] if model_name else None)
    except:
        gr.Warning("No models found for the chosen voice chosen, new models not loaded")
        ref_aud_path = update_reference_audio(voice)
        return ref_aud_path, gr.Dropdown(choices=[]) 

def load_settings(settingsType='generate'):
    if settingsType  == 'generate':
        global generateSettings, loaded
        try:
            with open(SETTINGS_FILE_PATH, "r") as f:
                generateSettings = yaml.safe_load(f)
                loaded = True
                return generateSettings
        except FileNotFoundError:
            if reference_audio_list:
                reference_file = reference_audio_list[0]
            else:
                reference_file = None
            if voice_list_with_defaults:
                voice = voice_list_with_defaults[0]
            else:
                voice = None
            
            settings_list = {
                "text": "Inferencing with this sentence, just to make sure things work!",
                "voice": voice,
                "reference_audio_file": reference_file,
                "seed" : "-1",
                "alpha": 0.3,
                "beta": 0.7,
                "diffusion_steps": 30,
                "embedding_scale": 1.0,
                "voice_model" : "models\pretrain_base_1\epochs_2nd_00020.pth"
            }
            return settings_list
        
    if settingsType == 'app':
        global appSettings
        try:
            with open(APP_SETTINGS_FILE_PATH, "r") as f:
                appSettings = yaml.safe_load(f)
                return appSettings
        except FileNotFoundError:
            appSettings = {
                "enableID3tagging": True
            }
            save_settings(appSettings,'app')
            return appSettings

def save_settings(settings,settingsType='generation'):
    if settingsType == 'generation':
        with open(SETTINGS_FILE_PATH, "w") as f:
            yaml.safe_dump(settings, f)
    if settingsType == 'app':
        with open(APP_SETTINGS_FILE_PATH, "w") as f:
            yaml.safe_dump(settings, f)
        
def update_button_proxy():
    voice_list_with_defaults = get_voice_list(append_defaults=True)
    datasets_list = get_voice_list(get_voice_dir("datasets"), append_defaults=True)
    train_list = get_folder_list(root="training")
    return gr.Dropdown(choices=voice_list_with_defaults), gr.Dropdown(choices=datasets_list), gr.Dropdown(choices=train_list), gr.Dropdown(choices=train_list)

def update_data_proxy(voice_name):
    train_data = os.path.join(TRAINING_DIR, voice_name,"train_phoneme.txt")
    val_data = os.path.join(TRAINING_DIR, voice_name, "validation_phoneme.txt")
    root_path = os.path.join(TRAINING_DIR, voice_name, "audio")
    return gr.Textbox(train_data), gr.Textbox(val_data), gr.Textbox(root_path)

def save_yaml_config(config,  voice_name):
    os.makedirs(os.path.join(TRAINING_DIR, voice_name), exist_ok=True)  # Create the output directory if it doesn't exist
    output_file_path = os.path.join(TRAINING_DIR, voice_name, f"{voice_name}_config.yml")
    with open(output_file_path, 'w') as file:
        yaml.dump(config, file)
        
def update_config(voice_name, save_freq, log_interval, epochs, batch_size, max_len, pretrained_model, load_only_params, F0_path, ASR_config, ASR_path, PLBERT_dir, train_data, val_data, root_path, diff_epoch, joint_epoch):
    with open(BASE_CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["log_dir"] = os.path.join(TRAINING_DIR, voice_name, "models")
    config["save_freq"] = save_freq
    config["log_interval"] = log_interval
    config["epochs"] = epochs
    config["batch_size"] = batch_size
    config["max_len"] = max_len
    config["pretrained_model"] = pretrained_model
    config["load_only_params"] = load_only_params
    config["F0_path"] = F0_path
    config["ASR_config"] = ASR_config
    config["ASR_path"] = ASR_path
    config["PLBERT_dir"] = PLBERT_dir
    config["data_params"]["train_data"] = train_data
    config["data_params"]["val_data"] = val_data
    config["data_params"]["root_path"] = root_path
    config["loss_params"]["diff_epoch"] = diff_epoch
    config["loss_params"]["joint_epoch"] = joint_epoch

    save_yaml_config(config, voice_name=voice_name)
    return "Configuration updated successfully."

def get_dataset_continuation(voice):
    try:
        training_dir = f"training/{voice}/processed"
        if os.path.exists(training_dir):
            processed_dataset_list = [folder for folder in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, folder))]
            if processed_dataset_list:
                processed_dataset_list.append("")
                return gr.Dropdown(choices=processed_dataset_list, value="", interactive=True)
    except Exception as e:
        print(f"Error getting dataset continuation: {str(e)}")
    return gr.Dropdown(choices=[], value="", interactive=True)

def load_whisper_model(language=None, model_name=None, progress=None):
    import whisperx
    # import whisper
    if torch.cuda.is_available():
        device = "cuda" 
    else:
        raise gr.Error("Non-Nvidia GPU detected, or CUDA not available")
    try:
        whisper_model = whisperx.load_model(model_name, device, download_root="whisper_models", compute_type="float16")
    except Exception as e: # for older GPUs
        print(f"Debugging info: {e}")
        whisper_model = whisperx.load_model(model_name, device, download_root="whisper_models", compute_type="int8")
    # whisper_align_model = whisperx.load_align_model(model_name="WAV2VEC2_ASR_LARGE_LV60K_960H" if language=="en" else None, language_code=language, device=device)
    print("Loaded Whisper model")
    return whisper_model

def get_training_folder(voice) -> str:
    '''
    voice(str) : voice to retrieve training folder from
    '''
    return f"./training/{voice}"

# Pretty much taken from the AI-voice-cloning repo for the code I implemented
def transcribe_other_language_proxy(voice, language, chunk_size, continuation_directory, align, rename, num_processes, keep_originals, 
                                    srt_multiprocessing, ext, speaker_id, whisper_model, progress=gr.Progress(track_tqdm=True)):

    whisper_model = load_whisper_model(language=language, model_name=whisper_model)
    num_processes = int(num_processes)
    training_folder = get_training_folder(voice)
    processed_folder = os.path.join(training_folder,"processed")
    dataset_dir = os.path.join(processed_folder, "run")
    merge_dir = os.path.join(dataset_dir, "dataset/wav_splits")
    audio_dataset_path = os.path.join(merge_dir, 'audio')
    train_text_path = os.path.join(dataset_dir, 'dataset/train.txt')
    validation_text_path = os.path.join(dataset_dir, 'dataset/validation.txt')
    
    large_file_num_processes = int(num_processes/2) # Used for instances where larger files are being processed, as to not run out of RAM
    
    items_to_move = [audio_dataset_path, train_text_path, validation_text_path]
    
    for item in items_to_move:
        if os.path.exists(os.path.join(training_folder, os.path.basename(item))):
            raise gr.Error(f'Remove ~~train.txt ~~validation.txt ~~audio(folder) from "./training/{voice}" before trying to transcribe a new dataset. Or click the "Archive Existing" button')
            
    if continuation_directory:
        dataset_dir = os.path.join(processed_folder, continuation_directory)

    elif os.path.exists(dataset_dir):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_dataset_dir = os.path.join(processed_folder, f"run_{current_datetime}")
        os.rename(dataset_dir, new_dataset_dir)

    from modules.tortoise_dataset_tools.audio_conversion_tools.split_long_file import get_duration, process_folder
    chosen_directory = os.path.join("./datasets", voice)
    items = [item for item in os.listdir(chosen_directory) if os.path.splitext(item)[1].lower() in VALID_AUDIO_EXT]

    # This is to prevent an error below when processing "non audio" files.  This will occur with other types, but .pth should
    # be the only other ones in the voices folder.
    # for file in items:
    #     if file.endswith(".pth"):
    #         items.remove(file)
    
    # In case of sudden restart, removes this intermediary file used for rename
    for file in items:
        if "file___" in file:
            os.remove(os.path.join(chosen_directory, file))
    
    file_durations = [get_duration(os.path.join(chosen_directory, item)) for item in items if os.path.isfile(os.path.join(chosen_directory, item))]
    progress(0.0, desc="Splitting long files")
    if any(duration > 3600*2 for duration in file_durations):
        process_folder(chosen_directory, large_file_num_processes)
    
    if not keep_originals:
        originals_pre_split_path = os.path.join(chosen_directory, "original_pre_split")
        try:
            shutil.rmtree(originals_pre_split_path)
        except:
            # There is no directory to delete
            pass
            
    progress(0.0, desc="Converting to MP3 files") # add tqdm later
    
    if ext=="mp3":
        import modules.tortoise_dataset_tools.audio_conversion_tools.convert_to_mp3 as c2mp3
        
        # Hacky way to get the functions working without changing where they output to...
        for item in os.listdir(chosen_directory):
            if os.path.isfile(os.path.join(chosen_directory, item)):
                original_dir = os.path.join(chosen_directory, "original_files")
                if not os.path.exists(original_dir):
                    os.makedirs(original_dir)
                item_path = os.path.join(chosen_directory, item)
                try:
                    shutil.move(item_path, original_dir)
                except:
                    os.remove(item_path)
        
        try:
            c2mp3.process_folder(original_dir, large_file_num_processes)
        except:
            raise gr.Error('No files found in the voice folder specified, make sure it is not empty.  If you interrupted the process, the files may be in the "original_files" folder')
        
        # Hacky way to move the files back into the main voice folder
        for item in os.listdir(os.path.join(original_dir, "converted")):
            item_path = os.path.join(original_dir, "converted", item)
            if os.path.isfile(item_path):
                try:
                    shutil.move(item_path, chosen_directory)
                except:
                    os.remove(item_path)
            
    if not keep_originals:
        originals_files = os.path.join(chosen_directory, "original_files")
        try:
            shutil.rmtree(originals_files)
        except:
            # There is no directory to delete
            pass

    progress(0.4, desc="Processing audio files")
    
    process_audio_files(base_directory=dataset_dir,
                        language=language,
                        audio_dir=chosen_directory,
                        chunk_size=chunk_size,
                        no_align=align,
                        rename_files=rename,
                        num_processes=num_processes,
                        whisper_model=whisper_model,
                        srt_multiprocessing=srt_multiprocessing,
                        ext=ext,
                        speaker_id=speaker_id,
                        sr_rate=24000
                        )
    progress(0.7, desc="Audio processing completed")

    progress(0.7, desc="Merging segments")
    merge_segments(merge_dir)
    progress(0.9, desc="Segment merging completed")

    try:
        for item in items_to_move:
            if os.path.exists(os.path.join(training_folder, os.path.basename(item))):
                print("Already exists")
            else:
                shutil.move(item, training_folder)
        shutil.rmtree(dataset_dir)
    except Exception as e:
        raise gr.Error(e)
        
    progress(1, desc="Transcription and processing completed successfully!")

    return "Transcription and processing completed successfully!"

def phonemize_files(voice, progress=gr.Progress(track_tqdm=True)):
    training_root = get_training_folder(voice)
    train_text_path = os.path.join(training_root, "train.txt")
    train_opt_path = os.path.join(training_root, "train_phoneme.txt")
    validation_text_path = os.path.join(training_root, "validation.txt")
    validation_opt_path = os.path.join(training_root, "validation_phoneme.txt")
    # Hardcoded to "both" to stay consistent with the train_to_phoneme.py script and not having to modify it
    option = "both"
    
    from modules.styletts2_phonemizer.train_to_phoneme import process_file
    
    progress(0.0, desc="Train Phonemization Starting")
    process_file(train_text_path, train_opt_path, option)
    progress(0.9, desc="Validation Phonemization Starting")
    process_file(validation_text_path, validation_opt_path, option)
    
    return "Phonemization complete!"
    
def archive_dataset(voice):
    training_folder = get_training_folder(voice)
    archive_root = os.path.join(training_folder,"archived_data")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_folder = os.path.join(archive_root,current_datetime)
    
    items_to_move = ["train.txt", "validation.txt", "audio", "train_phoneme.txt", "validation_phoneme.txt"]
    training_folder_contents = os.listdir(training_folder)

    if not any(item in training_folder_contents for item in items_to_move):
        raise gr.Error("No files to move")
    
    for item in items_to_move:
        os.makedirs(archive_folder, exist_ok=True)
        move_item_path = os.path.join(training_folder, item)
        if os.path.exists(move_item_path):
            try:
                shutil.move(move_item_path, archive_folder)
            except:
                raise gr.Error(f'Close out of any windows using where "{item} is located!')
    
    gr.Info('Finished archiving files to "archived_data" folder')

voice_list_with_defaults = get_voice_list(append_defaults=True)
datasets_list = get_voice_list(get_voice_dir("datasets"), append_defaults=True)
if voice_list_with_defaults:
    reference_audio_list = get_reference_audio_list(voice_list_with_defaults[0])
    train_list = get_folder_list(root="training")
else:
    reference_audio_list = None
    voice_list_with_default = None
    train_list = None
    
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) == 0

def tagWAV(filepath,newtags={}):
    '''
        Basic and easy implementation to tag generated files
        intention is to be able to inject generation settings
        into each wav, enabling easy setting repeats similar to
        that seen in Automatic1111 for SD.
        ex: tagWAV(filepath,{'seed':1806341205,'beta':0.2}))
        generation history done
        todo: implement "send to generation tab" functionality.
    '''

    if len(newtags) == 0:
        print('No tags provided')
        return

    wav = WAVE(filepath)
    id3 = wav.tags

    try:
        wave.WAVE.add_tags(wav)
        wav.save(filepath, v2_version=3)
        id3 = wav.tags
    except Exception as e:
        print(f"Error adding tag: {e}")

    try:
        for t in newtags:
            print(f'Adding {t}: {newtags[t]}')
            id3.add(TXXX(encoding=3, desc=t, text=str(newtags[t])))
            # id3.add(TXXX(encoding=3, desc="Alpha", text="1"))
    except Exception as e:
        print(f"Error tagging wav: {e}")

    separator = '/'
    id3.save(filepath, v23_sep=separator)
    wav = wave.WAVE(filepath)
    id3 = wav.tags

    for t in newtags:
        t = f'TXXX:{t}'
        print(f'{t}: {id3[t]}')

    return True

def getWAVtags(filepath):
    '''
        return all relevant tags from a wav file
    '''
    try:
        wav = wave.WAVE(filepath)
        id3 = wav.tags
    except Exception as e:
        print(f"Error loading wav tags: {e}")
        return False
    
    neededTags = ['voice','seed','original_seed','alpha','beta','diffusion_steps','embedding_scale','reference_audio_path','voice_model','rtf','text','date_generated']
    returnTags = {}

    for t in neededTags:
        tx = f'TXXX:{t}'
        # print(f'{t}: {id3[tx]}')
        returnTags[t] = id3[tx]

    return returnTags

def getGenHistory():
    '''
        Iterates over results folder and subfolders for .wav/.mp3 files that have ID3 tags generated by this app
        creates a set of lists to be run through np.column_stack to populate the dataframe on the history page with
    '''
    global historyFileList
    global genHistory

    fVoice = []
    fSeed = []
    fAlpha = []
    fBeta = []
    fSteps = []
    fScale = []
    fDate = []
    historyFileIter = 0
    historyFileList = []
    genHistory = {}

    tmp_historyFileList = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk('results') for f in filenames if f.endswith('.wav') or f.endswith('.mp3')]

    for f in tmp_historyFileList:
        ftags = getWAVtags(f)
        if ftags:
            #  print(ftags)
            historyFileList.append(f)
            fVoice.append(ftags['voice'][0])
            fDate.append(ftags['date_generated'][0])
            fSeed.append(ftags['seed'][0])
            fAlpha.append(ftags['alpha'][0])
            fBeta.append(ftags['beta'][0])
            fSteps.append(ftags['diffusion_steps'][0])
            fScale.append(ftags['embedding_scale'][0])

            genHistory[historyFileIter] = {
                'filepath':f,
                'voice_model':ftags['voice_model'][0],
                'voice':ftags['voice'][0],
                'reference_audio_path':ftags['reference_audio_path'][0],
                'date_generated':ftags['date_generated'][0],
                'seed':ftags['seed'][0],
                'original_seed':ftags['original_seed'][0],
                'alpha':ftags['alpha'][0],
                'beta':ftags['beta'][0],
                'diffusion_steps':ftags['diffusion_steps'][0],
                'embedding_scale':ftags['embedding_scale'][0],
                'rtf':ftags['rtf'][0],
                'text':ftags['text'][0],
            }

            historyFileIter += 1

    genHistoryArray = np.column_stack([fVoice,fSeed,fAlpha,fBeta,fSteps,fScale,fDate])

    # historyFileList = tmp_historyFileList
    recordCount = len(historyFileList)
    return recordCount,genHistoryArray

def populateGenHistoryData(value, evt: gr.EventData, sel: gr.SelectData):
    '''
        Grabs necessary info for displaying the clicked history record
    '''
    audioReturn = historyFileList[sel.index[0]]
    textReturn = f"{genHistory[sel.index[0]]['text']}"
    settingsReturn = np.array(list({k:v for k,v in genHistory[sel.index[0]].items() if k not in ['text']}.items()))

    return audioReturn,textReturn,settingsReturn
                
def UpdateMetadataFlag(value):
    appSettings['enableID3tagging'] = value
    save_settings(appSettings,settingsType='app')

def filterHistoryDF(filterHistoryVoiceSelection,filterHistoryEnteredText):
    fVoice,fSeed,fAlpha,fBeta,fSteps,fScale,fDate = [],[],[],[],[],[],[]

    for i in genHistory:
        ftags = genHistory[i]
        print(ftags['filepath'])
        #  print(ftags)
        if (ftags['voice'] in filterHistoryVoiceSelection or 'All' in filterHistoryVoiceSelection or not filterHistoryVoiceSelection or len(filterHistoryVoiceSelection) == 0):
            if all(substring.lower() in ftags['text'].lower() for substring in filterHistoryEnteredText) or len(filterHistoryEnteredText) == 0:     
                historyFileList.append(ftags['filepath'])
                fVoice.append(ftags['voice'])
                fDate.append(ftags['date_generated'])
                fSeed.append(ftags['seed'])
                fAlpha.append(ftags['alpha'])
                fBeta.append(ftags['beta'])
                fSteps.append(ftags['diffusion_steps'])
                fScale.append(ftags['embedding_scale'])

    genHistoryArray = np.column_stack([fVoice,fSeed,fAlpha,fBeta,fSteps,fScale,fDate])
    recordCount = len(historyFileList)

    return genHistoryArray
    

def main():
    global appSettings
    initial_settings = load_settings()
    appSettings = load_settings(settingsType='app')
    if voice_list_with_defaults:
        load_all_models(initial_settings["voice_model"])
        
        ref_audio_file_choices = get_reference_audio_list(initial_settings["voice"])
    else:
        # list_of_models = None
        ref_audio_file_choices = None

    with gr.Blocks(css=".gradio-container-4-38-1 table {height: 250px; width: 100%;overflow-x: hidden !important;; overflow-y: scroll !important; scrollbar-width: thin !important; padding-right: 17px !important; box-sizing: content-box !important;} ::-webkit-scrollbar {display: none;}") as demo:
        with gr.Tabs():
            with gr.TabItem("Generation"):
                with gr.Column():
                    with gr.Row():
                        GENERATE_SETTINGS["text"] = gr.Textbox(label="Input Text", value=initial_settings["text"])
                    with gr.Row():
                        with gr.Column():
                            GENERATE_SETTINGS["voice"] = gr.Dropdown(
                                choices=voice_list_with_defaults, label="Voice", type="value", value=initial_settings["voice"])
                            
                            
                            
                            GENERATE_SETTINGS["reference_audio_file"] = gr.Dropdown(
                                choices=ref_audio_file_choices, label="Reference Audio", type="value", value=initial_settings["reference_audio_file"]
                            )
                        with gr.Column():
                            GENERATE_SETTINGS["seed"] = gr.Textbox(
                                label="Seed", value=initial_settings["seed"]
                            )
                            GENERATE_SETTINGS["alpha"] = gr.Slider(
                                label="alpha", minimum=0, maximum=2.0, step=0.1, value=initial_settings["alpha"]
                            )
                            GENERATE_SETTINGS["beta"] = gr.Slider(
                                label="beta", minimum=0, maximum=2.0, step=0.1, value=initial_settings["beta"]
                            )
                            GENERATE_SETTINGS["diffusion_steps"] = gr.Slider(
                                label="Diffusion Steps", minimum=0, maximum=400, step=1, value=initial_settings["diffusion_steps"]
                            )
                            GENERATE_SETTINGS["embedding_scale"] = gr.Slider(
                                label="Embedding Scale", minimum=0, maximum=4.0, step=0.1, value=initial_settings["embedding_scale"]
                            )
                        with gr.Column():
                            generation_output = gr.Audio(label="Output")
                            seed_output = gr.Dataframe(
                                headers=["Seed"], 
                                datatype=["number"],
                                value=[], 
                                height=200,  
                                min_width=200  
                            )
                    with gr.Row():
                        update_button = gr.Button("Update Voices")
                        generate_button = gr.Button("Generate", variant="primary")
 
            with gr.TabItem("History"):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            selectedFilePlayer = gr.Audio(label="Player", show_label=False, interactive=False)
                            with gr.Accordion("Generation Settings", open=False):
                                generationHistorySettings = gr.Dataframe(
                                    headers=["Option", "Value"],
                                    datatype=["str", "str"],
                                    column_widths=["30%","70%"],
                                    row_count=12,
                                    col_count=(2, "fixed"),
                                    interactive=False,
                                    min_width="30px",
                                    )
                    
                    
                            with gr.Accordion("Generation Text", open=False):
                                generationHistoryText = gr.Markdown("")

                    recordCount,genHistoryArray = getGenHistory()

                    # grab voices for filter
                    voiceList = []
                    voiceList.append('All')
                    for i in genHistory:
                        if genHistory[i]['voice'] not in voiceList:
                            voiceList.append(genHistory[i]['voice'])

                    filterHistoryVoiceSelection = gr.Dropdown(voiceList, value="All", label="Voice Filter", multiselect=True)
                    filterHistoryEnteredText = gr.Dropdown(label="Text Filter", multiselect=True, allow_custom_value=True)

                    historyFiles = gr.Dataframe(
                        headers=["Voice", "Seed", "Alpha", "Beta", "Steps", "Scale", "Date"],
                        datatype=["str", "number", "number", "number", "number", "number", "date"],
                        column_widths=["50px","30px","30px","30px","30px","30px","90px"],
                        row_count=(0,'dynamic'),
                        col_count=(7, "fixed"),
                        interactive=False,
                        min_width="30px",
                        value=genHistoryArray
                        )
                    
                    historyFiles.select(populateGenHistoryData, inputs=[historyFiles], outputs=[selectedFilePlayer,generationHistoryText,generationHistorySettings])
                    
                    filterHistoryVoiceSelection.change(filterHistoryDF,
                                inputs=[filterHistoryVoiceSelection,filterHistoryEnteredText],
                                outputs=[
                                    historyFiles
                                ])
                    
                    
                    filterHistoryEnteredText.change(filterHistoryDF,
                                inputs=[filterHistoryVoiceSelection,filterHistoryEnteredText],
                                outputs=[
                                    historyFiles
                                ])

            with gr.TabItem("Generate Audiobook"):
                with gr.Column():
                    ebook_file = gr.File(label="eBook File",height=50)

                convert_btn = gr.Button("Convert to Audiobook", variant="primary")
                output = gr.Textbox(label="Conversion Status")
                download_btn = gr.Button("Download Audiobook Files")
                download_files = gr.File(label="Download Files", interactive=False)

                from Utils.ebookgenerator import convert_ebook_to_audio,download_audiobooks

                convert_btn.click(convert_ebook_to_audio,
                                    inputs=[ebook_file],
                                    outputs=[output]
                )

                download_btn.click(download_audiobooks,
                                    outputs=[download_files]
                )
                # ebook_file.upload(extract_metadata_and_cover, inputs=[ebook_file])

            with gr.TabItem("Training"):
                with gr.Tabs():
                    with gr.TabItem("Prepare Dataset"):
                        with gr.Column():
                            DATASET_SETTINGS = {}
                            EXEC_SETTINGS = {}
                            DATASET_SETTINGS['voice'] = gr.Dropdown(
                                choices=datasets_list, label="Dataset Source", type="value",value=datasets_list[0] if len(datasets_list) > 0 else "")
                            DATASET_SETTINGS['continue_directory'] = gr.Dropdown(
                                choices=[], label="Continuation Directory", value="", interactive=True
                            )
                            DATASET_SETTINGS['voice'].change(
                                fn=get_dataset_continuation,
                                inputs=DATASET_SETTINGS['voice'],
                                outputs=DATASET_SETTINGS['continue_directory'],
                            )
                            with gr.Row():
                                DATASET_SETTINGS['language'] = gr.Textbox(
                                    label="Language", value="en")
                                DATASET_SETTINGS['chunk_size'] = gr.Textbox(
                                    label="Chunk Size", value="15")
                                DATASET_SETTINGS['num_processes'] = gr.Textbox(
                                    label="Processes to Use", value=int(max(1, multiprocessing.cpu_count())-2))
                                
                            with gr.Row():
                                DATASET_SETTINGS['whisper_model'] = gr.Dropdown(
                                    WHISPER_MODELS, label="Whisperx Model", value="large-v3")
                                DATASET_SETTINGS['align'] = gr.Checkbox(
                                    label="Disable WhisperX Alignment", value=False   
                                )
                                DATASET_SETTINGS['rename'] = gr.Checkbox(
                                    label="Rename Audio Files", value=True
                                )
                                DATASET_SETTINGS['keep_originals'] = gr.Checkbox(
                                    label="Keep Original Files", value=True
                                )

                            advanced_toggle = gr.Button(value="Show Advanced Settings")

                            with gr.Row(visible=False) as advanced_settings_row:
                                DATASET_SETTINGS["srt_multiprocessing"] = gr.Checkbox(
                                    label="Disable if dataset files are < 20s", value=True
                                )
                                DATASET_SETTINGS["ext"] = gr.Dropdown(
                                    label="Audio Extension", value="wav", choices=["wav", "mp3"]
                                )
                                DATASET_SETTINGS["speaker_id"] = gr.Checkbox(
                                    label="Speaker ID", value=True
                                )
                            transcribe2_button = gr.Button(
                                value="Transcribe and Process")
                            
                            phonemize_button = gr.Button(
                                value="Run Phonemization")
                            
                            archive_button = gr. Button(
                                value="Archive Existing"
                            )
                        with gr.Column():
                            transcribe2_output = gr.Textbox(label="Progress Console")
                    
                    def toggle_advanced_settings(show):
                        if show == "Show Advanced Settings":
                            return gr.update(value="Hide Advanced Settings"), gr.update(visible=True)
                        else:
                            return gr.update(value="Show Advanced Settings"), gr.update(visible=False)
                    
                    advanced_toggle.click(
                        fn=toggle_advanced_settings,
                        inputs=[advanced_toggle],
                        outputs=[advanced_toggle, advanced_settings_row]
                    )
                    
                    archive_button.click(
                        archive_dataset,
                        inputs=[
                            DATASET_SETTINGS['voice']
                        ]
                    )                
                    
                    transcribe2_button.click(
                        transcribe_other_language_proxy,
                        inputs=[
                            DATASET_SETTINGS['voice'],
                            DATASET_SETTINGS['language'],
                            DATASET_SETTINGS['chunk_size'],
                            DATASET_SETTINGS['continue_directory'],
                            DATASET_SETTINGS["align"],
                            DATASET_SETTINGS["rename"],
                            DATASET_SETTINGS['num_processes'],
                            DATASET_SETTINGS['keep_originals'],
                            DATASET_SETTINGS["srt_multiprocessing"],
                            DATASET_SETTINGS['ext'],
                            DATASET_SETTINGS['speaker_id'],
                            DATASET_SETTINGS['whisper_model']
                        ],
                        outputs=transcribe2_output
                    )
                    
                    phonemize_button.click(
                        phonemize_files,
                        inputs=[
                            DATASET_SETTINGS["voice"]
                            ],
                        outputs=transcribe2_output
                    )
                    
                    with gr.TabItem("Generate Configuration"):
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    if train_list:
                                        training_dir = os.path.join(TRAINING_DIR, train_list[0])
                                        train_data_path = os.path.join(TRAINING_DIR, train_list[0], "train_phoneme.txt") if os.path.exists(os.path.join(training_dir, "train_phoneme.txt")) else ""
                                        val_data_path = os.path.join(TRAINING_DIR, train_list[0], "validation_phoneme.txt") if os.path.exists(os.path.join(training_dir, "validation_phoneme.txt")) else ""
                                        audio_data_path = os.path.join(TRAINING_DIR, train_list[0], "audio") if os.path.exists(os.path.join(TRAINING_DIR, train_list[0], "audio")) else ""
                                    else:
                                        train_data_path = None
                                        val_data_path = None
                                        audio_data_path = None
                                    with gr.Row():
                                        voice_name = gr.Dropdown(label="Voice Name", choices=train_list, value=train_list[0] if train_list else None, scale=6) 
                                        refresh_available_config_button = gr.Button(value="Refresh Available", scale=1)
                                        
                                    save_freq = gr.Slider(label="Save Frequency", minimum=1, maximum=1000, value=1, step=1)
                                    log_interval = gr.Slider(label="Log Interval", minimum=1, maximum=100, step=1, value=10)
                                    epochs = gr.Slider(label="Epochs", minimum=1, maximum=1000, step=1, value=500)
                                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=100, step=1, value=1)
                                    max_len = gr.Slider(label="Max Length", minimum=50, maximum=1000, step=10, value=160) # personal max, allows for generation during training without pushing into shared mem.
                                    
                                    list_of_models = get_voice_models()

                                    with gr.Row():
                                        pretrained_model = gr.Dropdown(
                                            choices=list_of_models, label="Pretrained Model", type="value", value=list_of_models[0], scale=6)
                                        refresh_models_available_button = gr.Button(
                                            value="Refresh Available"
                                        )
                                        
                                        def update_models():
                                            list_of_models = get_voice_models()
                                            return gr.Dropdown(choices=list_of_models, scale=1)
                                        
                                        refresh_models_available_button.click(fn=update_models,
                                                                            outputs=pretrained_model
                                        )
                                    load_only_params = gr.Checkbox(value=True, label="Load Only Params")
                                    
                                    diff_epoch = gr.Number(label="Diffusion Epoch", value=0)
                                    joint_epoch = gr.Number(label="Joint Epoch", value=0)
                                    
                                with gr.Column():
                                    F0_path = gr.Textbox(label="F0 Path", value=r"Utils\JDC\bst.t7")
                                    ASR_config = gr.Textbox(label="ASR Config", value=r"Utils\ASR\config.yml")
                                    ASR_path = gr.Textbox(label="ASR Path", value=r"Utils\ASR\epoch_00080.pth")
                                    PLBERT_dir = gr.Textbox(label="PLBERT Directory", value=r"Utils\PLBERT")
                                    train_data = gr.Textbox(label="Train Data", placeholder="Enter train data path", value=train_data_path)
                                    val_data = gr.Textbox(label="Validation Data", placeholder="Enter validation data path", value=val_data_path)
                                    root_path = gr.Textbox(label="Root Path", placeholder="Enter root path", value=audio_data_path)
                                    
                            update_config_button = gr.Button("Update Configuration")
                                
                            status_box = gr.Textbox(label="Update Status")    
                    
                    def get_training_config(voice):
                        config_path = os.path.join("training", voice, f"{voice}_config.yml")
                        return config_path
                    
                    
                    
                    def start_training_proxy(voice, progress=gr.Progress(track_tqdm=True)):
                        from styletts2.train_finetune_accelerate import main as run_train
                        config_path = get_training_config(voice)
                        
                        # Run training directly; tqdm handles the console progress
                        run_train(config_path)
                        return "Training Complete!"

                    def launch_tensorboard_proxy():
                        port = 6006
        
                        if is_port_in_use(port):
                            gr.Warning(f"Port {port} is already in use. Skipping TensorBoard launch.")
                        else:
                            subprocess.Popen(["launch_tensorboard.bat"], shell=True)
                            time.sleep(1)
                
                        webbrowser.open(f"http://localhost:{port}")
                        
                    with gr.TabItem("Run Training"):
                        with gr.Row():
                            with gr.Column():
                                training_voice_name = gr.Dropdown(label="Voice Name", choices=train_list, value=train_list[0] if train_list else None)
                                refresh_available_config_button_2 = gr.Button(value="Refresh Available")
                            with gr.Column():
                                training_console = gr.Textbox(label="Training Console")
                                start_train_button = gr.Button(value="Start Training")
                        with gr.Row():
                            launch_tensorboard_button = gr.Button(value="Launch Tensorboard")  
                    
            update_config_button.click(update_config, inputs=[
                                    voice_name, save_freq, log_interval, epochs, batch_size, max_len, pretrained_model,
                                    load_only_params, F0_path, ASR_config, ASR_path, PLBERT_dir, train_data, val_data,
                                    root_path, diff_epoch, joint_epoch], outputs=status_box)
            update_button.click(update_button_proxy,
                                outputs=[
                                    GENERATE_SETTINGS["voice"],
                                    DATASET_SETTINGS["voice"],
                                    voice_name,
                                    training_voice_name
                                ])
            
            refresh_available_config_button.click(update_button_proxy,
                                outputs=[
                                    GENERATE_SETTINGS["voice"],
                                    DATASET_SETTINGS["voice"],
                                    voice_name,
                                    training_voice_name
                                ])
            
            refresh_available_config_button_2.click(update_button_proxy,
                                outputs=[
                                    GENERATE_SETTINGS["voice"],
                                    DATASET_SETTINGS["voice"],
                                    voice_name,
                                    training_voice_name
                                ])
            
            start_train_button.click(start_training_proxy,
                                    inputs=[training_voice_name],
                                    outputs=[training_console]
                                    )
            launch_tensorboard_button.click(launch_tensorboard_proxy
                                            )
            
            voice_name.change(update_data_proxy,
                            inputs=voice_name,
                            outputs=[
                                train_data,
                                val_data,
                                root_path
                            ])
            
            with gr.TabItem("Settings"):
                list_of_models = get_voice_models()
                with gr.Row():
                    with gr.Column():
                        GENERATE_SETTINGS["voice_model"] = gr.Dropdown(
                            choices=list_of_models, interactive=True, label="Voice Models", type="value", value=initial_settings["voice_model"], scale=6)
                        refresh_models_available_button = gr.Button(
                            value="Refresh Models Available", scale=1, variant='primary')
                        unload_all_models_button = gr.Button(
                                value="Unload all loaded models")    
                    with gr.Column():
                        metadataFlagCheck = gr.Checkbox(value=appSettings['enableID3tagging'], interactive=True, label="Enable ID3 embedding")
                        with gr.Accordion("ID3 toggle further explanation", open=False):
                            gr.Markdown("Most media players don't have an issue playing WAV files with id3 tags, however some will see the file as corrupt. The history page is built around the ID3 tag embedding on generation so if this is disabled then there won't be any history.")
                
                def update_models():
                    list_of_models = get_voice_models()
                    return gr.Dropdown(choices=list_of_models)

                metadataFlagCheck.change(UpdateMetadataFlag,metadataFlagCheck)
                    
                
                refresh_models_available_button.click(fn=update_models,
                                                      outputs=GENERATE_SETTINGS["voice_model"]
                )
                
                unload_all_models_button.click(fn=unload_all_models)
                
                GENERATE_SETTINGS["voice_model"].change(fn=update_voice_model,
                                inputs=[GENERATE_SETTINGS["voice_model"]])
                
                GENERATE_SETTINGS["voice"].change(fn=update_voice_settings, 
                                        inputs=GENERATE_SETTINGS["voice"], 
                                        outputs=[GENERATE_SETTINGS["reference_audio_file"]]
                                                    )
                generate_button.click(generate_audio, 
                                        inputs=[GENERATE_SETTINGS["text"],
                                                GENERATE_SETTINGS["voice"],
                                                GENERATE_SETTINGS["reference_audio_file"],
                                                GENERATE_SETTINGS["seed"],
                                                GENERATE_SETTINGS["alpha"],
                                                GENERATE_SETTINGS["beta"],
                                                GENERATE_SETTINGS["diffusion_steps"],
                                                GENERATE_SETTINGS["embedding_scale"],
                                                GENERATE_SETTINGS["voice_model"]], 
                                        outputs=[generation_output, seed_output, historyFiles])
       
    webui_port = None         
    while webui_port == None:
        for i in range (7860, 7865):
            if is_port_in_use(i):
                print(f"Port {i} is in use, moving 1 up")
            else:
                webui_port = i
                break
    
    webbrowser.open(f"http://localhost:{webui_port}")
    demo.launch(server_name="0.0.0.0", server_port=webui_port)


    
if __name__ == "__main__":
    main()