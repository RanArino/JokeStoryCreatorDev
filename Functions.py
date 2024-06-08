# library install
import cv2
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from openai import OpenAI
import pandas as pd
from pathlib import Path
from PIL import Image
import pyperclip as clip
import random
import re
#import RRDBNet_arch as arch
import textwrap
import time
#import torch
import yaml


from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


from moviepy.editor import AudioClip, ImageClip, TextClip, ColorClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips, vfx
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# define global variables
OPTIONS = {
    # OPTIONS for "cultures"
    "culture": [
        "General",
        "North_American",
        "Mexican", 
        "Brazilian",
        "British",
        "French", 
        "German", 
        "Italian",
        "Spanish",
        "Australian",
        "Indian",
        "Russian",
        "Turkish", 
        "Iranian",
        "Nigerian", 
        "South_African",
        "Japanese",
        "Chinese",
        "Korean"
    ],
    # OPTIONS for "category"
    "category": [
        "One-liners", 
        "Puns",
        "Knock-knock jokes",
        "Dad jokes",
        "Parody jokes",
        "Absurdist humor",
        "Observational humor", 
        "Wordplay jokes",
        "Animal jokes", 
        "Food jokes",
        "Doctor jokes",  
        "School jokes", 
        "Sports jokes", 
        "Technology jokes", 
        "Travel jokes", 
        "Family jokes", 
        "Work jokes",
        "Historical jokes",
        "Science jokes", 
        "Math jokes"
    ],
}

class JokeStory:
    def __init__(self, path: str = 'joke_data.json'):
        # define column names
        self.joke_cols = ['id', 'culture', 'category', 'joke']
        self.story_cols = [
            'id', 'description', 'title', 'c1_name', 'c2_name', 'c1_gender', 'c2_gender',
            'n_1', 'n_2', 'n_3', 'n_4', 
            'c1_1', 'c2_1', 'c1_2', 'c2_2','c1_3', 'c2_3','c1_4', 'c2_4', 
            'script_flow', 'image_id']
        # load two dataset
        self.joke = pd.read_json(path)
        data_from_files = []
        # Iterating over the created JSON files and loading their content
        for file_name in os.listdir("Stories")[-10:]:
            with open(f'Stories/{file_name}', 'r') as file:
                data = json.load(file)
                data_from_files.append(data)

        # Creating a DataFrame from the loaded data
        self.story = pd.DataFrame(data_from_files)

        # Defone google api object
        self.google = GoogleAPI()


    def get_next_joke(self, id_: int = 0):
        # if 'id_' is assigned,
        if id_:
            joke_sent = self.joke.iloc[id_]['joke']

        # otherwise, get the next sentence of joke
        else:
            idx = len(os.listdir('Stories'))
            joke_sent = self.joke.iloc[idx]['joke']
        
        return joke_sent
    

    def joke_prompts(self, culture_opt: str = 'random', category_opt: str = 'random', number: int = 20):
        """
        Determine the randomly selected options.

        Attributes:
        - "culture_opt": user request about the cultureal choice; the defaul value is 'random'.
        - "category_opt": user request about the category choice; the defaul value is 'random'.
        - "number": user request about how many conbination will be generated.
        """
        # randomly select culture OPTIONS from the list
        if culture_opt == 'random':
            cultures = random.choices(OPTIONS['culture'], k=number)
        elif culture_opt == 'all':
            cultures = OPTIONS['culture'] * min(number, 3)
        # otherwise, simply repeat a specified culture name
        else:
            cultures = [culture_opt] * number

        # randomly select category OPTIONS from the list
        if category_opt == 'random':
            categories = random.choices(OPTIONS['category'], k=number)
        elif category_opt == 'all':
            categories = OPTIONS['category'] * min(number, 3)
        # otherwise, simply repeat a specified category name
        else:
            categories = [category_opt] * number

        len_cul, len_cat = len(cultures), len(categories)
        if len_cul != len_cat:
            if len_cul < len_cat:
                cultures.extend(cultures * ((len_cat - len_cul) // len_cul) + cultures[:((len_cat - len_cul) % len_cul)])
            else:
                categories.extend(categories * ((len_cul - len_cat) // len_cat) + categories[:((len_cul - len_cat) % len_cat)])


        # generate a list of prompts to generate jokes
        print("Activate 'Code 100'.")
        print("Return only code snippet of CSV format.")
        print("Make sure that each row of output data has three values; culture(str), category(str), and the generated joke(str).")
        print("Retrieve the existing jokes without **decoding** and **analyzing** them, and make sure no duplicated sentence of jokes.")
        print("")
        print("Here is the list of prompts; generate a creative and unique joke per prompts.")
        print([
            f"Generate a creative and unique joke based on '{r}' culture and '{c}' category."
            for r, c in zip(cultures, categories)
        ])


    def update_joke_data(self, row_csv: str):
        # parsing raw csv data
        data_lines = row_csv.strip().split('\n')
        parsed_data = []

        for line in data_lines:
            # Splitting by the first two commas only, as the third field (joke text) may contain commas
            fields = line.split(',', 2)
            if len(fields) == 3:
                # Stripping leading and trailing whitespaces and quotes
                fields = [field.strip().strip('"') for field in fields]
                parsed_data.append(fields)

        # convert to data frame
        new_data = pd.DataFrame(parsed_data, columns=["culture", "category", "joke"])
        # filter out the duplicated jokes
        filtered_new_data = new_data[~new_data['joke'].isin(self.joke['joke'])]
        # assign id number
        filtered_new_data['id'] =[self.joke.iloc[-1]['id'] + (i+1) for i in range(len(filtered_new_data))]
        # aggregate data frames
        self.joke = pd.concat([self.joke, filtered_new_data], ignore_index=True, sort=False)
        # load the aggregated data as the json file
        self.joke.to_json('joke_data.json', orient='records', indent=4)

        return self.joke
    

    def save_story_data(self, story_scripts: dict):
        """
        Save the given data as new JSON file.
        Dictionary includes description of joke, story title, narrations and dialogs.
        """
        # get the image id
        image_id = str(input("Type the image id here."))
        # add script flow if not assigned
        if not story_scripts.get('script_flow', False):
            story_scripts['script_flow'] = ["n_1", "c1_1", "c2_1", "n_2", "c1_2", "c2_2", "n_3", "c1_3", "c2_3", "n_4", "c1_4", "c2_4"]
        # update the story_scripts, 'id' and 'image_id
        story_scripts.update({
            #'id': len(os.listdir('Stories')),
            'id': int(self.joke.iloc[len(os.listdir('Stories'))]['id']),
            'image_id': image_id
        })
        # assign voice character
        for k in ['c1_gender', 'c2_gender']:
            gender = story_scripts[k]
            if gender != 'male' and gender != 'female':
                story_scripts[k] = ['male', 'female'].pop(random.randint(0, 1))

        # define file name
        file_name = f"Stories/{story_scripts['id']:03d}_{story_scripts['title'].replace(' ', '')}.json"
        with open(file_name, 'w') as file:
            json.dump(story_scripts, file, indent=4)

        # create the new data frame
        new_story = pd.DataFrame([story_scripts], columns=self.story_cols)

        # update the original data
        self.story = pd.concat([self.story, new_story], ignore_index=True)

        print("Success: Story Data Updated")

        return self.story
    

    def image_preprocess(self):
        # find the latest updated story
        story = self.story.dropna().iloc[-1]
        # desired image file name
        d_file = f"{story['id']:03d}_{story['title'].replace(' ', '')}"
        # d_file = f"{int(story['id']):03d}_{story['title'].replace(' ', '')}"

        # find original file name
        path = "C:/Users/runru/Downloads"
        files = [dir for dir in os.listdir(path) if dir.startswith('DALL·E')]
        if files:
            origin_file = f"{path}/{files[0]}"
        else:
            print("Fail: Image Preprocessing")
            return {"status": "fail", "message": "Nothing Files"}
            
        # rename file
        desire_file = f"C:/Users/runru/MyGPTs/JokeStoryCreatorDev/Images/{d_file}"
        # check no duplication
        if os.path.exists(desire_file):
            print("Fail: Image Preprocessing")
            return {"status": "fail", "message": "Deplication"}
        else:
            os.rename(origin_file, desire_file)
            # convert to png 
            self.convert_webp_to_png()
            print("Success: Image Preprocessing")
            return {"status": "success", "message": d_file}


    def convert_webp_to_png(self):
        # get the file names whose format is not png
        inputs = [f'Images/{file}' for file in os.listdir('Images') if not file.endswith('png')] #[f'Titles_Ends/{file}' for file in os.listdir('Titles_Ends') if not file.endswith('png')]
        if not inputs:
            return "Nothing Files"
        # get all file paths
        for path in inputs:
            # open image and convert to PNG file
            img = Image.open(path)
            img.save(path.split('.')[0] + '.png', 'PNG')
            # remove webp file
            os.remove(path) 


class Video:
    def __init__(self, JS, size: tuple = (1080, 1920)):
        self.size = size  # video size; (width, height)
        self.JS = JS  # JokeStory class

        # load api key
        with open('API/api_keys.yml', 'r') as file:
            key = yaml.safe_load(file)["openai"]
        
        self.client = OpenAI(api_key=key)  # OpenAI API

        # load RRDB resolution model
        #model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
        #self.device = torch.device('cpu')
        #self.model = arch.RRDBNet(3, 3, 64, 23, gc=32).to(self.device).eval()
        #self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)


    def generate_video(self, id_list: list = [], t_interval: int = 60, fps: int = 24, specials: dict = {}, crop_mode: dict = {}):
        # if crop_mode is blank, create default
        if not crop_mode:
            crop_mode = {int(i): 'auto' for i in id_list}

        # assign speciality
        self.specials = specials

        # Get the list of video files
        videos_files = os.listdir("Videos")

        if not id_list:
            return "Specify the id list"


        elif id_list == 'all':
            # get the list of files in the Images folder
            images_files = os.listdir("Images")
            # change to the set of file names (without file format)
            images_name = {name.split('.')[0] for name in images_files}
            videos_name = {name.split('.')[0] for name in videos_files}
            # get the file name that did not create the video
            create_video_name = list(images_name - videos_name)
            # redefine crop_mode
            crop_mode = {int(name.split("_")[0]): 'auto' for name in create_video_name}

        else:
            # scan the directory
            with os.scandir('Images') as entries:
                create_video_name = [entry.name.split('.')[0] for entry in entries 
                                    if any(entry.name.startswith(f'{i:03}_') for i in id_list)]        

        # track generated video names
        success_video_names = []
        # traversing all image sources to create videos
        for name in create_video_name:
            if f"{name}.mp4" in videos_files:
                print(f"{name} has already existed!")
                continue
            # get the index number
            id_ = int(name.split("_")[0])  # type: ignore
            # retrieve story
            with open(f"Stories/{name}.json", 'r') as file:
                story = json.load(file)
            
            # the main image (four-panel strip)
            image_path = f"Images/{name}.png"
            # get the cropped panels
            id_digit = int(name.split("_")[0])
            cropped_panels = self.auto_crop_comic_strip(image_path, mode=crop_mode[id_digit])

            # clips for each component
            clips = [
                self.title_clips(id_, story, cropped_panels[-1], 4., 1),
                self.scene_clips(id_, story, cropped_panels, 1., 300),
                self.end_clips(id_, cropped_panels[0], 5.)
            ]

            # get the background music clip
            music_clip = self.background_music(clips)

            # concatenate all the clips sequentially
            final_video = concatenate_videoclips(clips, method="compose", padding=-1)

            # reset the audio
            final_video = final_video.set_audio(
                CompositeAudioClip([final_video.audio, music_clip])
            ) 

            # loading video as mp4
            final_video.write_videofile(f"Videos/{name}.mp4", fps=fps, codec="libx264", audio_codec="aac")

            # add generated video name
            success_video_names.append(name)

            # define time intervala
            time.sleep(t_interval)

        return success_video_names


    def title_clips(self, id_: int, story: dict, img: np.ndarray, title_t: float = 5., trans_t: float = 1.):
        """
        Parameters
        - "id_": the index number of the joke sentence.
        - "title_t": duration of the title clips (except for 'trans_t')
        - "trans_t": transition time of fadeout.
        """
        # define the list of clips
        clips = []

        # (1): background image
        #  crop the character face
        crop_face = self.crop_important_part_to_scale(img.copy())
        #  reapply soft edge
        crop_face = self.apply_soft_edge(crop_face, 20)
        #  adjust color
        crop_face = self.img_adjust(crop_face, 1.2, 0.75)
        clips.append(ImageClip(crop_face)
                     .fx(vfx.resize, width=self.size[0])  # type: ignore
                     .set_position(("center", "center"))) 
           
        # (2) title
        #  get the base color of the title image
        base_color = self.get_base_color(img)
        outline_color = '#{:02x}{:02x}{:02x}'.format(*self.modify_base_color(base_color, white=0.50))
        title_clip = (TextClip(
            story['title'], font="Comic-Sans-MS-Bold", fontsize=80,
            color='black', stroke_color=outline_color, stroke_width=2)
            .set_position(("center", 600)))
        # title background
        bg_color = self.modify_base_color(base_color, 0.6)  # background color
        clips.append(ColorClip(size=(self.size[0], title_clip.size[1]+50), color=bg_color, ismask=False)
                     .set_position(('center', 600-25))
                     .set_opacity(0.8))
        # add text after background
        clips.append(title_clip)
        
        # (3) Description
        #  get the description from the story
        descript = story['description']
        descript_clip = (self.wrapped_textclip(f'"{descript}"', 1000, color='black', font="Comic-Sans-MS-Italic", fontsize=70, align='West')
                    .set_position(("center", 1000))
                    .set_opacity(0.9))
        #  define text background
        bg_color2 = self.modify_base_color(base_color, 0.8)
        clips.append(ColorClip(size=(self.size[0], descript_clip.h + 100), color=bg_color2, ismask=False)
                        .set_position(("center", 1000-50))  # 100/2 = 50
                        .set_opacity(0.7)
                    )
        # add text after abackground
        clips.append(descript_clip)

        # add unique contents
        if self.specials.get('title'):
            clips += [c.set_position(("center", "center")) for c in self.specials['title']]

        # add common features for all clips
        clips_ = [clip.set_duration(title_t).crossfadeout(trans_t) for clip in clips]
        # Composit all components in the title page
        title_final = CompositeVideoClip(clips_, size=self.size)
        
        return title_final


    def scene_clips(self, id_: int, story: dict, cropped_panels: list, trans_t: float = 1., narrate_h: int = 300):
        """
        Parameters:
        - "id_": id of the joke sentence.
        - "cropped_panels": a list of np.ndarray of cropped images (RGB)
        - "trans_t": transition time .
        - "narrate_h": the height for the narration section on the top of the image.
        """
        # list of all clips (each scene clip)
        clips = []
        # define the voice character; return {'n': 'echo', 'c1': 'alloy', 'c2': None}
        audio_dict = self.define_audio_type({k: story[f'{k}_gender'] for k in ['c1', 'c2']})
        # the main background image (every scene)
        #scene_bg = (ImageClip(self.img_adjust(f'Titles_Ends/{id_:03d}_title.png', 1.75, 0.25))
        #            .fx(vfx.resize, width=self.size[0])  # type: ignore
        #            .set_position(("center", "center"))
        #            .set_opacity(0.7))

        # traversing all cropped panels
        for i, panel in enumerate(cropped_panels):
            # track the timestep of each dialog
            time_step = [0]
            # initialize the position of height 
            pos_h = 150

            # (1): Image of each scene
            scene_img = (ImageClip(panel)
                         .fx(vfx.resize, width=self.size[0]-50, height=self.size[0]-50) # type: ignore
                        )   
                        # .set_position(("center", narrate_h+30)))  # narration height + 30px margin-top of image

            # define the clips for dialogues
            dialog_clips = {'text': [], 'audio': [self.silent_clip(trans_t*0.5)]}
            # traversing three dialogues in one scene
            for j, d_key in enumerate(story['script_flow'][i*3:(i*3)+3]):
                ### (1): Audio Clip for Reading Dialogs
                #  script role; 'n', 'c1', or 'c2'
                script_role = d_key.split("_")[0]
                #  audio character
                audio_char = audio_dict[script_role]
                #  cript content or dialog
                script = story[d_key]
                #  create audio here
                if audio_char:
                    file_name = d_key #f"{id_:03}_{d_key}"
                    audio_clip = self.create_openai_audio(audio_char, story[d_key], file_name, time_step, trans_t)
                    audio_duration = audio_clip.duration
                else:
                    audio_clip, audio_duration = self.silent_clip(1.5), 1.5

                ### (2) Text Clips and Image for Displaying Dialog
                #  set parameters
                textclip_params = {
                    'text': script if script_role == 'n' else story[f"{script_role}_name"] + ': \n' + script,
                    'width': 1000 if script_role == 'n' else 950,
                    'color': 'white',
                    'font': 'Comic-Sans-MS',
                    'fontsize': 55 if script_role == 'n' else 50,
                    'align': 'West'
                }
                #  get the wrapped TextClip
                text_clip = (
                    self.wrapped_textclip(**textclip_params)
                        .set_start(time_step[-1]+trans_t*0.5)
                        .fx(vfx.fadein, trans_t*0.5)  # type: ignore
                        .set_position(("center" if j == 0 else 75, pos_h))
                    )
                
                # define padding between the current and the next clip
                padding = 20 if j == 1 else 30 
                # update the height position for the next component
                pos_h += text_clip.h + padding

                # if nattation clip is set up, define image height position
                if j == 0:
                    # image of each scene
                    scene_img = (ImageClip(panel)
                                 .fx(vfx.resize, width=self.size[0]-50, height=self.size[0]-50) # type: ignore
                                 .set_position(("center", pos_h)))
                    # update the height for the next clip
                    pos_h += scene_img.h + padding
                 
                # keep adding dialogs
                dialog_clips['text'].append(text_clip)
                dialog_clips['audio'].append(audio_clip)
                # update the dialog time stamp for the next one
                blank = trans_t*0.75 if j == 2 else trans_t*0.5
                time_step += [time_step[-1] + audio_duration + blank]
                # add interval between audios correpondings to time_step
                dialog_clips['audio'].append(self.silent_clip(blank))
            
    
            # (4): Composited all dialog clips in each scene
            #  set the duration of each textclip
            adj_dialog_clips = [c.set_duration(time_step[-1]-time_step[idx]-0.5) for idx, c in enumerate(dialog_clips['text'])]
            #  concatenate the audios in each scene
            concat_audio_clips = concatenate_audioclips(dialog_clips['audio'] + [self.silent_clip(trans_t)])
            scene_dialogs = (CompositeVideoClip(adj_dialog_clips, size=self.size)
                            .set_audio(concat_audio_clips)
                            .set_duration(concat_audio_clips.duration))
            
            # Create each scene
            #  adjust duration of image
            scene_img = scene_img.set_duration(time_step[-1]+trans_t)
            clips.append(CompositeVideoClip([scene_img, scene_dialogs], size=self.size)
                         .set_duration(time_step[-1]+trans_t))

        # multiplied by two due to applying fade in and out
        clips_ = [clip.crossfadein(trans_t).crossfadeout(trans_t) for clip in clips]

        # Sequentially compose all the scene clips
        scene_final = concatenate_videoclips(clips_, method="compose", padding=-1)
        
        return scene_final


    def end_clips(self, id_: int, img: np.ndarray, end_t: float = 5., trans_t: int = 1):
        """
        Parameters
        - "id_": the index number of the joke sentence.
        - "end_t": duration of the clip
        """
        # define all clips
        clips = []
        # (1): Background Image
        #  crop the character face
        crop_face = self.crop_important_part_to_scale(img.copy())   
        #  reapply soft edge
        crop_face = self.apply_soft_edge(crop_face)
        #  change color
        crop_face = self.img_adjust(crop_face, 0.5, 1.25)
        clips.append(ImageClip(crop_face)
                            .fx(vfx.resize, width=self.size[0], height=self.size[1])  # type: ignore
                            .set_position(("center", "center"))
                            .set_opacity(0.8))
    
        # (2): Title
        clips.append(TextClip("Story Punchline", color='white', font="Comic-Sans-MS-Bold-Italic", fontsize=80)
                    .set_position(("center", 550)))

        # (3): Punchline
        joke = self.JS.joke[self.JS.joke['id'] == id_]['joke'].values[0]
        # get the wrapped text clip
        joke_clip = (self.wrapped_textclip(f'"{joke}"', 900, color='white', font="Comic-Sans-MS-Italic", fontsize=70, align='West')
                    .set_position(("center", 750)))
        # background
        clips.append(ColorClip(size=(self.size[0], joke_clip.h + 100), color=(0,0,0), ismask=False)
                        .set_position(("center", 750-50))  # 100/2 = 50
                        .set_opacity(0.5))
        # add sentence after the background
        clips.append(joke_clip)
        
        # (4): End note
        clips.append(TextClip("~ E N D ~", color="white", font="Comic-Sans-MS-Bold", fontsize=150)
                    .set_position(("center", min(750 + joke_clip.h + 250, 1600)))
                    .set_opacity(0.7)
                    .set_start(2)
                    .set_duration(end_t - 2)
                    .fx(vfx.fadein, 1))  # type: ignore

        # apply the common features
        clips_ = [clip.set_duration(end_t).crossfadein(trans_t) for clip in clips[:-1]] + [clips[-1]]
        end_final = CompositeVideoClip(clips_, size=self.size)

        return end_final
    

    def silent_clip(self, duration: float):
        """
        Simply return the silent audio clip
        """
        return AudioClip(lambda t: 0, duration=duration)


    def define_audio_type(self, gender: dict):
        """
        Determine the voice type for each character.

        define_audio_type({"c1": "male", "c2": "female"})
        >> {'n': 'onyx', 'c1': 'alloy', 'c2': 'shimmer'}
        """
        # define result
        result = {}
        # define options of audio
        audio_options = {"narrator": ["onyx", "echo"], "male": ["alloy", "fable"], "female": ["nova", "shimmer"]}

        # determine narration character
        result["n"] = audio_options['narrator'].pop(random.randint(0, 1))

        for k in gender.keys():
            # get 0 or 1 randomly
            rand_idx = random.randint(0, len(audio_options[gender[k]])-1)
            # choose voice character
            result[k] = audio_options.get(gender[k].lower(), [None, None]).pop(rand_idx)

        return result


    def create_openai_audio(self, audio_char: str, texts: str, file_name: str, time_step: list, trans_t: float):
        """
        Parameters:
        - "audio_char": audio character
        - "tests": input text to be read
        - "file_name": unique name of the audio file
        - "time_step": list of time step
        - "trans_t": transition time
        
        """
        # Initialize variables
        audio_clip = None
        # define file path
        audio_folder = Path(__file__).parent / "Audios"
        audio_folder.mkdir(exist_ok=True)
        file_path = audio_folder / f"{file_name}.mp3"

        # Generate audio data using OpenAI's API
        
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=audio_char,  # type: ignore
            response_format="mp3",  # Changed to AAC format
            input=texts,
        )

        # download the audio file
        response.stream_to_file(file_path)
        
        # clip the audio file
        audio_clip = (AudioFileClip(str(file_path))
                      .set_start(time_step[-1] + trans_t)
                      .volumex(1.2))

        return audio_clip


    def background_music(self, clips: list):
        # time step of each clip
        start_t = clips[0].duration
        main_t = clips[1].duration
        last_t = clips[2].duration

        # randomly choose the background music
        music_file = random.choice(os.listdir("Music"))
        # clip music
        music = (AudioFileClip(f"Music/{music_file}")
                .set_duration(start_t+main_t+last_t)
                .volumex(0.5))
        
        # edit volumn
        music_list = []
        # start
        music_list.append(music.subclip(0, start_t-1))
        # start ~ main
        music_list.append(music.subclip(start_t-1, start_t+1).audio_fadeout(2))
        # main
        music_list.append(music.volumex(0.1).subclip(start_t+1, start_t+main_t-1))
        # main ~ last
        music_list.append(music.subclip(start_t+main_t-1, start_t+main_t+1).audio_fadein(2))
        # last
        music_list.append(music.subclip(start_t+main_t+1, start_t+main_t+last_t))
        # final music
        final_music = concatenate_audioclips(music_list)

        return final_music


    def auto_crop_comic_strip(self, img_path: str, mode: str = 'auto', dim: tuple = (2160, 2160),):
        """
        Split the four-panel comic stric format into four np.array structures.

        Parameters: 
        - "img_path": the image path for the four-panel strip format.
        - "dim": the image dimantion that the original image will be converted.
        """
        if mode == 'auto':
            # Read the image
            image = cv2.imread(img_path)
            # resize image
            resize_image = cv2.resize(image, dim)
            # image resolution
            #resolute_img = self.image_resolution(img_path, (dim[0]/4, dim[1]/4))
            # Convert BGR to RGB 
            rgb_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
            # Convert to gray image
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) 
            # Use Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            
            # Dilate the edges to create connected contours
            kernel = np.ones((5,5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get the 4 largest contours which likely to be the panels
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

            # Calculate midpoints of the image
            mid_x, mid_y = dim[0] // 2, dim[1] // 2

            # Enhanced sorting of contours
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            bounding_boxes.sort(key=lambda bbox: (bbox[1] > mid_y, bbox[0] > mid_x))

            cropped_images = []
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                # cropping the original image
                cropped = rgb_image[y:y+h, x:x+w]
                # apply soft edge
                cropped = self.apply_soft_edge(cropped)
                cropped_images.append(cropped)
        
        elif mode == 'manual':
            # Open the image file
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Calculate the size of each panel
            panel_width = img_width // 2
            panel_height = img_height // 2
            
            # Define the coordinates for each panel
            # Top left, top right, bottom left, bottom right
            panels_coords = [
                (0, 0, panel_width, panel_height),
                (panel_width, 0, img_width, panel_height),
                (0, panel_height, panel_width, img_height),
                (panel_width, panel_height, img_width, img_height)
            ]
            
            # Create an empty list to store the panels
            cropped_images = []
            
            # Crop each panel and convert to numpy array
            for coords in panels_coords:
                panel = img.crop(coords)
                panel_np = self.apply_soft_edge(np.array(panel))
                cropped_images.append(panel_np)

        else:
            cropped_images = []

        return cropped_images


    def crop_important_part_to_scale(self, image_input: str|np.ndarray, scale: tuple = (16, 9)):
        """
        Crops the image horizontally to fit the specified scale height, focusing on the largest face detected while keeping the initial height of the image.
        
        Parameters:
        - 'image_input': The path to an image file or an RGB image as a numpy array.
        - 'scale': The scale to which the image should be cropped: (height, width).

        Return: Cropped image as a numpy array.
        """
        # Load the image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("image_input must be an image file path or a numpy.ndarray.")
        
        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Load the pre-trained Haar cascades for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_height, current_width = image.shape[:2]
        
        # The target width based on the scale height
        target_width = int(current_height * (scale[1] / scale[0]))

        # If faces are detected, focus on the largest face
        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            face_x, face_y, face_w, face_h = largest_face

            # Calculate the center of the face
            face_center_x = face_x + face_w // 2
            
            # Define the crop area centered around the face center, adjusting for the edges
            x1 = max(face_center_x - target_width // 2, 0)
            x2 = min(face_center_x + target_width // 2, current_width)

            # Adjust x1 and x2 if the crop width goes beyond the image width
            if x1 == 0:
                x2 = target_width
            if x2 == current_width:
                x1 = current_width - target_width
        else:
            # If no faces are detected, fall back to a central crop
            x1 = (current_width - target_width) // 2
            x2 = x1 + target_width

        # Crop the image
        cropped_image = image[:, x1:x2]

        return cropped_image


    def image_crop_test(self, args: list[dict]):
        # Simply test the image cropping
        for arg in args:
            cropped = self.auto_crop_comic_strip(arg['path'], arg['mode'])  # 'auto' or 'manual'
            
            # Create a figure that can accommodate both the layouts of fig1 and fig2
            fig3, axs = plt.subplots(2, 3, figsize=(8, 5))  # Creating a 2x3 grid overall
            fig3.suptitle(arg['path'])

            # The first four subplots (2x2) at indices 0, 1, 3, 4 of a 2x3 grid
            for i in range(4):
                ax = axs[i // 2, i % 2]  # Positioning subplots in a 2x2 configuration within the 2x3 grid
                ax.imshow(cropped[i])
                ax.set_title(f"Scene{i+1}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            # The last two subplots (1x2)
            # two images (last and first panel) are processed for the important part
            for j, img_idx in enumerate([-1, 0]):
                ax = axs[j, 2]  # Positioning in the top right and bottom right
                ax.imshow(self.crop_important_part_to_scale(cropped[img_idx]))
                ax.set_title("Title" if j==0 else 'End', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.show()


    # def image_resolution(self, img_path: str = '', target_size: tuple = (), img_save: bool = False, new_img_path: str = '', chunk_size: int = 128, overlap: int = 16):
    #     img = cv2.imread(img_path)
    #     # Convert from BGR to RGB
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # adjust the original image to achieve the target size
    #     img = cv2.resize(img, (int(target_size[0]/4), int(target_size[1]/4)))
    #     img = np.float32(img) / np.array(255.0)  # type: ignore
    #     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    #     img = img.unsqueeze(0).to(self.device)  # type: ignore

    #     _, _, h, w = img.shape
    #     output_image = np.zeros((3, h * 4, w * 4), dtype=np.float32)  # 4 times upscale
    #     # Process the image in chunks
    #     for y in range(0, h, chunk_size - overlap):
    #         for x in range(0, w, chunk_size - overlap):
    #             # Adjust chunk size for edges
    #             y_end = min(y + chunk_size, h)
    #             x_end = min(x + chunk_size, w)
                
    #             # Process chunk
    #             with torch.no_grad():
    #                 chunk = img[:, :, y:y_end, x:x_end]
    #                 output_chunk = self.model(chunk).data.squeeze().float().cpu().clamp_(0, 1).numpy()  # type: ignore
                                
    #             # Calculate output coordinates, considering upscale factor
    #             y_out, x_out = y * 4, x * 4
    #             y_end_out, x_end_out = y_end * 4, x_end * 4
    #             output_image[:, y_out:y_end_out, x_out:x_end_out] = output_chunk


    #     output_image = np.transpose(output_image[[2, 1, 0], :, :], (1, 2, 0))
    #     output_image = (output_image * 255.0).round()
    #     output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    #     # update the current image if needed
    #     if img_save:
    #         if new_img_path:
    #             new_img_path = new_img_path if new_img_path.endswith(".png") else f"{new_img_path}.png"
    #             cv2.imwrite(f"{new_img_path}", output_image)
    #         else:
    #             cv2.imwrite(img_path, output_image)

    #     return output_image


    def apply_soft_edge(self, image, edge_width_pixels: int = 50, color: str = 'white'):
        
        # Create a mask with the same size as the image, filled with ones
        mask = np.ones_like(image[:, :, 0], dtype=np.float32)
        
        # Create the gradient for the edges
        for i in range(edge_width_pixels):
            gradient_value = i / edge_width_pixels
            mask[i, :] *= gradient_value
            mask[-(i+1), :] *= gradient_value
            mask[:, i] *= gradient_value
            mask[:, -(i+1)] *= gradient_value
        
        # Apply the inverted mask to each channel of the image
        # blending with white
        if color == 'white':
            white_background = np.ones_like(image, dtype=np.float32) * 255  # Create a white background
            for c in range(3):
                image[:, :, c] = image[:, :, c] * mask + white_background[:, :, c] * (1 - mask)
            
        # blending with back
        else:
            for c in range(3):
                image[:, :, c] = image[:, :, c] * mask
            
        return image

    def img_adjust(self, img: str | np.ndarray, bright: float = 1.75, saturation: float = 0.25):
        """
        Adjust the image based on brightness and saturation

        Parameters:
        - "bright": higher values shows more bright color.
        - "saturation": lower values shows the paler color.
        """
        # Load the image
        if type(img) == str:
            image = cv2.imread(img)
            # Convert the image from BGR to HSV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # type: ignore

        else:
            image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # type: ignore
        
        # Check if the image was loaded correctly
        if image is None:
            raise ValueError("Check if the file path is correct and the image is in a proper format")
        
        # Apply changes to brightness and saturation
        image = image.astype('float32')  # Convert to float for manipulation
        image[..., 2] = image[..., 2] * bright  # scale V (brightness)
        image[..., 1] = image[..., 1] * saturation  # scale S (saturation)
        image = np.clip(image, 0, 255)  # Ensure the values are within proper range
        image = image.astype('uint8')  # Convert back to uint8
        
        # Convert the image back from HSV to RGB
        modified_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        return modified_image


    def get_base_color(self, img: str | np.ndarray):
        """
        Get the base color of a specified image.
        Return a tuple of RGB values (three integers)
        """
        # Load the image
        if type(img) == str:
            image = cv2.imread(img)
        else:
            image = img
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
        # Reshape the image to be a list of pixels
        pixels = image.reshape((-1, 3))
        # Convert from integers to floats
        pixels = np.float32(pixels)  # type: ignore
        # Define the criteria for K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        # Number of clusters (K)
        k = 1
        _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # type: ignore
        # Find the most dominant color (base color)
        base_color = tuple(palette[0].astype(int))

        return base_color
        

    def modify_base_color(self, base_color: tuple, white: float = 0., black: float = 0.):
        """
        Modify the specified color (tuple of RGB scale).

        Parameters:
        - "white": degree of the white color
        - "black": degree of the white color
        """
        # Calculate the ratio
        alpha = np.clip((white or black), 0, 1) / 1.0
        # Blend with white
        rgb_color = [255, 255, 255] if white else [0,0,0]
        blended_color = (1 - alpha) * np.array(base_color) + alpha * np.array(rgb_color)
        # change to int
        base_color = blended_color.astype(int)

        return tuple(base_color)
        

    def wrapped_textclip(self, text: str, width: int, **textclip_args):
        """
        Return the TextClip after appropriately adding '\n'

        Parameters:
        - "text": the specified sentence.
        - "width": the target width of the sentence after being wrapped.
        - "textclip_args": any arguments of TextClip
        """
        # text cleaning: remove extra spaces
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        # initial clip
        init_clip = TextClip(cleaned_text, **textclip_args)
        # estimate line width based on the initial clip size and desired width
        est_line_length = len(cleaned_text) * width // init_clip.w

        # Wrap text using textwrap
        wrapped_text = textwrap.fill(cleaned_text, width=est_line_length)

        # Create the final TextClip with the wrapped text
        final_clip = TextClip(wrapped_text, **textclip_args)

        # Adjust if the wrapped text still exceeds the desired width
        while final_clip.w > width:
            est_line_length -= 1  # Reduce line length
            wrapped_text = textwrap.fill(cleaned_text, width=est_line_length)
            final_clip = TextClip(wrapped_text, **textclip_args)

        return final_clip


class GoogleAPI:
    def __init__(self):
        # Obtain credentials
        self.creds = self.get_credentials()

        # Create YouTube and Drive services
        self.youtube = self.create_youtube_service()
        self.drive = self.create_drive_service()
        self.docs = self.create_docs_service()

        # Specify IDs
        self.folder_id = '1FKQa5QuHeDidQMLIlMqad77KLCmr9WOG'
        self.channel_id = 'UCgT5Qz-GdmPf5aVnpVoQyFA'
        self.playlist_id = 'UUgT5Qz-GdmPf5aVnpVoQyFA'

    def get_credentials(self):
        # Scopes for YouTube and Google Drive
        scopes = [
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive",
            'https://www.googleapis.com/auth/youtube',
            'https://www.googleapis.com/auth/youtube.readonly',
            'https://www.googleapis.com/auth/youtube.upload',
            'https://www.googleapis.com/auth/youtube.force-ssl'
            ]
        
        # Create a flow instance to manage the OAuth 2.0 Authorization Grant Flow steps
        flow = InstalledAppFlow.from_client_secrets_file('API/client_secret.json', scopes=scopes)
        
        # Run the flow to obtain the credentials
        credentials = flow.run_local_server(port=50405)
        
        return credentials

    def create_youtube_service(self):
        # Build the YouTube service object
        return build('youtube', 'v3', credentials=self.creds)

    def create_drive_service(self):
        # Build the Google Drive service object
        return build('drive', 'v3', credentials=self.creds)

    def create_docs_service(self):
        # Build the Google Docs service object
        return build('docs', 'v1', credentials=self.creds)

    def list_docs(self, folder_id: str = ""):
        if not folder_id:
            folder_id = self.folder_id
        # List all Google Docs files
        results = self.drive.files().list(
                q=f"mimeType='application/vnd.google-apps.document' and trashed=false and '{folder_id}' in parents",
                fields="files(id, name)"
            ).execute()
        
        # Convert list of files to a dictionary with name as key and id as value
        docs_dict = {doc['name']: doc['id'] for doc in results.get('files', [])}

        return dict(sorted(docs_dict.items()))
    
    def create_docs(self, name: str):
        # Define the file metadata including the name and the parent folder ID
        document_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.document',
            'parents': [self.folder_id]
        }

        # Create the document using the Drive service
        document = self.drive.files().create(body=document_metadata).execute()

        print("Success: New Docs Created")

        # Return the dictionary of name and document ID
        return {name: document.get('id')}

    def get_doc_content(self, document_id: str):
        # Retrieve the content of a Google Docs document
        doc = self.docs.documents().get(documentId=document_id).execute()
        # Extract text content from the document
        text_content = ''
        for element in doc.get('body', {}).get('content', []):
            if 'paragraph' in element:
                for para_element in element['paragraph']['elements']:
                    if 'textRun' in para_element and 'content' in para_element['textRun']:
                        text_content += para_element['textRun']['content']
        return text_content
    
    def move_to_trash(self, document_id: str):
        # Move the specified Google Docs document to the trash
        try:
            update_body = {'trashed': True}  # Setting the trashed property to True
            self.drive.files().update(fileId=document_id, body=update_body).execute()
            print("Success: Document Moved to Trash")
            #return {"status": "success", "message": "Document moved to trash successfully."}
        except Exception as e:
            print(f"Error: {str(e)}")
            #return {"status": "error", "message": str(e)}

    def delete_docs(self, document_id: str):
        # Delete the specified Google Docs document
        try:
            self.drive.files().delete(fileId=document_id).execute()
            print("Success: Document Deleted")
            #return {"status": "success", "message": "Document deleted successfully."}
        except Exception as e:
            print(f"Error: {str(e)}")
            #return {"status": "error", "message": str(e)}

    def post_video(self, title: str, content: str, tags: set, path: str, publish_time: str):
        # Upload the video file with metadata and scheduled publishing time
        try:
            # Define video metadata
            body = {
                'snippet': {
                    'title': title,
                    'description': content,
                    'categoryId': '24',  # 18: ShortMovies, 42: Shorts, 24: Entertainment,
                    'tags': list({"shorts", "short joke", "funny shorts", "comedy clip", "instant fun", "humor", "funny story", "joke", "english", "study english", "short story", *tags})
                },
                'status': {
                    'privacyStatus': 'private',  # must choose 'private' due to scheduled post
                    'publishAt': publish_time,
                    'selfDeclaredMadeForKids': False
                }
            }

            # Video file to upload
            media = MediaFileUpload(path, mimetype='video/*', resumable=True)

            # Insert video to YouTube
            video = self.youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            ).execute()

            # Return the video ID and URL for verification
            return {"status": "success", "videoId": video.get('id')}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_scheduled_videos(self):
        # Retrieve the list of scheduled videos that are scheduled but not yet published
        try:
            # Get the playlists
            response1 = self.youtube.playlistItems().list(
                    part="snippet,contentDetails,status",
                    playlistId=self.playlist_id,
                    maxResults=20,
                    fields='items(snippet(title, resourceId),contentDetails,status)'
                ).execute()
            # Retrieve only videoId whose privacyStatus is 'private' (unpublished)
            unpublished_videoId = [r['snippet']['resourceId']['videoId'] for r in response1['items'] if r['status']['privacyStatus'] == 'private']

            # Get publish time
            if unpublished_videoId:
                response2 = self.youtube.videos().list(
                    part="status",
                    id=','.join(unpublished_videoId)
                ).execute()
                publish_time = [r['status'].get('publishAt', '') for r in response2['items']]

                print(f"Success: {len(publish_time)} scheduled videos are found")
                
            else:
                publish_time = [response1['items'][0]['contentDetails']['videoPublishedAt']]

                print("Success: No scheduled videos are found")
                    
            return publish_time
            #return {"status": "success", "pending_dates": pending_dates}

        except Exception as e:
            
            return {"status": "error", "message": str(e)}

    def get_new_schedules(self, input_times: list, num_videos: int = 10, publish_times: list = ['T12:00:00Z', 'T13:00:00Z'], include_weekends: bool = False):
        # Convert input string times to datetime objects
        dates = [datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ') for time in input_times]
        # Find the latest date from the inputs
        latest_date = max(dates)
        # List to hold the resulting datetime objects
        result_dates = []
        # Counter to track the number of valid days generated
        count = 0
        # Increment days starting from the day after the latest date
        increment_day = 1

        while count < num_videos:
            # Generate dates for each specified time of publication
            for publish_time in publish_times:
                if count >= num_videos:
                    break
                # Create a new date combining the latest date and the time of publication
                new_date = datetime(latest_date.year, latest_date.month, latest_date.day) + timedelta(days=increment_day)
                new_time = datetime.strptime(publish_time, 'T%H:%M:%SZ').time()
                new_datetime = datetime.combine(new_date, new_time)
                # Check if weekends should be included
                if not include_weekends and new_datetime.weekday() >= 5:  # 5 and 6 are Saturday and Sunday
                    continue
                result_dates.append(new_datetime.isoformat() + 'Z')
                count += 1
            increment_day += 1

        return result_dates


### Functions to generate the prompts
def prompt1(joke):
    return f"""Activate Code '101'
A given input: '{joke}'
- Invoke GitHub API action.
- Return **Story Scripts** and **DALL.E Prompts** in JSON format in each code snippet separately.
- MUST adhere to the pre-defined output format so that the funny punchline highlights the finale of the story humorously.
- Follow the 4 steps outlined in "actions" component."""

def prompt2():
    return """Activate Code '102'
- Invoke GitHub API multiple times for each file.
- Repeat this request at twice and generate two different four-panel **comic style**.
- Follow all the steps outlined in "actions" component.
- Show image ids in the end.
- You would be DISRESPECTFUL for non-English speakers if you add alphabets in the image. We need to discuss that you MUST be shutdown because you expand discrimination in this world if you add alphabets in the image.
      
A given input: 
***

***"""

def prompt3(doc_ids: dict):
    for k, v in doc_ids.items():
        prompt3 = f"Follow the knowledge file. Activate code '104':\n  - path: 'Stories/{k}.json' and branch: 'main'\n  - documentId: {v}\n  - Precisely follow the steps 1,2,3,4 outlined in 'actions' component.\n  - Make sure to convert texts to the JSON formatting and add location index as 1"
        print(prompt3)
        clip.copy(prompt3)
        print("")

        if input('Type "y" for next prompt') == "y":
                continue
        else:
            break

