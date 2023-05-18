import os
import openai
import csv 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from urllib.request import urlopen

# openai.api_key = "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"
openai.api_key = "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"
os.environ['HYDRA_FULL_ERROR'] = '1'
def read_example_prompts(file_path):
    
    with open(file_path, 'r') as f:
       data = f.read()
    return data
# hh = read_example_prompts('test.txt')
# print(hh)
def generate_box(text):
   
    example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image. Each object in the image is one rectangle or square box in the layout and size of boxes should be as large as possible comapre to the image size. The size of the image is 512 * 512 You should return each object and the corresponding coordinate of its boxes.\nthe prompt :\"three cats in the field\", \ncat: (51, 82, 399, 279)\ncat: (288, 128, 472, 299)\ncat: (27, 355, 418, 494)\nthe prompt: \"a cat on the left of a dog on the road\"\ncat: (63, 196, 223, 394)\ndog: (289, 131, 466, 360)\nthe prompt: \"four balls in the room\"\nball: (72, 81, 254, 243)\nball: (316, 44, 483, 218)\nball: (287, 295, 453, 462)\nball: (50, 323, 196, 484)\nthe prompt: \"A donut to the right of a toilet\"\ndonut: (287, 140, 467, 335)\ntoilet: (31, 97, 216, 286)\nthe prompt: \"A cat sitting on the top of a car\"\ncar: (94, 236, 414, 407)\ncat: (124, 139, 273, 252)\nthe prompt: \"A dog underneath a tree\"\ndog: (133, 232, 308, 445)\ntree: (121, 29, 324, 258)\nthe prompt: \"a small ball is put on the top of a box on the table. there is a red vase on the right of the box on the table\"\nsmall ball: (92, 30, 165, 134)\nbox: (93, 132 , 205, 324, 310)\nred vase: (214, 164, 297, 301)\ntable: (36, 259, 418, 463)\n"
    print('example ', example_prompt)
    prompt = example_prompt  + text
    # call api
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    # complete text
    completed_text = response['choices'][0]['text']
    print('text', completed_text)
    # getting boxes and object names
    boxes = completed_text.split('\n')
    d = {}
    name_objects = []
    boxes_of_object = []
    # convert boxes from string to int
    # import pdb; pdb.set_trace()
    for b in boxes:
        if b == '': continue
       
        b_split = b.split(":")
        name_objects.append(b_split[0])
        boxes_of_object.append(text_list(b_split[1]))


    # for o in list_object:
    #     for b in boxes:
            
    #         if o in b.lower():

    #             if not o in d.keys():
    #                 d[o] = [text_list(b.split(": ")[1])]
    #             else:      
    #                 d[o].append(text_list(b.split(": ")[1]))
    
    return name_objects, boxes_of_object
def generate_box_gpt4(inputs):
    messages =[{"role": "user", "content": "Can you act like an programmer.I will provide the description of an image, you should output the corresponding layout of this image includding . The size of the image is 512 * 512. Can you return names and corresponding coodinate locations of objects with the prompt: three cats in the field"}, {"role": "system", "content": "cat: (51, 82, 399, 279)\ncat: (288, 128, 472, 299)\ncat: (27, 355, 418, 494)"}]
    # the prompt: \"A donut to the right of a toilet\"\ndonut: (287, 140, 467, 335)\ntoilet: (31, 97, 216, 286)\n
    #  "three cats in the field\", \ncat: (51, 82, 399, 279)\ncat: (288, 128, 472, 299)\ncat: (27, 355, 418, 494)\n"
    # constraint='Each object in the image is one rectangle or square box in the layout. The size of the image is 512 * 512.'
    constraint = 'Can you return names and corresponding coodinate locations of objects with the prompt: '
    message = constraint + inputs +' Do not add any number or special characters to the names of objects.'
    messages.append(
            {"role": "user", "content": message},
        )
    chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages
        )
    completed_text = chat.choices[0].message.content
    print('text gpt', completed_text)
    # import pdb; pdb.set_trace()
    boxes = completed_text.split('\n')
    d = {}

    name_objects = []
    boxes_of_object = []
    # convert boxes from string to int
    for b in boxes:
        if b == '': continue
       
        b_split = b.split(":")
        name_objects.append(b_split[0])
        boxes_of_object.append(text_list(b_split[1]))
    return name_objects, boxes_of_object
    
def text_list(text):
    text =  text.replace(' ','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    # coodinate chat GPT api is opposite
    # tempt = result[0]
    # result[0] = result[1]
    # result[1] = tempt 
    # tempt = result[2]
    # result[2] = result[3]
    # result[3] = tempt
    return tuple(result)
def read_csv(path_file, t):
    list_prompts = []
    with open(path_file,'r') as f:
        reader = csv.reader(f)
        # import pdb; pdb.set_trace()
        for i, row in enumerate(reader):
            if i >0:
                
                if  row[1] == t: #'Positional'or row[1] == 'Counting': #row[1] == 'Positional' or
                    list_prompts.append(row[0])
    return list_prompts

def read_txt_label(file_path):
    labels = {}
    with open(file_path, 'r') as f:
        for x in f:
            x = x.replace(' \n', '')
            x = x.replace('\n', '')
            x = x.split(',')
            labels.update({x[0]: x[2]})
    return labels

def draw_box(text, boxes,output_folder, img_name):
    sample = Image.fromarray((np.ones((512, 512)) * 255).astype(np.uint8))
    
    draw = ImageDraw.Draw(sample)
    font = ImageFont.truetype(urlopen("https://criptolibertad.s3.us-west-2.amazonaws.com/img/fonts/Roboto-LightItalic.ttf"), size=20)
    import pdb; pdb.set_trace()
    for i, box in enumerate(boxes):
       
        t = text[i]
        draw.rectangle([(box[0], box[1]),(box[2], box[3])], outline=128, width=2)
        draw.text((box[0]+5, box[1]+5), t, fill=200,font=font )
    sample.save(os.path.join(output_folder, img_name))
