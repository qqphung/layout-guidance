import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import compute_ca_loss, Pharse2idx, draw_box, setup_logger
import hydra
import os
from tqdm import tqdm
import numpy as np
from chatgpt import generate_box_gpt4, read_csv
import pandas as pd
import pickle
import random
import argparse
def remove_numbers(text):
    result = ''.join([char for char in text if not char.isdigit()])
    return result
def process_box_phrase(names, bboxes):
    d = {}
    # import pdb; pdb.set_trace()
    for i, phrase in enumerate(names):
        phrase = phrase.replace('_',' ')
        list_noun = phrase.split(' ')
        for n in list_noun:
            n = remove_numbers(n)
            if not n in d.keys():
                d.update({n:[np.array(bboxes[i])/512]})
            else:
                d[n].append(np.array(bboxes[i])/512)
    return d
def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.','')
    prompt = prompt.replace(',','')
    prompt_list = prompt.strip('.').split(' ')

    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'s' in prompt_list:
                obj_first_index = prompt_list.index(word+'s') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'es' in prompt_list:
                obj_first_index = prompt_list.index(word+'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word == 'person' and 'person' in prompt_list:
                obj_first_index = prompt_list.index('people') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word == 'mouse' and 'mice' in prompt_list:
                obj_first_index = prompt_list.index('mice') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
        if in_prompt :
            bbox_to_self_att.append(np.array(name_box[obj]))

            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att

def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg, logger, seed, object_positions):


    logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}")

    # Get Object Positions

    logger.info("Conver Phrases to Object Positions")
    
    #object_positions = Pharse2idx(prompt, phrases)
    

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma

    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0
        


        while loss.item() / cfg.inference.loss_scale > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter:# and index < cfg.inference.max_index_step:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            #import pdb; pdb.set_trace()
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)

            # update latents with guidance
            loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                   object_positions=object_positions) * cfg.inference.loss_scale
           
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], allow_unused=True)[0]

            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2
            iteration += 1
            torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        logger.info("Decode Image...")
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

def get_phrase(names, prompt):
    phrase = ''
    for name in names:
        name_split = name.split(' ')
        if name_split == 1: name_split = name.split('_')
        for n in name_split:
            if n=='': continue
            # import pdb; pdb.set_trace()
            if n in prompt :
                phrase += n + ';'
            elif n+'s' in prompt:
                phrase += n +'s;'
            elif n+'es' in prompt:
                phrase += n+ 'es;'
    return phrase[:-1]
def load_box(pickle_file):
    with open(pickle_file,'rb') as f:
        data = pickle.load(f)
    return data
def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    meta = []
    syn_prompt = []

    # import pdb; pdb.set_trace()
    for sample in gt_data:
        meta.append(sample['meta_prompt'])
        syn_prompt.append(sample['synthetic_prompt'])
    return meta, syn_prompt
def save_img(folder_name, img, prompt, iter_id, img_id):
    os.makedirs(folder_name, exist_ok=True)
    img_name = str(img_id) + '_' + str(iter_id) + '_' + prompt.replace(' ','_')+'.png'
    img.save(os.path.join(folder_name, img_name))



@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):
    
    
    # build and load model
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)
    # import pdb; pdb.set_trace()
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)



    # ------------------ example input ------------------
    # examples = {"prompt": "A couch on the left of a chair.",
    #             "phrases": "couch; chair",
    #             "bboxes": [[[0.0,0.09,0.4,0.5]], [[0.5,0.11,0.9,0.7]]], # [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]],
    #             'save_path': cfg.general.save_path
    #             }
    # ---------------------------------------------------
    # Prepare the save path
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    logger = setup_logger(cfg.general.save_path, __name__)

    logger.info(cfg)
    # Save cfg
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))
    #prompts = read_csv('/vulcanscratch/chuonghm/GLIGEN/drawbench.csv')
    # Inference
    # if type == 'spatial':
    # prompts,_ = load_gt('/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/spatial_compositions_prompts.csv')
    # # elif args.type == 'counting':
    # # _ ,prompts = load_gt('/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/counting_prompts.csv')
    # list_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting.p',
    #             '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting_500_1499.p',
    #             '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting_1500_2499.p',
    #             '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting_5.p']
    
    # # list_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting.p',
    # #         '/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/counting_5.p']
    # list_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/spatial.p']
    # list_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/size.p']
    # prompts,_ = load_gt('/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/size_compositions_prompts.csv')
    # prompts = read_csv('/vulcanscratch/chuonghm/data_evaluate_LLM/drawbench/drawbench.csv','Positional')
    # list_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box_drawbench/spatial.p']
    prompts, _ = load_gt("/vulcanscratch/chuonghm/data_evaluate_LLM/HRS/colors_composition_prompts.csv")   
    list_box = ['/vulcanscratch/chuonghm/data_evaluate_LLM/gpt_generated_box/color.p']
    for box_pickle in list_box:

        data_boxes = load_box(box_pickle)

        for id_img, prompt in enumerate(prompts):
            if not prompt in data_boxes.keys(): continue
            #if id_img == 2535 or id_img ==131 or id_img==2880: continue
            #if id_img < 1382 : continue
            if id_img ==2: continue
            
            
            # names, boxes = generate_box_gpt4(prompt[0])
            names, boxes = data_boxes[prompt]
            phrases = get_phrase(names, prompt)
            name_box = process_box_phrase(names, boxes)
            position, box_att = Pharse2idx_2(prompt, name_box)

        
            boxes = np.expand_dims(boxes, axis=1)
            examples = {"prompt": prompt,
                    "phrases": phrases,
                    "bboxes": box_att, # [[[0.0,0.09,0.4,0.5]], [[0.5,0.11,0.9,0.7]]], # [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]],
                    'save_path': cfg.general.save_path
                    }
            for i in range(1):
                seed = random.randint(0,10000)
                pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['bboxes'], examples['phrases'], cfg, logger, seed,position)

            # Save example images
                for index, pil_image in enumerate(pil_images):
                    save_img(cfg.general.save_path,pil_image, prompt, i, id_img)
                    # image_path = os.path.join(cfg.general.save_path, prompt.replace(' ', "_") + 'example_{}.png'.format(index))
                    # logger.info('save example image to {}'.format(image_path))
                    # draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", type=str,  default="HRS_save", help="root folder for output")
    # parser.add_argument("--type", type=str)
    # parser.add_argument("--box_pickle", type=str)
    # args = parser.parse_args()
    main()
