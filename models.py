

from PIL import Image
from lavis.models import load_model_and_preprocess
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

from utils import find_image_list

#loads blip model
def load_blip_2(name="blip2_t5", model_type="caption_coco_flant5xl"):
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True)

    return model, vis_processors

#generates captions
def generate_captions(model, vis_processors, df):
    image_list = find_image_list(df)
    df["image"] = image_list
    captions = []

    for i in tqdm(range(len(df))):
        image_data = Image.open(df["image"].iloc[i]).convert("RGB")
        image = vis_processors["eval"](image_data).unsqueeze(0)
        x = model.generate(
            {"image": image, "prompt": "Give Caption for this Image along with details on objects specifie"})
        captions.append(x)

    df.drop(columns=["image", "image_id_y"], inplace=True)
    df["captions_BLIP"] = captions

    return df

#loads flan model
def load_flan_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")

    return tokenizer, model


#generates output answers
def find_flan_output_answers(model, tokenizer, df):
    answer_flan = []

    for idx in tqdm(range(len(df))):
        prompt = f"""Given the caption, provide an answer for the question provided below.
                Question: '''{df["question"].iloc[idx]}'''
                Caption: '''{df["captions_BLIP"].iloc[idx]}'''
                """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        answer_flan.append(tokenizer.decode(outputs[0]))

    cleaned_flan_answers = [sentence.replace("<pad>", "").replace("</s>", "") for sentence in answer_flan]

    df["answer_flan"] = cleaned_flan_answers

    return df

#generates output explanations
# this is explanations based on actual answers
def find_flan_output_explainations(model, tokenizer, df):
    explain_flan = []

    for idx in tqdm(range(len(df))):
        prompt = f"""#Based on the following data provided, answer the following question: "Why is that? Give an explanation/rationale for the answer provided."
                Question: '''{df["question"].iloc[idx]}'''
                Answer: '''{df["answer"].iloc[idx]}'''
                Caption: '''{df["captions_BLIP"].iloc[idx]}'''
                """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        explain_flan.append(tokenizer.decode(outputs[0]))

    cleaned_flan_explain = [sentence.replace("<pad>", "").replace("</s>", "") for sentence in explain_flan]

    df["explain_flan"] = cleaned_flan_explain

    return df

#explanation based on generated answers

def find_flan_output_explainations_gen(model, tokenizer, df):
    explain_flan = []

    for idx in tqdm(range(len(df))):
        prompt = f"""#Based on the following data provided, answer the following question: "Why is that? Give an explanation/rationale for the answer provided."
                Question: '''{df["question"].iloc[idx]}'''
                Answer: '''{df["answer_flan"].iloc[idx]}'''
                Caption: '''{df["captions_BLIP"].iloc[idx]}'''
                """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        explain_flan.append(tokenizer.decode(outputs[0]))

    cleaned_flan_explain = [sentence.replace("<pad>", "").replace("</s>", "") for sentence in explain_flan]

    df["explain_flan_gen"] = cleaned_flan_explain

    return df
             
