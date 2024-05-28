import warnings
import requests
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          AutoProcessor, AutoModelForVision2Seq, pipeline)
from PIL import Image
from io import BytesIO
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

image_url = "https://preview.redd.it/rainbow-reticulated-python-v0-14q4uz5w6z5a1.jpg?width=640&crop=smart&auto=webp&s=9a4287bc010d8273c3eb4cbb3c5bd40bc1244167"

response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

text_blip = "An image of"
inputs_blip = processor_blip(image, text_blip, return_tensors="pt")

out_blip = model_blip.generate(**inputs_blip)
caption_blip = processor_blip.decode(out_blip[0], skip_special_tokens=True)
print(f"Salesforce BLIP Large caption: {caption_blip}")

processor_kosmos = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
model_kosmos = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")

inputs_kosmos = processor_kosmos(text="An image of", images=image, return_tensors="pt")
generated_ids_kosmos = model_kosmos.generate(
    pixel_values=inputs_kosmos["pixel_values"],
    input_ids=inputs_kosmos["input_ids"],
    attention_mask=inputs_kosmos["attention_mask"],
    image_embeds=None,
    image_embeds_position_mask=inputs_kosmos["image_embeds_position_mask"],
    use_cache=True,
    max_new_tokens=128,
)

generated_text_kosmos = processor_kosmos.batch_decode(generated_ids_kosmos, skip_special_tokens=True)[0]
print(f"Microsoft Kosmos-2 caption: {generated_text_kosmos}")

image_to_text_vit_gpt2 = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

caption_vit_gpt2 = image_to_text_vit_gpt2(image)[0]['generated_text']
print(f"NLPConnect VIT-GPT-2 caption: {caption_vit_gpt2}")
