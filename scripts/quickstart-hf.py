from PIL import Image
from attr.validators import max_len
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from starvector.data.util import process_and_rasterize_svg
import torch
import time

# model_name = "starvector/starvector-1b-im2svg"
model_name = "starvector/starvector-8b-im2svg"

starvector = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
processor = starvector.model.processor
tokenizer = starvector.model.svg_transformer.tokenizer

# starvector.cuda()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
starvector.to(device)
starvector.eval()

image_pil = Image.open('assets/examples/sample-18.png')
timestamp = time.strftime("%Y%m%d-%H%M%S")
svg_filename = f"generated_output_{timestamp}.svg"
png_filename = f"generated_output_{timestamp}.png"

# image = processor(image_pil, return_tensors="pt")['pixel_values'].cuda()
image = processor(image_pil, return_tensors="pt")['pixel_values'].to(device)

if not image.shape[0] == 1:
    image = image.squeeze(0)
batch = {"image": image}

# raw_svg = starvector.generate_im2svg(batch, max_length=100)[0]
raw_svg = starvector.generate_im2svg(batch, max_new_tokens=100, max_length=1000)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)
with open(svg_filename, "w") as f:
    f.write(svg)
raster_image.save(png_filename)
print(f"SVG saved to {svg_filename}, raster image saved to {png_filename}")