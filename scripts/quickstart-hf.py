from PIL import Image
from transformers import AutoModelForCausalLM, AutoConfig
from starvector.data.util import process_and_rasterize_svg, ImageTrainProcessor
import torch
import time

model_name = "starvector/starvector-1b-im2svg"
# model_name = "starvector/starvector-8b-im2svg"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
starvector = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=config, trust_remote_code=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
starvector.to(device)
starvector.eval()

image_pil = Image.open('assets/examples/sample-18.png')
image_processor = ImageTrainProcessor(size=config.image_size)
image_tensor = image_processor(image_pil).unsqueeze(0).to(device)

batch = {"image": image_tensor}

# Generate SVG from image
timestamp = time.strftime("%Y%m%d-%H%M%S")
svg_filename = f"generated_output_{timestamp}.svg"
png_filename = f"generated_output_{timestamp}.png"

raw_svg = starvector.generate_im2svg(batch, max_new_tokens=100, max_length=1000)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)

with open(svg_filename, "w") as f:
    f.write(svg)
raster_image.save(png_filename)
print(f"SVG saved to {svg_filename}, raster image saved to {png_filename}")
