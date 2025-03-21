from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from transformers import AutoConfig
from starvector.data.util import process_and_rasterize_svg, ImageTrainProcessor
import torch
import time

model_name = "starvector/starvector-1b-im2svg"
# model_name = "starvector/starvector-8b-im2svg"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
starvector = StarVectorForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=config, trust_remote_code=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
starvector.to(device)
starvector.eval()

image_pil = Image.open("assets/examples/sample-18.png")
image_pil = image_pil.convert('RGB')
image = starvector.process_images([image_pil])[0].to(torch.float16).to(device)
batch = {"image": image}

# Generate SVG from image
timestamp = time.strftime("%Y%m%d-%H%M%S")
svg_filename = f"generated_output_qs_{timestamp}.svg"
png_filename = f"generated_output_qs_{timestamp}.png"

raw_svg = starvector.generate_im2svg(batch, max_length=1000, temperature=1.5, length_penalty=-1, repetition_penalty=3.1)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)

with open(svg_filename, "w") as f:
    f.write(svg)
raster_image.save(png_filename)
print(f"SVG saved to {svg_filename}, raster image saved to {png_filename}")
