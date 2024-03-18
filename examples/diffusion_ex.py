from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from io import BytesIO
import requests

pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
)
pipe = pipe.to("cuda")

url = "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

out = pipe(image, num_images_per_prompt=3, guidance_scale=15)
out["images"][0].save("result.jpg")
