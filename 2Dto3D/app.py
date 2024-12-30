# 58e87482-2bcf-46e2-8208-8ca528f41fc4:250a35190ce3f17a189647938ac0c8f0

import fal_client
import fal_client
from dotenv import load_dotenv
import os

load_dotenv()
FAL_KEY = os.getenv("FAL_KEY")

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

result = fal_client.subscribe(
    "fal-ai/trellis",
    arguments={
        "image_url": "https://jooinn.com/images/macro-photography-of-red-petal-flower-2.jpg",
        "ss_guidance_strength": 7.5,
        "ss_sampling_steps": 12,
        "slat_guidance_strength": 3,
        "slat_sampling_steps": 12,
        "mesh_simplify": 0.95,
        "texture_size": 1024
    },
    with_logs=True,
    on_queue_update=on_queue_update,
)
print(result)


# https://gltf-viewer.donmccurdy.com/

# https://github.com/donmccurdy/three-gltf-viewer