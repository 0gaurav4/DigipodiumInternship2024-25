
import fal_client
from dotenv import load_dotenv
import os

load_dotenv()
FAL_KEY = os.getenv("FAL_KEY")

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])
import fal_client

async def main():
    handler = await fal_client.submit_async(
        "fal-ai/fast-sdxl",
        arguments={
            "prompt": "photo of a cat wearing a kimono"
        },
    )

    async for event in handler.iter_events(with_logs=True):
        if isinstance(event, fal_client.InProgress):
            print(event)

    result = await handler.get()
    return result

import asyncio
result = asyncio.run(main())