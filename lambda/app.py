import base64
import json
import tempfile

from fastapi import FastAPI, HTTPException

from image import make_sticker

app = FastAPI()


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


def base64_to_image(base64_string, output_file_path):
    with open(output_file_path, "wb") as image_file:
        decoded_image = base64.b64decode(base64_string)
        image_file.write(decoded_image)


@app.post("/sticker_gsa")
async def process_image(image_data: dict):
    # Extract parameters from the JSON data
    image_base64 = image_data.get("image", "")
    text_prompt = image_data.get("text_prompt", "")

    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        input_path = f"{tmpdirname}/input.jpeg"
        input_then_path = f"{tmpdirname}/input_then.png"
        output_path = f"{tmpdirname}/output.png"
        base64_to_image(image_base64, input_path)
        is_segmented = make_sticker(
            input_path, output_path, text_prompt, input_then_path
        )
        if is_segmented:
            return json.dumps({"image": image_to_base64(output_path)})
        return json.dumps({"image": ""})

    # Process the image and the provided data here
    # You can perform image processing, object detection, etc. using the image, boxes, and points.

    # return {"message": "Image processed successfully", "boxes": boxes, "points": points}
