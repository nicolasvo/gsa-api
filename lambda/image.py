# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam_vit_l, SamPredictor
import cv2
import numpy as np
from scipy import ndimage

# diffusers
import torch

from huggingface_hub import hf_hub_download


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

device = "cpu"
sam_checkpoint = "./weights/sam_vit_l_0b3195.pth"
sam = build_sam_vit_l(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)


def rescale_image(image, px=512, padding=0):
    height, width, _ = image.shape
    if [height, width].index(max([height, width])) == 0:
        factor = px / height
        height = px
        width = int(width * factor)
    else:
        factor = px / width
        width = px
        height = int(height * factor)

    image_resized = cv2.resize(
        image, dsize=(width, height), interpolation=cv2.INTER_LINEAR
    )

    # Create a larger canvas with the same number of channels as the input image
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padded_image = np.zeros(
        (padded_height, padded_width, image.shape[2]), dtype=np.uint8
    )

    # Calculate the position to place the resized image in the center
    x_offset = (padded_width - width) // 2
    y_offset = (padded_height - height) // 2

    # Place the resized image in the center of the padded canvas
    padded_image[
        y_offset : y_offset + height, x_offset : x_offset + width
    ] = image_resized

    return padded_image


def add_outline(image, stroke_size, outline_color):
    # Ensure the image has an alpha channel for transparency
    if image.shape[-1] != 4:
        raise ValueError("Input image must have an alpha channel (4 channels).")

    # Create a copy of the original image
    outlined_image = image.copy()

    # Create a mask for fully transparent parts of the image
    mask = (image[:, :, 3] == 0).astype(np.uint8)

    # Calculate the kernel size based on the desired stroke size
    kernel_size = int(stroke_size * 0.2) * 2 + 1  # Ensure it's an odd number
    if kernel_size < 1:
        kernel_size = 1

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion to round the outline
    outline = cv2.erode(mask, kernel, iterations=1)

    # Use the eroded mask to get the outline of the fully transparent parts
    outline = mask - outline

    # Apply Gaussian blur to smooth the outline
    outline = cv2.GaussianBlur(
        outline.astype(np.float32), (kernel_size, kernel_size), 0
    )

    # Threshold the blurred outline to make it binary
    _, outline = cv2.threshold(outline, 0.5, 1, cv2.THRESH_BINARY)

    # Apply the outline color only to the outline region
    for c in range(4):  # Loop through RGBA channels
        outlined_image[:, :, c] = (
            outlined_image[:, :, c] * (1 - outline) + outline_color[c] * outline
        )

    return outlined_image


def extract_bounding_box(image, bbox):
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        bounding_box_image = image[min_y : max_y + 1, min_x : max_x + 1]
        return bounding_box_image
    else:
        return None


def get_bbox_from_image(image):
    alpha_channel = image[
        :, :, 3
    ]  # Assuming alpha channel is the last channel (index 3)
    non_transparent_mask = alpha_channel > 0

    # Find the coordinates of non-transparent pixels
    non_transparent_indices = np.argwhere(non_transparent_mask)

    if non_transparent_indices.size > 0:
        x1, y1 = non_transparent_indices.min(axis=0)
        x2, y2 = non_transparent_indices.max(axis=0)

        # Bounding box coordinates
        x1, y1, x2, y2 = y1, x1, y2, x2  # Swap x and y for correct format
        return x1, y1, x2, y2
    return False


def keep_small_transparent_regions(mask, h_area_threshold=None, w_area_threshold=None):
    if h_area_threshold is None:
        h_area_threshold = mask.shape[0] * 0.01
    if w_area_threshold is None:
        w_area_threshold = mask.shape[1] * 0.01

    labeled, num_features = ndimage.label(mask == 0)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes[labeled]
    mask_filled = mask.copy()

    for label in range(1, num_features + 1):
        area = np.sum(labeled == label)
        if area <= h_area_threshold * w_area_threshold:
            h, w = np.where(labeled == label)
            mask_filled[h, w] = 1

    return mask_filled


def remove_small_nontransparent_regions(
    mask, h_area_threshold=None, w_area_threshold=None
):
    if h_area_threshold is None:
        h_area_threshold = mask.shape[0] * 0.01
    if w_area_threshold is None:
        w_area_threshold = mask.shape[1] * 0.01

    labeled, num_features = ndimage.label(mask == 1)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes[labeled]
    mask_removed = mask.copy()

    for label in range(1, num_features + 1):
        area = np.sum(labeled == label)
        if area <= h_area_threshold * w_area_threshold:
            h, w = np.where(labeled == label)
            mask_removed[h, w] = 0

    return mask_removed


def segment(input_path, text_prompt):
    local_image_path = input_path
    TEXT_PROMPT = text_prompt.replace(" and ", " . ")
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(local_image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB

    # set image
    sam_predictor.set_image(image_source)

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    ).to(device)
    if transformed_boxes.shape[0] == 0:
        return False, False
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    image = cv2.imread(input_path)
    masks = masks.numpy().squeeze(1)
    mask = np.sum(masks, axis=0)
    mask = np.where(mask >= 1, 1, 0)
    mask = keep_small_transparent_regions(mask)
    mask = remove_small_nontransparent_regions(mask)
    alpha_channel = np.where(mask == 0, 0, 255).astype(np.uint8)
    image = cv2.merge((image, alpha_channel))

    return image


def make_sticker(input_path, output_path, text_prompt):
    image = segment(input_path, text_prompt)
    bbox = get_bbox_from_image(image)
    if isinstance(bbox, bool) and bbox is False:
        return False
    image = extract_bounding_box(image, bbox)
    image = rescale_image(image, padding=13)
    image = add_outline(image, 40, (255, 255, 255, 255))
    image = rescale_image(image, padding=0)
    cv2.imwrite(output_path, image)

    return True
