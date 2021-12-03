"""Submodule for video creation. 

Copied from: 
https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python

Adapted from purpose."""

import cv2
import os
from pathlib import Path


def make_video(
    string_time: str,
    generator_name: str,
    batch_size: int,
    num_batches: int,
    latent_space_dimension: int,
    neuron: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    image_dir = Path.cwd() / "video" / f"neuron_{neuron}"
    image_folder = str(image_dir)
    l, r = str(lr).split(".")
    lr = l + ":" + r
    weight, decay = str(weight_decay).split(".")
    weight_decay = weight + ":" + decay
    video_dir = Path.cwd() / "video" / string_time
    video_dir.mkdir(parents=True, exist_ok=True)
    video_name = f"video_{neuron}_{generator_name}_batchsize_{batch_size}_numbatches_{num_batches}_latent_{latent_space_dimension}_epochs_{epochs}_lr_{lr}_l2_{weight_decay}.avi"
    video_name = str(video_dir / video_name)

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    # delete images for video creation
    for f in image_dir.iterdir():
        f.unlink()
    image_dir.rmdir()
