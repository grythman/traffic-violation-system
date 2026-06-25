# Custom model weights

Place your custom YOLOv8 plate-detection weights here, for example:

    models/mn_plate_yolov8.pt

Then set in your `.env`:

    PLATE_MODEL_PATH=/app/models/mn_plate_yolov8.pt

This folder is mounted read-only into the API container at `/app/models`.
The folder is committed (via this README) but the large `.pt` files are
git-ignored.
