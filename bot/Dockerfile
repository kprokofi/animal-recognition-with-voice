FROM python:3.8-slim-buster

COPY . /bot
WORKDIR /bot

RUN pip install --upgrade pip && pip install --no-cache-dir -r /bot/bot_requirements.txt
ENV PYTHONUNBUFFERED=1\
    SOUNDS_PATH=/bot/static/sounds\
    IMAGENET_CLASSES_PATH=/bot/static/config/imagenet_classes.txt\
    YOLO_CLASSES_PATH=/bot/static/config/yolo_classes.txt\
    YOLO_ANCHORS_PATH=/bot/static/config/yolo4_anchors.txt\
    YOLO_PATH=/bot/static/models/yolo3_mobilenet_lite_saved_model.h5
CMD python /bot/run_bot.py