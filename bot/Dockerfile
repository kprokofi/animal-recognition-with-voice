FROM python:3.8-slim-buster

COPY . /bot
WORKDIR /bot

RUN pip install --upgrade pip && pip install --no-cache-dir -r /bot/bot_requirements.txt
ENV PYTHONUNBUFFERED=1\
    SOUNDS_PATH=/bot/static/sounds\
    IMAGENET_CLASSES_PATH=/bot/imagenet_classes.txt
CMD python /bot/run_bot.py