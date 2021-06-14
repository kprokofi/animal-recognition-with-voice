import logging
import os
from PIL import Image
from io import BytesIO
from telegram import Update
from telegram.ext import CallbackContext

from recognizer_bot.bot_actions import send_text, send_voice_message
from recognizer_bot.voicy import get_voice_by_label

logger = logging.getLogger(__name__)
CLASSIFIER = None


def init_classifier(classifier):
    global CLASSIFIER
    CLASSIFIER = classifier


def help_command(update: Update, context: CallbackContext):
    '''
        /help command handler
    '''
    update.message.reply_text('Send me a photo of an animal')


def make_sound(update: Update, context: CallbackContext):
    '''
        Handler for image message.
        Run classifier and choose appropriate sound.

    '''
    image_bio = BytesIO()
    file = context.bot.getFile(update.message.photo[-1].file_id)
    logger.info('Photo downloaded')

    file.download(out=image_bio)
    image = Image.open(image_bio)
    if CLASSIFIER:
        label, conf = CLASSIFIER.classify(image)
    else:
        logger.error('Classifier not found')
    logger.info(f'Image classified as {label} with confidence {conf:.2f}%')

    send_text(update, context, f'Looks like it is {label}')

    sounds_path = os.getenv('SOUNDS_PATH')
    voice_path = get_voice_by_label(sounds_path=sounds_path, label=label)
    if voice_path:
        send_voice_message(update, context, voice_path)
    else:
        send_text(update, context, 'Sorry, but no sounds found')
