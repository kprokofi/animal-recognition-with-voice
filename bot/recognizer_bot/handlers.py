import logging
from PIL import Image
from io import BytesIO
from telegram import Update
from telegram.ext import CallbackContext

from recognizer_bot.bot_actions import *
from recognizer_bot.classificator import classify
from recognizer_bot.voicy import choose_voice

logger = logging.getLogger(__name__)


def help_command(update: Update, context: CallbackContext):
    '''
        /help command handler
    '''
    update.message.reply_text('Send me a photo of an animal')



def choose_sound(update: Update, context: CallbackContext):
    '''
        Handler for image message.
        Run classifier and choose appropriate sound.

    '''
    image_bio = BytesIO()
    file = context.bot.getFile(update.message.photo[-1].file_id)
    logger.info('Photo downloaded')

    file.download(out=image_bio)
    image = Image.open(image_bio)
    label, conf = classify(image)
    logger.info(f'Image classified as {label} with confidence {conf:.2f}%')

    send_text(update, context, f'Looks like it is {label}')
    voice_path = choose_voice(label)
    if voice_path:
        send_voice_message(update, context,voice_path)
    else:
        send_text(update, context, f'Sorry, but no sounds found')

