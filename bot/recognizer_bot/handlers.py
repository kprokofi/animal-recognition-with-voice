import logging
import os
from PIL import Image
from io import BytesIO
from telegram import Update
from telegram.ext import CallbackContext

from recognizer_bot.bot_actions import send_text, send_voice_message, send_image
from recognizer_bot.voicy import get_voice_by_label

logger = logging.getLogger(__name__)
CLASSIFIER = None
DETECTOR = None


def init_classifier(classifier):
    global CLASSIFIER
    CLASSIFIER = classifier


def init_detector(detector):
    global DETECTOR
    DETECTOR = detector


def help_command(update: Update, context: CallbackContext):
    '''
        /help command handler
    '''
    update.message.reply_text('Send me some animal photo 📸')


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

    # Detect anumals, crop and feed each cropped image to classificator
    if DETECTOR and CLASSIFIER:
        image_array, out_boxes, out_classnames, out_scores = DETECTOR.detect_image(image)
        logger.info(f'Detected: {out_classnames} with scores {out_scores}')

        # cache already detected and classified animals to prevent sending same voice many times
        classified_animals = set()

        if len(out_boxes) == 0:
            send_text(update, context, "I can't find any animals on photo 🤔")  # TODO

        for box in out_boxes:
            # Crop image
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_image = image_array[ymin:ymax, xmin:xmax]
            # Apply classifier

            label, conf = CLASSIFIER.classify(Image.fromarray(cropped_image))
            logger.info(
                f'Image classified as {label} with confidence {conf:.2f}%')
            # Sending response to user
            if label not in classified_animals:
                send_image(update, context, Image.fromarray(cropped_image))
                send_text(update, context, f"I guess it's a {label.replace('_',' ')} 🧐 ")  # TODO make emoji list
                classified_animals.add(label)
                sounds_path = os.getenv('SOUNDS_PATH')
                voice_path = get_voice_by_label(
                    sounds_path=sounds_path, label=label)
                if voice_path:
                    send_text(update, context, f"...and {label.replace('_',' ')} says ")
                    send_voice_message(update, context, voice_path)
                else:
                    send_text(update, context, "Sorry, I don't what it says 😟")
    else:
        logger.error('Detector or Classifier not found')
