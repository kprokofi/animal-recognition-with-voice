import os
import logging
from dotenv import load_dotenv
from recognizer_bot.classifier import Classifier
from recognizer_bot.handlers import init_classifier, help_command, make_sound
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, Dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def add_handlers(dp: Dispatcher) -> None:
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.photo, make_sound))


def main():
    init_classifier(
        Classifier(classes_path=os.getenv('IMAGENET_CLASSES_PATH'))
    )
    updater = Updater(
        os.getenv('TOKEN'), use_context=True
    )
    dp = updater.dispatcher
    add_handlers(dp)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    load_dotenv(
        dotenv_path='/home/friday/HSE/animal-recognition-with-voice/.env')
    main()
