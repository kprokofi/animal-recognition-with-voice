import os
import logging
from dotenv import load_dotenv
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, Dispatcher

from recognizer_bot.classifier import Classifier
from recognizer_bot.detector import Detector

from recognizer_bot.handlers import init_classifier, init_detector, help_command, make_sound

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def add_handlers(dp: Dispatcher) -> None:
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.photo, make_sound))


def prepare_detector_conifg():
    return {
        'pruning_model': False,
        'anchors_path': os.getenv('YOLO_ANCHORS_PATH'),
        'classes_path': os.getenv('YOLO_CLASSES_PATH'),
        'score': 0.1,
        'iou': 0.4,
        'model_image_size': (416, 416),
        'elim_grid_sense': False,
        'gpu_num': 1,
    }


def main():
    init_classifier(
        Classifier(classes_path=os.getenv('IMAGENET_CLASSES_PATH'))
    )
    init_detector(
        Detector(model_path=os.getenv('YOLO_PATH'), config=prepare_detector_conifg())
    )
    updater = Updater(
        os.getenv('TOKEN'), use_context=True
    )
    dp = updater.dispatcher
    add_handlers(dp)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    load_dotenv()
    main()
