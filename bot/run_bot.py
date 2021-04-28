import os
from recognizer_bot.handlers import *
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, Dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

TOKEN=os.getenv('TOKEN')


def add_handlers(dp:Dispatcher)-> None:
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.photo, choose_sound))


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    add_handlers(dp)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
