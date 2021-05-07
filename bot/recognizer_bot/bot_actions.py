from telegram import Update
from telegram.ext import CallbackContext


def send_text(update: Update, context: CallbackContext, mes: str):
    context.bot.send_message(chat_id=update.effective_chat.id, text=mes)


def send_voice_message(update: Update, context: CallbackContext, path: str):
    with open(path, 'rb') as f:
        context.bot.sendVoice(chat_id=update.effective_chat.id, voice=f)
