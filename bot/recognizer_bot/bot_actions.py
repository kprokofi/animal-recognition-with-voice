from telegram import Update
from telegram.ext import CallbackContext


def send_text(update: Update, context: CallbackContext, mes: str):
    context.bot.send_message(chat_id=update.effective_chat.id, text=mes)


def send_voice_message(update: Update, context: CallbackContext, path: str):
    with open(path, 'rb') as f:
        context.bot.sendVoice(chat_id=update.effective_chat.id, voice=f)

# def send_image(update: Update, context: CallbackContext, path: str):
#     from io import BytesIO
#     bio = BytesIO()
#     bio.name = 'image.jpeg'
#     image.save(bio, 'JPEG')
#     bio.seek(0)
#     bot.send_photo(chat_id, photo=bio)

#     with open(path, 'rb') as f:
#         context.bot.send_photo(f)