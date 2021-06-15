import os

labels_to_voice = {
    'hippopotamus': 'hippopotamus',
    'black-footed_ferret': 'ferret',
    'lion': 'lion',
    'king_penguin': 'penguin',
    'hyena': 'hyena',
    'Irish_wolfhound': 'wolf',
    'timber_wolf': 'wolf',
    'white_wolf': 'wolf',
    'red_wolf': 'wolf',
    'tiger_cat': 'tiger',
    'Persian_cat': 'cat',
    'Siamese_cat': 'cat',
    'Egyptian_cat': 'cat',
    'Madagascar_cat': 'cat',
    'robin': 'robin',
    'alligator_lizard': 'lizard',
    'American_alligator': 'alligator',
    'zebra': 'zebra',
    'tiger_shark': 'shark',
    'tiger': 'tiger',
    'tiger_beetle': 'tiger',
    'frilled_lizard': 'lizard',
    'green_lizard': 'lizard',
    'leopard': 'leopard',
    'snow_leopard': 'leopard',
    'lesser_panda': 'panda',
    'giant_panda': 'panda',
    'cabbage_butterfly': 'butterfly',
    'sulphur_butterfly': 'butterfly',
    'guinea_pig': 'pig',
    'piggy_bank': 'pig',
    'great_grey_owl': 'owl',
    'Indian_elephant': 'elephant',
    'African_elephant': 'elephant',
    'scorpion': 'scorpion',
    'Maltese_dog': 'dog',
    'Old_English_sheepdog': 'dog',
    'Shetland_sheepdog': 'dog',
    'Greater_Swiss_Mountain_dog': 'dog',
    'Bernese_mountain_dog': 'dog',
    'French_bulldog': 'dog',
    'Eskimo_dog': 'dog',
    'African_hunting_dog': 'dog',
    'Sussex spaniel': 'dog',
    'vizsla': 'dog',
    'Chesapeake_Bay_retriever': 'dog',
    'Labrador_retriever': 'dog',
    'Arabian_camel': 'camel',
    'water_buffalo': 'buffalo',
    'great_white_shark': 'shark',
    'bee': 'bee',
    'black_swan': 'swan',
    'bullfrog': 'frog',
    'tree_frog': 'frog',
    'tailed_frog': 'frog',
    'wood_rabbit': 'rabbit',
    'tabby': 'cat',
    'beagle': 'dog',
    'pug': 'dog'
}


def prepare_voice_path(sounds_path, voice_path: str):
    return os.path.join(sounds_path, voice_path + '.ogg')


def get_voice_by_label(sounds_path: str, label: str):
    label = label.strip()
    if label in labels_to_voice:
        return prepare_voice_path(sounds_path, labels_to_voice[label])
    return None
