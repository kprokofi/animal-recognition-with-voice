#!/bin/bash
for filename in /home/friday/HSE/animal-recognition-with-voice/sounds2/*; do
    ffmpeg -i "$filename" -ac 1 -map 0:a -codec:a libopus -b:a 128k -vbr off -ar 24000 "$filename".ogg;
done
