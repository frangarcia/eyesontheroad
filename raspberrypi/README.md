Image classifier

    python3 classify.py

Download the eyesontheroad tflite file

    wget -O eyesontheroad.tflite.zip https://www.dropbox.com/s/9dh1038nimi8ooz/model.tflite.zip?dl=0

Extract the model

    unzip eyesontheroad.tflite.zip

Rename file

    mv model.tflite eyesontheroad.tflite

Eyes on the road classifier

    python3 classify.py --model=eyesontheroad.tflite