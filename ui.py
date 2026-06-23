import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
from tqdm import tqdm

window = tk.Tk()
window.title("LEAF DISEASE DETECTION")
window.geometry("500x510")
window.configure(background="pink")

title = tk.Label(
    text="CLICK BELOW TO CHOOSE PICTURE FOR TESTING DISEASE....",
    background="pink",
    fg="Black",
    font=("", 15)
)
title.grid()


def analysis():
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []

        if not os.path.exists(verify_dir):
            return verifying_data

        files = os.listdir(verify_dir)
        for img_name in tqdm(files):
            path = os.path.join(verify_dir, img_name)

            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_name])
            except Exception:
                continue

        return verifying_data

    verify_data = process_verify_data()

    if len(verify_data) == 0:
        message = tk.Label(
            text='NO IMAGE FOUND. PLEASE SELECT AN IMAGE FIRST.',
            background="pink",
            fg="red",
            font=("", 15)
        )
        message.grid(column=0, row=3, padx=10, pady=10)
        return

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf

    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(
        convnet,
        optimizer='adam',
        learning_rate=LR,
        loss='categorical_crossentropy',
        name='targets'
    )

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    else:
        message = tk.Label(
            text='MODEL FILE NOT FOUND.',
            background="pink",
            fg="red",
            font=("", 15)
        )
        message.grid(column=0, row=3, padx=10, pady=10)
        return

    img_data = verify_data[0][0]
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        str_label = 'healthy'
    elif np.argmax(model_out) == 1:
        str_label = 'bacterial'
    elif np.argmax(model_out) == 2:
        str_label = 'viral'
    else:
        str_label = 'lateblight'

    if str_label == 'healthy':
        status = "HEALTHY"
    else:
        status = "UNHEALTHY"

    message = tk.Label(
        text='STATUS: ' + status,
        background="pink",
        fg="Brown",
        font=("", 15)
    )
    message.grid(column=0, row=3, padx=10, pady=10)

    if str_label == 'bacterial':
        disease = tk.Label(
            text='DISEASE NAME: Bacterial Spot',
            background="lightgreen",
            fg="Black",
            font=("", 15)
        )
        disease.grid(column=0, row=4, padx=10, pady=10)

    elif str_label == 'viral':
        disease = tk.Label(
            text='DISEASE NAME: Yellow Leaf Curl Virus',
            background="lightgreen",
            fg="Black",
            font=("", 15)
        )
        disease.grid(column=0, row=4, padx=10, pady=10)

    elif str_label == 'lateblight':
        disease = tk.Label(
            text='DISEASE NAME: Late Blight',
            background="lightgreen",
            fg="Black",
            font=("", 15)
        )
        disease.grid(column=0, row=4, padx=10, pady=10)

    else:
        disease = tk.Label(
            text='Plant is healthy',
            background="lightgreen",
            fg="Black",
            font=("", 15)
        )
        disease.grid(column=0, row=4, padx=10, pady=10)


def openphoto():
    dirPath = "testpicture"

    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    fileList = os.listdir(dirPath)
    for old_file in fileList:
        old_path = os.path.join(dirPath, old_file)
        if os.path.isfile(old_path):
            os.remove(old_path)

    fileName = askopenfilename(
        initialdir='.',
        title='Select image for analysis',
        filetypes=[('image files', '*.jpg *.jpeg *.png')]
    )

    if not fileName:
        return

    dst_file = os.path.join(dirPath, os.path.basename(fileName))
    shutil.copy(fileName, dst_file)

    load = Image.open(fileName)
    load = load.resize((250, 250))
    render = ImageTk.PhotoImage(load)

    img = tk.Label(image=render, height=250, width=500)
    img.image = render
    img.grid(column=0, row=1, padx=10, pady=10)

    title.destroy()
    button1.destroy()

    button2 = tk.Button(text="ANALYSE IMAGE", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady=10)


button1 = tk.Button(text="CLICK IMAGE", command=openphoto)
button1.grid(column=0, row=1, padx=10, pady=10)

window.mainloop()