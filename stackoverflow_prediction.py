#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pickle
import tensorflow as tf
import os
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.widgets import TextArea

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

#encoder = tf.keras.models.load_model('DAN/')
encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

# load json and create model
json_file = open('RN_usencoder2tags_model_config.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("RN_usencoder2tag_weights.h5")

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'])

def get_dataset(X, batch_size):
    def generator():
        # x = tf.reshape(tf.convert_to_tensor(' '.join(X[i]), dtype=tf.string), (1,)) # convert words list to sentences
        x = tf.reshape((tf.convert_to_tensor(X, dtype=tf.string)), (1,))
        x = tf.reshape(encoder(x), (512,))  # encode post 
        yield x

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types= tf.float64,
                                             output_shapes= tf.TensorShape([512, ]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset			  

batch_size = 256

# load tags


class ReusableForm(Form):
    name = TextAreaField('Post:',
                        validators=[validators.required()],
                        widget=TextArea(),
                        render_kw={"rows": 20, "cols": 120})
    

@app.route("/", methods=['GET', 'POST'])

def hello():

    with open('RN_usencoder2tags_tags', 'rb') as f:
        tags = pickle.load(f)
		
    form = ReusableForm(request.form)
    
    print(form.errors)
    
    if request.method == 'POST':
        name=request.form['name']
        test_dataset = get_dataset(name, batch_size)
        print(name)
    
    if form.validate():
        prediction = np.concatenate(np.array([model.predict(data) for data in test_dataset]))
        p = np.mean(prediction[0])+2*np.std(prediction[0])
        index = np.where(prediction[0] > p)
        tags_pred = tags[index]
        tags = ' | '.join(tags_pred)
        flash(tags)
    else:
        flash('')
    
    return render_template('layout.html', form=form)

if __name__ == "__main__":
    app.run()