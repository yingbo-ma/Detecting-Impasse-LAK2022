# Python codes for multimodal classification models, this file will be updated later

from nltk.tokenize import sent_tokenize
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import pandas as pd
import bert
import numpy as np
from scipy import stats


print("\n**************************************************************************************************")
print("loading acoustic data...")
Average_Time_Interval = 7 # we set the average time interval for each sentence is 7 seconds
data_list = pd.read_csv('./acoustic_feature_data/acoustic_data.csv', delimiter='\t', header=None).values.tolist()
acoustic_feature_lists = []

for list in data_list:
    new_list = list[0].strip('][').split(', ')
    feature_list = []
    for number in new_list:
        if ("'" in number):
            number = int(number[1:-1])
        else:
            number = int(number)
        feature_list.append(number)

    feature_list = feature_list[:Average_Time_Interval*128]
    feature_list = np.array(feature_list)
    feature_list = stats.zscore(feature_list).tolist() ## z-score normalization, standardized by mean and standard deviation of the input array
    acoustic_feature_lists.append(feature_list)

print("The length of the visual feature list is: ", len(acoustic_feature_lists))
print("Done!")

print("\n**************************************************************************************************")
print("loading visual data...")
data_list = pd.read_csv('./visual_feature_data/visual_feature_data.csv', delimiter='\t', header=None).values.tolist()
visual_feature_lists = []

for list in data_list:
    new_list = list[0].strip('][').split(',')
    if (len(new_list) == 1): # in case the visual feature list is empty
        visual_feature_lists.append(visual_feature_lists[-1]) # if empty we append the lastest one directly
    else:
        feature_list = []
        for number in new_list:
            if ("'" in number):
                number = float(number[1:-1])
            else:
                number = float(number)
            feature_list.append(number)

        if (len(feature_list) < Average_Time_Interval*82): # for visual feature we have 76 numbers for each second
            feature_list += [0.0]*(Average_Time_Interval*82-len(feature_list))
        else:
            feature_list = feature_list[:Average_Time_Interval*82]

        feature_list = np.array(feature_list)
        feature_list = stats.zscore(feature_list).tolist()
        visual_feature_lists.append(feature_list)

visual_feature_lists = visual_feature_lists[:-1] # discard the last weird [SUB] token
print("The length of the visual feature list is: ", len(visual_feature_lists))
print("Done!")

print("\n**************************************************************************************************")
print("loading textual data...")
print("create tokenizer...")
def createTokenizer():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    modelsFolder = os.path.join(currentDir, "models", "uncased_L-12_H-768_A-12")
    vocab_file = os.path.join(modelsFolder, "vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

tokenizer = createTokenizer()

data_list = pd.read_csv('./csv_data/data/data.csv', delimiter='\t', header=None).values.tolist()

utterance_tokens = []
pair_tokens = []

for sentences in data_list:
    sentence_list = sent_tokenize(sentences[0])
    sentences_tokens = []
    for sentence in sentence_list:
        words = tokenizer.tokenize(sentence)
        words.append('[SEP]')
        sentences_tokens += words
    sentences_tokens += ['[EOT]']
    utterance_tokens.append(sentences_tokens)

for token_index in range(len(utterance_tokens)-1):
    current_token = utterance_tokens[token_index]
    next_token = utterance_tokens[token_index+1]
    turn_token = ['[CLS]'] + current_token + next_token
    turn_token = turn_token[:-1]
    pair_tokens.append(turn_token)

print("The length of the textual data is: ", len(pair_tokens))

print("Done!")

print("\n**************************************************************************************************")
print("loading labels...")

label_list = pd.read_csv('./csv_data/label/label.csv', delimiter='\t', header=None).values.tolist()
print("The length of the label list is: ", len(label_list))
print("Done!")


print("\n**************************************************************************************************")
print("create bert layer...")
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

MAX_SEQ_LEN = 40

def createBertLayer(max_seq_length):
    global bert_layer
    currentDir = os.path.dirname(os.path.realpath(__file__))
    bertDir = os.path.join(currentDir, "models", "uncased_L-12_H-768_A-12")
    bert_params = bert.params_from_pretrained_ckpt(bertDir)
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    model_layer = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer
    ])
    model_layer.build(input_shape=(None, max_seq_length))
    bert_layer.apply_adapter_freeze() # use this to use pre-trained BERT weights; otherwise the model will train the BERT model from the scratch
    # bert_layer.trainable = False

createBertLayer(MAX_SEQ_LEN)

def loadBertCheckpoint():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    bertDir = os.path.join(currentDir, "models", "uncased_L-12_H-768_A-12")
    checkpointName = os.path.join(bertDir, "bert_model.ckpt")
    bert.load_stock_weights(bert_layer, checkpointName)

loadBertCheckpoint()
print("done!")

print("\n**************************************************************************************************")
print("create model...")

use_language_model = False
use_acoustic_model = False
use_visual_model = False
use_language_acoustic_model = False
use_language_visual_model = False
use_acoustic_visual_model = True
use_triple_joint_model = False

Average_Time_Interval = 7
MAX_ACOUSTIC_LEN = Average_Time_Interval*128 # for audios we have 128 dimensional vector for each second, we set the average time is 7 seconds
MAX_VISUAL_LEN = Average_Time_Interval*82 # for video we have 82 dimensional vector for each second, was 76 before

def createModel():
    global model

    if use_language_model:
        print("Use language model!")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32', name='input_ids'),
            bert_layer,
            tf.keras.layers.BatchNormalization(momentum=0.99),
            tf.keras.layers.Lambda(lambda x: x[:, 0, :]),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        model.build(input_shape=(None, MAX_SEQ_LEN))
        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        print(model.summary())

    elif use_acoustic_model:
        print("Use acoustic model!")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(MAX_ACOUSTIC_LEN,), dtype='int32', name='input_ids'),
            tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        model.build(input_shape=(None, MAX_ACOUSTIC_LEN))
        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        print(model.summary())

    elif use_visual_model:
        print("Use visual model!")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(MAX_VISUAL_LEN,), dtype='int32', name='input_ids'),
            tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        model.build(input_shape=(None, MAX_VISUAL_LEN))
        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        print(model.summary())

    elif use_language_acoustic_model:
        print("Use language and acoustic model!")
        inputA = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32', name='input_ids')  # textual input
        inputB = tf.keras.layers.Input(shape=(MAX_ACOUSTIC_LEN,))  # acoustic input

        A = bert_layer(inputA)
        A = tf.keras.layers.BatchNormalization(momentum=0.99)(A)
        A = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(
            A)  # We are only only interested in BERT’s output for the [CLS] token, so here select that slice of the cube and discard everything else.
        A = tf.keras.Model(inputs=inputA, outputs=A)

        Combined = tf.keras.layers.concatenate([A.output, inputB])
        Combined = tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(Combined)

        model = tf.keras.Model(inputs=[A.input, inputB], outputs=Combined)

        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        print(model.summary())

    elif use_language_visual_model:
        print("Use language and visual model!")
        inputA = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32', name='input_ids')  # textual input
        inputC = tf.keras.layers.Input(shape=(MAX_VISUAL_LEN,))  # visual input

        A = bert_layer(inputA)
        A = tf.keras.layers.BatchNormalization(momentum=0.99)(A)
        A = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(
            A)  # We are only only interested in BERT’s output for the [CLS] token, so here select that slice of the cube and discard everything else.
        A = tf.keras.Model(inputs=inputA, outputs=A)

        Combined = tf.keras.layers.concatenate([A.output, inputC])
        Combined = tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(Combined)

        model = tf.keras.Model(inputs=[A.input, inputC], outputs=Combined)

        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        print(model.summary())

    elif use_acoustic_visual_model:
        inputB = tf.keras.layers.Input(shape=(MAX_ACOUSTIC_LEN,))  # acoustic input
        inputC = tf.keras.layers.Input(shape=(MAX_VISUAL_LEN,))  # visual input

        Combined = tf.keras.layers.concatenate([inputB, inputC])
        Combined = tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(Combined)

        model = tf.keras.Model(inputs=[inputB, inputC], outputs=Combined)

        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        print(model.summary())

    elif use_triple_joint_model:
        print("Use language, acoustic and visual triple joint model!")
        inputA = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32', name='input_ids')  # textual input
        inputB = tf.keras.layers.Input(shape=(MAX_ACOUSTIC_LEN,))  # acoustic input
        inputC = tf.keras.layers.Input(shape=(MAX_VISUAL_LEN,))  # visual input

        A = bert_layer(inputA)
        A = tf.keras.layers.BatchNormalization(momentum=0.99)(A)
        A = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(A)  # We are only only interested in BERT’s output for the [CLS] token, so here select that slice of the cube and discard everything else.
        A = tf.keras.Model(inputs=inputA, outputs=A)

        Combined = tf.keras.layers.concatenate([A.output, inputB, inputC])
        Combined = tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(Combined)
        Combined = tf.keras.layers.Dropout(0.4)(Combined)
        Combined = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(Combined)

        model = tf.keras.Model(inputs=[A.input, inputB, inputC], outputs=Combined)

        model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        print(model.summary())

    else:
        print("ERROR! Please select a model type to build!")

createModel()
print("done!")

print("\n**************************************************************************************************")
print("preparing training and testing data...")

tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in pair_tokens]
textual_token_ids = pad_sequences(tokens_ids, maxlen=MAX_SEQ_LEN, dtype="long", truncating="post", padding="post")

text_train, text_vali, audio_train, audio_vali, video_train, video_vali, label_train, label_vali = train_test_split(textual_token_ids, acoustic_feature_lists, visual_feature_lists, label_list, test_size=0.2)

text_train = np.array(text_train)
text_vali = np.array(text_vali)
audio_train = np.array(audio_train)
audio_vali = np.array(audio_vali)
video_train = np.array(video_train)
video_vali = np.array(video_vali)
label_train = np.array(label_train)
label_vali = np.array(label_vali)
print("done!")

print("\n**************************************************************************************************")
print("start training...")


if use_language_model:
    print("Use language model!")
    history = model.fit(
        text_train,
        label_train,
        batch_size=16,
        epochs=10,
        validation_data=(text_vali, label_vali),
        verbose=1
    )

elif use_acoustic_model:
    print("Use acoustic model!")
    history = model.fit(
        audio_train,
        label_train,
        batch_size=16,
        epochs=10,
        validation_data=(audio_vali, label_vali),
        verbose=1
    )

elif use_visual_model:
    print("Use visual model!")
    history = model.fit(
        video_train,
        label_train,
        batch_size=16,
        epochs=10,
        validation_data=(video_vali, label_vali),
        verbose=1
    )

elif use_language_acoustic_model:
    print("Use language and acoustic model!")
    history = model.fit(
        [text_train, audio_train],
        label_train,
        batch_size=16,
        epochs=10,
        validation_data=([text_vali, audio_vali], label_vali),
        verbose=1
    )

elif use_acoustic_visual_model:
    print("Use language and acoustic model!")
    history = model.fit(
        [audio_train, video_train],
        label_train,
        batch_size=16,
        epochs=10,
        validation_data=([audio_vali, video_vali], label_vali),
        verbose=1
    )

elif use_language_visual_model:
    print("Use language and visual model!")
    history = model.fit(
        [text_train, video_train],
        label_train,
        batch_size=16,
        epochs=10,
        validation_data=([text_vali, video_vali], label_vali),
        verbose=1
    )

elif use_triple_joint_model:
    print("Use language, acoustic and visual triple joint model!")
    history = model.fit(
        [text_train, audio_train, video_train],
        label_train,
        batch_size=16,
        epochs=30,
        validation_data=([text_vali, audio_vali, video_vali], label_vali),
        verbose=1
    )

else:
    print("ERROR! Please select a model type to train!")

print("Done!")

print("\n**************************************************************************************************")
print("Curves plotting...")
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("\n**************************************************************************************************")
print("confusion matrix on training set...")

if use_language_model:
    y_pred = model.predict(text_train,verbose=0)
elif use_acoustic_model:
    y_pred = model.predict(audio_train,verbose=0)
elif use_visual_model:
    y_pred = model.predict(video_train,verbose=0)
elif use_language_acoustic_model:
    y_pred = model.predict([text_train, audio_train],verbose=0)
elif use_language_visual_model:
    y_pred = model.predict([text_train, video_train],verbose=0)
elif use_triple_joint_model:
    y_pred = model.predict([text_train, audio_train, video_train],verbose=0)
elif use_acoustic_visual_model:
    y_pred = model.predict([audio_train, video_train], verbose=0)
else:
    print("ERROR! Please select a model type to predict!")

y_pred_list = []
for pred_index in range(len(y_pred)):
    if y_pred[pred_index][0] < 0.5:
        y_pred_list.append(0)
    elif y_pred[pred_index][0] > 0.5:
        y_pred_list.append(1)
    else:
        print("ERROR! pred probability equals to 0.5!")
print(classification_report(label_train, y_pred_list))

print("\n**************************************************************************************************")
print("confusion matrix on test set...")
if use_language_model:
    y_pred = model.predict(text_vali,verbose=0)
elif use_acoustic_model:
    y_pred = model.predict(audio_vali,verbose=0)
elif use_visual_model:
    y_pred = model.predict(video_vali,verbose=0)
elif use_language_acoustic_model:
    y_pred = model.predict([text_vali, audio_vali],verbose=0)
elif use_language_visual_model:
    y_pred = model.predict([text_vali, video_vali],verbose=0)
elif use_triple_joint_model:
    y_pred = model.predict([text_vali, audio_vali, video_vali],verbose=0)
elif use_acoustic_visual_model:
    y_pred = model.predict([audio_vali, video_vali], verbose=0)
else:
    print("ERROR! Please select a model type to predict!")

y_pred_list = []
for pred_index in range(len(y_pred)):
    if y_pred[pred_index][0] < 0.5:
        y_pred_list.append(0)
    elif y_pred[pred_index][0] > 0.5:
        y_pred_list.append(1)
    else:
        print("ERROR! pred probability equals to 0.5!")
print(classification_report(label_vali, y_pred_list))