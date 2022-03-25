# Python codes for extracting language-based pre-trained BERT feature described in Section 4.2
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import bert
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def createTokenizer():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    bert_models_dir_path = os.path.dirname(currentDir)
    modelsFolder = os.path.join(bert_models_dir_path, "NLP", "ICMI_21", "models", "uncased_L-12_H-768_A-12") # this is my customized path for downloaded bert model
    vocab_file = os.path.join(modelsFolder, "vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

tokenizer = createTokenizer()
tokenized_text_corpus = []

for session in text_corpus:
    utterances_tokens = []
    for utterance in session:
        if (utterance[0] != utterance[0]): # check if utterance = ['nan']
            utterance = ['empty']
        words = tokenizer.tokenize(utterance[0])
        words.append('[SEP]')
        words = ['[CLS]'] + words
        utterances_tokens.append(words)
    tokenized_text_corpus.append(utterances_tokens)

MAX_SEQ = 0 # MAX sequence per utterance time length

for session in tokenized_text_corpus: # this function gets the max sequence length of each utterance
    for utterance in session:
        if len(utterance) > MAX_SEQ:
            MAX_SEQ = len(utterance)
        else:
            MAX_SEQ = MAX_SEQ

MAX_UTT = 0 # MAX utterance per session

for session in tokenized_text_corpus:
    if len(session) > MAX_UTT:
        MAX_UTT = len(session)
    else:
        MAX_UTT = MAX_UTT

print("MAX utterance per session", MAX_UTT)
print("MAX sequence per utterance time length", MAX_SEQ)

def createBertLayer(max_seq_length):
    global bert_layer
    currentDir = os.path.dirname(os.path.realpath(__file__))
    bert_models_dir_path = os.path.dirname(currentDir)
    modelsFolder = os.path.join(bert_models_dir_path, "NLP", "ICMI_21", "models", "uncased_L-12_H-768_A-12")
    bert_params = bert.params_from_pretrained_ckpt(modelsFolder)
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    model_layer = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer
    ])
    model_layer.build(input_shape=(None, max_seq_length))
    bert_layer.apply_adapter_freeze() # use this to use pre-trained BERT weights; otherwise the model will train the BERT model from the scratch
    # bert_layer.trainable = False

createBertLayer(MAX_SEQ)

def loadBertCheckpoint():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    bert_models_dir_path = os.path.dirname(currentDir)
    modelsFolder = os.path.join(bert_models_dir_path, "NLP", "ICMI_21", "models", "uncased_L-12_H-768_A-12")
    checkpointName = os.path.join(modelsFolder, "bert_model.ckpt")
    bert.load_stock_weights(bert_layer, checkpointName)

loadBertCheckpoint()
print("done!")

print("Converting Tokens Into IDs... and Post Padding...")
tokenized_text_corpus_ids = []

for session in tokenized_text_corpus:
    train_tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in session]
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=MAX_SEQ, dtype="long", truncating="post", padding="post")
    tokenized_text_corpus_ids.append(train_tokens_ids)

def createModel():
    global model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQ,), dtype='int32', name='input_ids'),
        bert_layer,
        tf.keras.layers.BatchNormalization(momentum=0.99),
        tf.keras.layers.Lambda(lambda x: x[:, 0, :])
    ])
    model.build(input_shape=(None, MAX_SEQ))
    model.compile()
    print(model.summary())

createModel()

for session in tokenized_text_corpus_ids:
    predict_list = model.predict(session).tolist() # generated utterance embedding from pre_trained model
