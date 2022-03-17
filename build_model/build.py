"""
This file contains the code for building and evaluating the deep-learning image captioning model, based on the Flickr8K dataset.
Steps:
1. Extract features from images using pre-trained CNN (here I use VGG-16)
2. Pre-process text data
3. Build the deep-learning model (my model is based on the merge model as described by Tanti, et al. (2017). Where to put the Image in an Image Caption Generator.)
4. Progressive model training (since the memory of my computer is insufficient for loading the entire dataset at once)
5. Evaluation based on BLEU score
"""

import os
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import string
import json
from pickle import load, dump
from nltk.translate.bleu_score import corpus_bleu

"""
1. Extract image features using pre-trained CNN.
"""
def feature_extractions(directory):
    """
    Input: directory of images
    Return: A dictionary of features extracted by VGG-16, size 4096.
    """
    
    model = tf.keras.applications.vgg16.VGG16()
    model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output) #Remove the final layer
    
    features = {}
    for f in os.listdir(directory):
        filename = directory + "/" + f
        identifier = f.split('.')[0]
        
        image = keras.preprocessing.image.load_img(filename, target_size=(224,224))
        arr = keras.preprocessing.image.img_to_array(image, dtype=np.float32)
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = keras.applications.vgg16.preprocess_input(arr)
    
        feature = model.predict(arr, verbose=0)
        features[identifier] = feature
        
        print("feature extraction: {}".format(f))
    return(features)
# =============================================================================
# 
# if __name__ == "__main__":
#     features = feature_extractions("Flickr8k_Dataset")
#     
#     #save features for future use.
#     with open("features.pkl", "wb") as f:
#        dump(features, f)
# =============================================================================
   
"""
2. Text data pre-processing.
"""
def caption_dictionary(raw_caption):
    """
    Input: raw_caption as retrieved from the dataset
    Return: A dictionary mapping [photo_id] -> caption_list of that photo
    """
    captions = {}
    for line in raw_caption.split('\n'):
        if len(line) < 1:
            continue
        
        tmp = line.split()
        photo_id = tmp[0].split('.')[0] #id
        caption = ' '.join(tmp[1:])
        
        if photo_id not in captions.keys():
            captions[photo_id] = []
        captions[photo_id].append(caption) #A list of captions

    return(captions)

def caption_cleaning(caption_dict):
    """
        Input: caption_directory.
        Perform text pre-processing for captions
        Return: caption_directory after pre-processing
    """
    trans_table = str.maketrans('', '', string.punctuation)
    for photo_id, caption_list in caption_dict.items():
        for i in range(len(caption_list)):
            caption = caption_list[i]
            caption = caption.lower()
            #caption = re.sub('[^\w\s]', '', caption) #remove punctuation & special character
            caption = caption.translate(trans_table) #remove punctuation only
            
            caption_list[i] = caption
            
    return(caption_dict)

if __name__ == "__main__":
    with open("Flickr8k_text/Flickr8k.token.txt", "r") as f:
        raw_caption = f.read()
        
    caption_dict = caption_dictionary(raw_caption)
    caption_dict = caption_cleaning(caption_dict)
    
    #Save the caption_dict for future use
    with open("captions.pkl", "wb") as f:
        dump(caption_dict, f)

"""
Load the extracted image features and the pre-processed caption dictionary.
"""

def caption_to_list(caption_dict):
    #Return: a list of all captions from the caption_dict
    captions = []
    for caption_list in caption_dict.values():
        for c in caption_list:
            captions.append(c)
    return(captions)
    
def create_tokenizer(caption_dict, num_vocab=None):
    """
    Input: caption dictionary, num_vocab
    Output: Tokenizer fitted on the captions in the dictionary, with maximum number of vocab = num_vocab
    """
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_vocab, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    captions = caption_to_list(caption_dict)
            
    tokenizer.fit_on_texts(captions)
    return(tokenizer)

#Identify training / development / testing set
def dataset(filename):
    """
    Input: filename of dataset
    Output: A list of identifier in the dataset
    """
    dataset = []
    with open(filename, "r") as f:
        text = f.read()
    
    for line in text.split('\n'):
        if len(line) < 1:
            continue
        dataset.append(line.split('.')[0])
        
    return(dataset)

def load_features(dataset):
    """
    Input: dataset (list of identifier)
    Output: The VGG-16 features according to the identifiers
    """
    with open("features.pkl", "rb") as f:
        features = load(f)
    features = {photo_id: features[photo_id] for photo_id in dataset}
    
    return(features)
    
def load_captions(dataset, wrapping = 0):
    """
    Input: dataset (list of identifier), wrapping (by startseq / endseq)
    Output: The caption_dict according to the identifiers, with / without wrapping of (startseq, endseq)
    """
    with open("captions.pkl", "rb") as f:
        caption_dict = load(f)
    caption_dict = {photo_id: caption_dict[photo_id] for photo_id in dataset}
    
    if wrapping:
        for photo_id, caption_list in caption_dict.items():
            for i in range(len(caption_list)):
                tmp = caption_list[i].split()
                caption_list[i] = '<startseq> ' + ' '.join(tmp) + ' <endseq>'
    
    return(caption_dict)

if __name__ == "__main__":
    train = dataset("Flickr8k_text/Flickr_8k.trainImages.txt")
    dev = dataset("Flickr8k_text/Flickr_8k.devImages.txt")
    test = dataset("Flickr8k_text/Flickr_8k.testImages.txt")
    
    train_caption_dict = load_captions(train, 1)
    train_features = load_features(train)
    dev_caption_dict = load_captions(dev, 1)
    dev_features = load_features(dev)
    test_caption_dict = load_captions(test, 0)
    test_features = load_features(test)
    
    #Initialize the tokenizer
    vocab_size = 3001
    tokenizer = create_tokenizer(train_caption_dict, num_vocab=vocab_size)
    
    #Save the tokenizer for caption_generation
    with open("../tokenizer.json", "w") as f:
        json.dump(tokenizer.to_json(), f)
    
    print("vocab size: {}".format(vocab_size))
    max_length = max([len(c.split()) for c in caption_to_list(train_caption_dict)]) - 1 #Maximum length of input sequence
    print("max length: {}".format(max_length))

"""
Load the pre-trained embeddings from GloVe
"""
#Check coverage
def check_coverage(tokenizer, coverage, vocab_size):
    iw = tokenizer.index_word
    wc = tokenizer.word_counts
    
    coverage_vocab = 0
    coverage_count = 0
    total_count = 0
    oov = dict()
    for i in range(1, vocab_size):
        w = iw[i]
        if w in coverage:
            coverage_vocab += 1
            coverage_count += wc[w]
        else:
            oov[w] = wc[w]
        total_count += wc[w]
        
    print("Coverage in vocab: {}".format(coverage_vocab / vocab_size))
    print("Coverage in word_count: {}".format(coverage_count / total_count))
    return(oov)

if __name__ == "__main__":
    path_to_glove_file = "./glove.6B/glove.6B.100d.txt"
    
    embeddings_index = {}
    with open(path_to_glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    
    #Initialize embedding matrix
    embed_dim = 100
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    coverage = set()
    for i in range(1, vocab_size):
        w = tokenizer.index_word[i]
        if w in embeddings_index.keys():
            embedding_matrix[i] = embeddings_index.get(w) #use the pre-trained embeddings, set 0 otherwise
            coverage.add(w)
            
    oov = check_coverage(tokenizer, coverage, vocab_size=vocab_size)
    print(oov)

"""
3. Build the deep-learning model.
"""
#Model definition
def define_model(max_length, vocab_size, dp_rate=0.1, embed_size=100, embedding_matrix=None):
    img_inputs = keras.layers.Input(shape=(4096,))
    img_dp1 = keras.layers.Dropout(rate=dp_rate)(img_inputs)
    img_dense = keras.layers.Dense(64)(img_dp1)
    img_bn1 = keras.layers.BatchNormalization()(img_dense)
    img_outputs = keras.layers.Activation(activation='relu')(img_bn1)
    #RepeatVector for matching timestep
    img_rep = keras.layers.RepeatVector(n=max_length)(img_outputs)
    
    text_inputs = keras.layers.Input(shape=(max_length,))
    if embedding_matrix is None:
        text_embed = keras.layers.Embedding(input_dim=vocab_size, 
                                            output_dim=embed_size, 
                                            mask_zero=True,
                                            name='text_embedding')(text_inputs)
        
    else:
        text_embed = keras.layers.Embedding(input_dim=vocab_size, 
                                            output_dim=embed_size, 
                                            mask_zero=True,
                                            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                            trainable=False,
                                            name='text_embedding')(text_inputs)
            
    text_dp1 = keras.layers.Dropout(rate=dp_rate)(text_embed)
    text_lstm = keras.layers.LSTM(64, return_sequences=True)(text_dp1)
    
    decoder_inputs = keras.layers.Concatenate()([img_rep, text_lstm])
    decoder_dp1 = keras.layers.Dropout(rate=dp_rate, noise_shape=(None, 1, 128))(decoder_inputs)
    decoder_dense1 = keras.layers.Dense(256)(decoder_dp1)
    decoder_dp2 = keras.layers.Dropout(rate=dp_rate, noise_shape=(None, 1, 256))(decoder_dense1)
    decoder_relu1 = keras.layers.Activation(activation='relu')(decoder_dp2)
    decoder_outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder_relu1)
    
    model = keras.models.Model(inputs = [img_inputs, text_inputs], outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    return(model)

"""
4. Progressive model training
"""

def generate_dataset(caption_dict, features, tokenizer, max_length, vocab_size, num_photos=32):
    """
    A generator of dataset for model training / validation.
    Input: train / val (caption_dict & features)
    Yield: a batch of [[X_img, X_text], Y] as the model input for model.fit_generator() / model.evaluate_generator()
    """
    
    #list of photo_ids
    photo_ids = list(caption_dict.keys())
    while 1:
        s = np.random.choice(np.arange(len(photo_ids)), size=len(photo_ids), replace=False)
        
        count = 0 #counter
        X_img = []
        X_text = []
        Y = []
        while 1:
            #sample the photo_id along the random sequence
            s1 = s[count]
            photo_id = photo_ids[s1]
            caption_list = caption_dict[photo_id]
            
            #sample the caption randomly
            s2 = np.random.choice(np.arange(len(caption_list)), size=1, replace=True)[0]
            caption = caption_list[s2]
            encoded = tokenizer.texts_to_sequences([caption])[0]
            
            tmp_text, tmp_Y = encoded[:-1], encoded[1:]
            padded_text = keras.preprocessing.sequence.pad_sequences([tmp_text], maxlen=max_length, padding='pre')[0]
            padded_Y = keras.preprocessing.sequence.pad_sequences([tmp_Y], maxlen=max_length, padding='pre')[0]
            padded_Y = tf.keras.utils.to_categorical(padded_Y, num_classes=vocab_size)
            
            X_img.append(features[photo_id][0])
            X_text.append(padded_text)
            Y.append(padded_Y)
            
            #Generate until: num_photos photos; iterate through all photos + restart
            count += 1
            if count % num_photos == 0:
                yield([[np.array(X_img), np.array(X_text)], np.array(Y)])
                
                X_img = []
                X_text = []
                Y = []
            if count >= len(photo_ids):
                if len(Y) > 0:
                    yield([[np.array(X_img), np.array(X_text)], np.array(Y)])
                break

#Training:
if __name__ == "__main__":                
    num_epoches = 10
    num_photos = 32 #photos per batch
    
    steps_per_epoch = int(len(train_caption_dict) / num_photos)
    val_steps = int(len(dev_caption_dict) / num_photos)
    
    model = define_model(max_length, vocab_size, dp_rate=0.1, embed_size=embed_dim, embedding_matrix=embedding_matrix)
    
    train_generator = generate_dataset(train_caption_dict, train_features, tokenizer, max_length, vocab_size, num_photos)
    dev_generator = generate_dataset(dev_caption_dict, dev_features, tokenizer, max_length, vocab_size, num_photos)
    
    for i in range(num_epoches):
        hist = model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, epochs=5, verbose=1)
        
        train_loss = hist.history['loss'][-1]
        dev_loss = model.evaluate_generator(dev_generator, steps = val_steps, verbose=1)
        print("The dev_loss at {i}-th epoch: {dev_loss}".format(i=i, dev_loss=np.round(dev_loss,2)))
        
        model.save("../model/model_v{i}.h5".format(i=i))

"""
5. Model Evaluation.
It is evaluated based on BLEU score, which is a kind of generation metrics (measure how close the generated caption is w.r.t. the reference captions).
Also, I try to visualize the captions using the images from the testing dataset.
"""

def sample_caption(model, tokenizer, max_length, vocab_size, feature):
    """
    Input: model, photo feature: shape=[1,4096]
    Return: A generated caption of that photo feature. Remove the startseq and endseq token.
    """
    
    caption = "<startseq>"
    while 1:
        #Prepare input to model
        encoded = tokenizer.texts_to_sequences([caption])[0]
        padded = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=max_length, padding='pre')[0]
        padded = padded.reshape((1, max_length))
        
        pred_Y = model.predict([feature, padded])[0,-1,:]
        next_word = tokenizer.index_word[pred_Y.argmax()]
        
        #Update caption
        caption = caption + ' ' + next_word
        
        #Terminate condition: caption length reaches maximum / reach endseq
        if next_word == '<endseq>' or len(caption.split()) >= max_length:
            break
    
    #Remove the (startseq, endseq)
    caption = caption.replace('<startseq> ', '')
    caption = caption.replace(' <endseq>', '')
    
    return(caption)

def evaluate_model(model, tokenizer, test_caption_dict, test_features, max_length, vocab_size):
    """
    Print: The evaluation score based on BLEU. 
    Also, sample 3 captions using the test images.
    """
    hypo_captions_list = []
    ref_captions_list = []
    disp_captions_dict = {}
    for photo_id, caption_list in test_caption_dict.items():
        #Reference texts
        ref_captions = []
        for c in caption_list:
            ref_captions.append(c.split())
        ref_captions_list.append(ref_captions)
        
        #Sampled texts
        feature = test_features[photo_id]
        samp_caption = sample_caption(model, tokenizer, max_length, vocab_size, feature)
        hypo_captions_list.append(samp_caption.split())
        
        #save the sampled caption for each test photo
        disp_captions_dict[photo_id] = samp_caption
    
    #Evaluate the model
    bleu1 = np.round(corpus_bleu(ref_captions_list, hypo_captions_list, weights=(1, 0, 0, 0)), 2)
    bleu2 = np.round(corpus_bleu(ref_captions_list, hypo_captions_list, weights=(0.5, 0.5, 0, 0)), 2)
    bleu3 = np.round(corpus_bleu(ref_captions_list, hypo_captions_list, weights=(0.3, 0.3, 0.3, 0)), 2)
    bleu4 = np.round(corpus_bleu(ref_captions_list, hypo_captions_list, weights=(0.25, 0.25, 0.25, 0.25)), 2)
    
    print("BLEU Score on Test Set: {b1}, {b2}, {b3}, {b4}".format(b1=bleu1, b2=bleu2, b3=bleu3, b4=bleu4))
    
    #Visualize the images and captions
    photo_ids = list(test_caption_dict.keys())
    np.random.seed(1)
    samples = np.random.choice(np.arange(len(photo_ids)), 5, replace=False)
    
    for i in range(len(samples)):
        photo_id = photo_ids[samples[i]]
        
        fn = "Flickr8k_Dataset/" + photo_id + '.jpg'
        img = keras.preprocessing.image.load_img(fn)
        plt.figure(i+1)
        plt.imshow(img)
        plt.figtext(0.5, 0.01, disp_captions_dict[photo_id], wrap=True, horizontalalignment='center', fontsize=12)
        
if __name__ == "__main__":
    evaluate_model(model, tokenizer, test_caption_dict, test_features, max_length, vocab_size)
    
#Fine-tuning on embedding layers
if __name__ == "__main__":
    model.get_layer('text_embedding').trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    for i in range(5):
        hist = model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, epochs=5, verbose=1)
        
        train_loss = hist.history['loss'][-1]
        dev_loss = model.evaluate_generator(dev_generator, steps = val_steps, verbose=1)
        print("The dev_loss at {i}-th epoch: {dev_loss}".format(i=i, dev_loss=np.round(dev_loss,2)))
        
        model.save("../model/model_v{i}.h5".format(i=i+num_epoches))
        evaluate_model(model, tokenizer, test_caption_dict, test_features, max_length, vocab_size)
