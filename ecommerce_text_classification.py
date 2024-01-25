#%%
#1. Setup - Importing packages
import pandas as pd 
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications, models
import matplotlib.pyplot as plt
import sklearn
import os, pickle
import datetime
#%%
#2. Data loading
URL= 'C:\\Users\\ruzan\\Documents\\Ruzana\\SHRDC\\DL\\Hands_On\\dlts\\natural_language_processing\\e-commerce\\ecommerceDataset.csv'
#URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df = pd.read_csv(URL)

#%%
#3. Data inspection
print("Shape of the data: ", df.shape)
print("Data info:\n ", df.info())
print("Data description:\n", df.describe().transpose())
print("Example data:\n", df.head(1))
#%%
#4. Data cleaning
#Only include df with category of "Electronics", "Household", "Books", "Clothing & Accessories"
df.columns = ['category', 'text']
df.columns
valid_categories = ["Electronics", "Household", "Books", "Clothing & Accessories"]
df = df[df['category'].isin(valid_categories)]

#%%
print(df.isna().sum())
print("--------------------")
print(df.duplicated().sum())

#%%
#drop na value
df = df.dropna()

#%%
#5. Dealing with duplicates 
categories_list = df['category'].unique()
print(df["category"].value_counts())

#%%
#6. Remove the duplicates and see the class representations again
df_no_duplicates = df.drop_duplicates() 
print(df_no_duplicates['category'].value_counts())
#%%
df.columns

#%%
#7. Data preprocessing 
#7.1. Split the data into features and labels
features = df_no_duplicates['text'].values
labels = df_no_duplicates['category'].values

#%%
#7.2 Convert the categorical label into integer - label encoding
from sklearn. preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_oh_encoder = OneHotEncoder()
labels_oh_encoded = labels_oh_encoder.fit_transform(labels.reshape(-1,1))

#%%
#7.3 Perform train test split
from sklearn.model_selection import train_test_split
seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, train_size=0.8, random_state=seed)

#%%
#7.4 Process the input texts
#(A) Tokenization
#Define parameters for the following process
vocab_size = 5000
oov_token = '<OOV>'
max_length = 200
embedding_dim = 64

#(B) Define the Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words = vocab_size,
    split=" ",
    oov_token=oov_token
)
tokenizer.fit_on_texts(X_train)
#%%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

#%%
#(C) Transform texts into tokens
X_train_tokens =  tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

#%%
#(D) Perform padding
X_train_padded = keras.utils.pad_sequences(X_train_tokens, maxlen=max_length, padding="post", truncating="post")
X_test_padded = keras.utils.pad_sequences(X_test_tokens, maxlen=max_length, padding="post", truncating="post")

#%%
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_tokens(tokens):
    return " ".join([reverse_word_index.get(i,"?") for i in tokens])

print(X_train[2])
print("---------------------")
print(decode_tokens(X_train_padded[2]))
#%%
#8. Model development
#(A) Create a sequential model, then start with embedding layer
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim))
#(B) Build RNN model, here we use bidirectional LSTM
model.add(keras.layers.Bidirectional(keras.layers.LSTM(48)))
model.add(keras.layers.Dense(48, activation='relu'))
model.add(keras.layers.Dense(48, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(len(np.unique(labels)), activation='softmax'))
model.add(keras.layers.Dropout(0.3))
model.summary()

#Plot model architecture
model_plot = tf.keras.utils.plot_model(model, show_dtype=True, 
                       show_layer_names=True, show_shapes=True,  
                       to_file='model.png')
  
model_plot
#%%
#9. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%
#Prepare the callback objects for model.fit()
PATH = os.getcwd()
logpath = os.path.join(PATH,"tensorboard_log")
datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = keras.callbacks.TensorBoard(logpath)

#%%
#10. Model training
max_epoch = 20
early_stopping = keras.callbacks.EarlyStopping(patience=5)
history = model.fit(X_train_padded, y_train, 
                    validation_data=(X_test_padded, y_test), 
                    epochs=max_epoch, callbacks=[early_stopping,tb])
#%%
#11. Plot graphs to display training result
#(A) Loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss', 'Validaton Loss'])
plt.show()
#%%
#(B) Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy', 'Validaton Accuracy'])
plt.show()
#%%
#12. Model deployment
#(A) Get an example input
#test_string = r"Tesla is recalling more than 1.6 million vehicles in China over issues with steering software and door-locking systems, the country's regulator says. The recall includes its models S, X, 3 and Y, and 7,538 imported vehicles. The problems will be fixed by remote updates to software, meaning the vehicles will not need to be taken to dealerships or garages. It comes less than a month after Tesla recalled two million cars in the US due to autopilot software issues. In May last year, the Chinese regulator said more than a million vehicles may have acceleration and braking system issues. The American electric car giant then discovered problems with assisted driving functions and door-locking systems. The Chinese regulator, the State Administration for Market Regulation (SAMR), described the planned vehicle update as a recall, even though it will happen remotely. Tesla will release an over-the-air software update for a total of 1,610,105 vehicles, including imported Models S and X and the China-made Models 3 and Y cars made from 2014 to 2023, the SAMR said. The regulator added that this was to tackle issues with the autosteer function and cut the risk of collision."
test_string = r"LINANAS 3-seat sofa, with chaise longue/Vissle dark grey. RM1,495. 10 year guarantee. Product features Firm. Choose cover with chaise longue/Vissle dark grey. Delivery Available. See options at checkout"
#(B) Convert the text into tokens
test_token = tokenizer.texts_to_sequences(test_string)

#%%
#Remove Nan values
def remove_space(token):
    temp=[]
    for i in token:
        if i!=[]:
            temp.append(i[0])
    return temp

test_token_processed = np.expand_dims(np.array(remove_space(test_token)), axis=0)
#%%
#(C) Perform padding and truncating
test_token_padded = keras.preprocessing.sequence.pad_sequences(test_token_processed,
maxlen=max_length, padding='post', truncating='post')
#%%
#(D) Perform prediction using the model
y_pred = np.argmax(model.predict(test_token_padded))
#%%
#(E) Use label encoder to find the class
class_prediction =  label_encoder.inverse_transform(y_pred.flatten())
print(class_prediction)

# %%
#13. Save important components so that we can diploy the 
#NLP model in other application
#(A) Tokenizer
os.chdir(r"C:\Users\ruzan\Documents\Ruzana\SHRDC\DL\Hands_On\dlts\natural_language_processing\e-commerce")
PATH = os.getcwd()
print(PATH)
# %%
tokenizer_save_path = os.path.join(PATH, "tokenizer.pkl")
with open(tokenizer_save_path,"wb") as f:
    pickle.dump(tokenizer,f)

# %%
#(B) Label encoder 
label_encoder_save_path = os.path.join(PATH, "label_encoder.pkl")
with open(label_encoder_save_path,'wb') as f:
    pickle.dump(label_encoder, f)

# %%
#(C) Save the Keras model
model_save_path = os.path.join(PATH,"nlp_model")
keras.models.save_model(model, model_save_path)
    
#%%
#14. Generate classification report
from sklearn.metrics import classification_report

y_pred_new= np.argmax(model.predict(X_test_padded), axis=1)
classification_report = classification_report(y_test, y_pred_new)

# Print the classification report
print(classification_report)

