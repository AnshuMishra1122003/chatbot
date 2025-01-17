import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, schedules

lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize data structures
words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty) 
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    
# Shuffle and convert training data to arrays
random.shuffle(training)
training = np.array(training, dtype=object)

# Separate features (X) and labels (Y)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create a learning rate schedule
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=64, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Save training history (optional, for later analysis)
with open('training_history.pkl', 'wb') as history_file:
    pickle.dump(hist.history, history_file)

print('Model and training history saved successfully')