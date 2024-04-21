
#    This code defines a deep learning model for Image-Text Matching (ITM) classification. Here's an overview of its functionality:

#     Libraries and Modules Import: The code imports various libraries and modules necessary for deep learning tasks including TensorFlow, NumPy, Matplotlib, Transformers, and scikit-learn.

#     ITM_DataLoader Class: This class handles data loading, preprocessing, and dataset creation. 
#                         It loads sentence embeddings, preprocesses image and text data, pads sequences, and prepares TensorFlow datasets for training, validation, and testing.

#     ITM_Classifier Class: This class extends ITM_DataLoader and defines methods for building, training, and testing the ITM classifier model. 
#                             It creates vision and text encoders using pre-trained ResNet50V2 and BERT models, respectively. 
#                             The class also implements methods for creating the classifier model, training it with specified parameters, and evaluating its performance.

#     Model Training and Testing: The ITM_Classifier instance is created, which triggers the model building, training, and testing processes.
#                                  The model's performance metrics such as accuracy, AUC, recall, precision, and F1-score are computed and printed.


#     - The Code Has been tested on NVIDIA RTX3070 GPU, Windows 11 
#     - To run the code in Windows, Install Anaconda, Create a virtual environment with Python 3.10.8
#     - Open Virtual Env Terminal run these commands to install Tensorflow with GPU -
#                 First step - conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
#                 Second Step - python -m pip install "tensorflow<2.11"
#                 Thrird Step - python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" [Run this line for GPU Check]
#     - Install Libraries- pip install transformers
#                          pip install tensorflow-addons
#                          pip install scikit-learn
#                          pip install matplotlib
#                          pip install einops
#     - Open New Window in VS Code, open the folder, change the line 59 Img Path to your own path     



import sys  # Module for system-specific parameters and functions
import os  # Module for interacting with the operating system
import time  # Module for time-related functions
import pickle  # Module for serializing and deserializing Python objects
import random  # Module for generating random numbers and selecting random items
import numpy as np  # NumPy library for numerical computations
import tensorflow as tf  # TensorFlow library for deep learning
from tensorflow import keras  # Keras API for building and training neural networks
from tensorflow.keras import layers  # Keras layers for constructing neural network architectures
import matplotlib.pyplot as plt  # Matplotlib library for data visualization
import tensorflow_addons as tfa  # TensorFlow Addons for additional functionality
import sklearn.metrics # Module for computing various machine learning metrics


from transformers import BertTokenizer, TFBertModel # Transformers library for Text Encoding tasks
from tensorflow.keras.applications import ResNet50 # Pre-trained ResNet50 model for  vision encoding tasks



class ITM_DataLoader():
    BATCH_SIZE = 16  # Number of samples per batch during training
    IMAGE_SIZE = (224, 224)  # Size of images for resizing
    IMAGE_SHAPE = (224, 224, 3)  # Shape of images for model input
    SENTENCE_EMBEDDING_SHAPE = (384)  # Shape of sentence embeddings
    max_sentence_length = 50  # Maximum length of sentences
    AUTOTUNE = tf.data.AUTOTUNE  # Autotune parameter for optimization
    IMAGES_PATH = "C:/Users/Computing/Downloads/flickr8k.dataset-cmp9137-item1/flickr8k.dataset-cmp9137-item1/flickr8k-resised"  # Path to image directory
    train_data_file = IMAGES_PATH + "/../flickr8k.TrainImages.txt"  # Path to training data file
    dev_data_file = IMAGES_PATH + "/../flickr8k.DevImages.txt"  # Path to development data file
    test_data_file = IMAGES_PATH + "/../flickr8k.TestImages.txt"  # Path to test data file
    sentence_embeddings_file = IMAGES_PATH + "/../flickr8k.cmp9137.sentence_transformers.pkl"  # Path to sentence embeddings file
    sentence_embeddings = {}  # Dictionary to store sentence embeddings
    train_ds = None  # Placeholder for training dataset
    val_ds = None  # Placeholder for validation dataset
    test_ds = None  # Placeholder for test dataset

    def __init__(self):
        self.sentence_embeddings = self.load_sentence_embeddings()
        self.train_ds = self.load_classifier_data(self.train_data_file)
        self.val_ds = self.load_classifier_data(self.dev_data_file)
        self.test_ds = self.load_classifier_data(self.test_data_file)
        print("done loading data...")

   
    def load_sentence_embeddings(self):
        # Dictionary to store sentence embeddings
        sentence_embeddings = {}
        # Printing a message indicating the start of reading sentence embeddings
        print("READING sentence embeddings...")
        # Opening the file containing sentence embeddings
        with open(self.sentence_embeddings_file, 'rb') as f:
            # Loading data from the file
            data = pickle.load(f)
            # Iterating over each sentence and its corresponding dense vector
            for sentence, dense_vector in data.items():
                # Assigning the dense vector to its corresponding sentence
                sentence_embeddings[sentence] = dense_vector
        # Printing a message indicating the completion of reading sentence embeddings
        print("Done reading sentence_embeddings!")
        # Returning the dictionary of sentence embeddings
        return sentence_embeddings


    def padded_tensor(self, unpadded_tensor):
        # List to store padded tensors
        padded_tensors = []
        # Iterating over each tensor in the input list
        for tensor in unpadded_tensor:
            # Getting the original length of the tensor
            original_length = tf.shape(tensor[0])[0]
            # Calculating the padding length required
            padding_length = self.max_sentence_length - original_length
            # Padding the tensor with zeros at the end
            padded_tensor = tf.concat([tf.cast(tensor[0], tf.int32), tf.zeros([padding_length], tf.int32)], axis=0)
            # Reshaping the padded tensor for compatibility
            reshaped_tensor = tf.reshape(padded_tensor, (1, self.max_sentence_length))
            # Appending the padded tensor to the list
            padded_tensors.append(reshaped_tensor)
        # Returning the list of padded tensors
        return padded_tensors    


    def process_input(self, img_path, text, label, text_input_ids, text_attention_mask):
        # Reading and preprocessing image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.cast(img, tf.float32) / 255
        # Creating a dictionary to hold features
        features = {}
        features["image_input"] = img
        features["text_input_ids"] = text_input_ids[0]
        features['text_attention_mask'] = text_attention_mask[0]
        features["caption"] = text
        features["file_name"] = img_path
        # Returning the dictionary of features along with the label
        return features, label


    def load_classifier_data(self, data_files):
        # Printing a message indicating the start of loading data
        print("LOADING data from " + str(data_files))
        print("=========================================")
        # Initializing a BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Lists to store image data, text data, label data, and BERT input ids and attention masks
        image_data = []
        text_data = []
        label_data = []
        bert_input_ids = []
        bert_attention_mask = []

        # Tokenizing the text data and extracting image, text, and label information
        with open(data_files) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("	")
                img_name = os.path.join(self.IMAGES_PATH, img_name.strip())
                # Converting match/no-match answers to binary labels
                label = [1, 0] if raw_label == "match" else [0, 1]
                image_data.append(img_name)
                text_data.append(text)
                label_data.append(label)
                tokenized_texts = tokenizer(
                    text, padding=True, truncation=True, return_tensors='tf')
                bert_input_ids.append(tokenized_texts['input_ids'])
                bert_attention_mask.append(
                    tokenized_texts['attention_mask'])

        # Padding BERT input ids and attention masks
        bert_input_ids = self.padded_tensor(bert_input_ids)
        bert_attention_mask = self.padded_tensor(bert_attention_mask)

        # Printing information about the loaded data
        print("|image_data|=" + str(len(image_data)), image_data[0])
        print("|text_data|=" + str(len(text_data)), text_data[0])
        print("|label_data|=" + str(len(label_data)), label_data[0])

        # Creating a TensorFlow dataset using the loaded data
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_data, text_data, label_data, bert_input_ids, bert_attention_mask))
        dataset = dataset.shuffle(self.BATCH_SIZE * 8)
        dataset = dataset.map(self.process_input,
                            num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(self.AUTOTUNE)
        # Returning the dataset
        return dataset

    def print_data_samples(self, dataset):
        print("PRINTING data samples...")
        print("-----------------------------------------")
        for features_batch, label_batch in dataset.take(1):
            for i in range(1):
                print(f'Image pixels: {features_batch["image_input"]}')
                print(
                    f'Sentence embeddings: {features_batch["text_embedding"]}')
                print(f'Caption: {features_batch["caption"].numpy()}')
                label = label_batch.numpy()[i]
                print(f'Label : {label}')
        print("-----------------------------------------")


class ITM_Classifier(ITM_DataLoader):
    epochs = 20
    learning_rate = 1e-5
    class_names = {'match', 'no-match'}
    num_classes = len(class_names)
    classifier_model = None
    history = None
    classifier_model_name = 'ITM_Classifier-flickr'

    def __init__(self):
        super().__init__()
        self.build_classifier_model()
        self.train_classifier_model()
        self.test_classifier_model(self.history)

    # return learnt feature representations of input data (images)
    def create_vision_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        res_model = ResNet50(include_top= False, weights="imagenet", input_shape=self.IMAGE_SHAPE)
        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")
        res_x = res_model(img_input, training=True)  # Set training to True for fine-tuning

        # Add more layers
        res_x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(res_x)
        res_x = layers.MaxPooling2D(pool_size=(2, 2))(res_x)
        res_x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(res_x)
        res_x = layers.MaxPooling2D(pool_size=(2, 2))(res_x)
        
        # Apply global average pooling
        res_x = layers.GlobalAveragePooling2D()(res_x)

        outputs = self.project_embeddings(res_x, num_projection_layers, projection_dims, dropout_rate)
        return img_input, outputs

    # return learnt feature representations based on dense layers, dropout, and layer normalisation
    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.relu(projected_embeddings)
            x = layers.Dense(projection_dims, activation='relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    # return learnt feature representations of input data (text embeddings in the form of dense vectors)
    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        # Freeze some BERT layers and make the rest trainable
        for layer in bert_model.layers[:4]:
            layer.trainable = True

        # Input layers
        max_sequence_length = 50
        text_input_ids = tf.keras.Input(shape=(max_sequence_length,), dtype=tf.int32, name='text_input_ids')
        text_attention_mask = tf.keras.Input(shape=(max_sequence_length,), dtype=tf.int32, name='text_attention_mask')
        
        # BERT model
        text_output = bert_model(text_input_ids, attention_mask=text_attention_mask)
        text_features = text_output.last_hidden_state
         # Additional dense layers
        text_features = tf.keras.layers.Dense(128, activation='relu')(text_features)
        text_features = tf.keras.layers.Dropout(dropout_rate)(text_features)
        text_features = tf.keras.layers.Dense(128, activation='relu')(text_features)
        text_features = tf.keras.layers.LayerNormalization()(text_features)
        # Pooling layer
        text_features = tf.keras.layers.GlobalAveragePooling1D()(text_features)
        
        # Project embeddings
        text_features = self.project_embeddings(text_features, num_projection_layers, projection_dims, dropout_rate)
        
       
        return text_input_ids, text_attention_mask, text_features

    # put together the feature representations above to create the image-text (multimodal) deep learning model
    def build_classifier_model(self):
        # Printing a message indicating the start of model building process
        print(f'BUILDING model')
        
        # Creating the vision encoder with specified parameters
        img_input, vision_net = self.create_vision_encoder(
            num_projection_layers=2, projection_dims=512, dropout_rate=0.1)
        
        # Creating the text encoder with specified parameters
        text_input_ids, text_attention_mask, text_net = self.create_text_encoder(
            num_projection_layers=6, projection_dims=512, dropout_rate=0.1)
        
        # Concatenating the outputs of vision and text encoders
        net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])
        net = tf.keras.layers.Dropout(0.1)(net)

        # Adding dense layers with ReLU activation and dropout
        net = tf.keras.layers.Dense(256, activation='relu')(net)
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(256, activation='relu')(net)
        net = tf.keras.layers.Dropout(0.1)(net)
        
        # Final output layer with softmax activation, representing the number of classes
        net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        
        # Creating the classifier model using specified inputs and outputs
        self.classifier_model = tf.keras.Model(inputs=[img_input, text_input_ids, text_attention_mask], outputs=net)
        
        # Setting the layers in the vision encoder to be trainable
        for layer in self.classifier_model.layers[2].layers[:]:
            layer.trainable = True
        
        # Printing summary of the classifier model
        self.classifier_model.summary()

    def train_classifier_model(self):
        # Printing a message to indicate the training process
        print(f'Training model...')
    
        # Calculating steps per epoch
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()

        # Calculating total number of training steps and warmup steps
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.2 * num_train_steps)

        # Defining loss function and evaluation metrics
        loss = tf.keras.losses.KLDivergence()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn')
        ]

        # Defining optimizer with weight decay
        optimizer = tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=0.00)

        # Compiling the classifier model with optimizer, loss function, and metrics
        self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Training time calculation
        start_time = time.time()
        self.history = self.classifier_model.fit(x=self.train_ds, validation_data=self.val_ds, epochs=self.epochs)
        end_time = time.time()
        training_time = end_time - start_time
        
        # Printing training time
        print("Training Time: {:.2f} seconds".format(training_time))
        print("Model trained!")
        return self.history
    def test_classifier_model(self,history):
        print("TESTING classifier model (showing a sample of image-text-matching predictions)...")
        num_classifications = 0
        num_correct_predictions = 0

        # read test data for ITM classification
        for features, groundtruth in self.test_ds:
            groundtruth = groundtruth.numpy()
            predictions = self.classifier_model(features)
            predictions = predictions.numpy()
            captions = features["caption"].numpy()
            file_names = features["file_name"].numpy()

            # read test data per batch
            for batch_index in range(0, len(groundtruth)):
                predicted_values = predictions[batch_index]
                probability_match = predicted_values[0]
                probability_nomatch = predicted_values[1]
                predicted_class = "[1 0]" if probability_match > probability_nomatch else "[0 1]"
                if str(groundtruth[batch_index]) == predicted_class: 
                    num_correct_predictions += 1
                num_classifications += 1

                # print a sample of predictions -- about 10% of all possible
                if random.random() < 0.1:
                    caption = captions[batch_index]
                    file_name = file_names[batch_index].decode("utf-8")
                    print("ITM=%s PREDICTIONS: match=%s, no-match=%s \t -> \t %s" % (caption, probability_match, probability_nomatch, file_name))

        # Calculate AUC, Recall, and other metrics
        predictions = []
        groundtruths = []
        for features, groundtruth in self.test_ds:
            groundtruths.extend(groundtruth.numpy())
            prediction = self.classifier_model.predict(features)
            predictions.extend(prediction)

        predictions = np.array(predictions)
        true_classes = np.argmax(groundtruths, axis=1)

        # Calculate balanced accuracy
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(true_classes, predictions.argmax(axis=1))

        # reveal test performance using our own calculations above
        accuracy = num_correct_predictions/num_classifications
        print("TEST accuracy=%4f" % (accuracy))
        print("Balanced accuracy:", balanced_accuracy)
        print("model tested!")
        auc = sklearn.metrics.roc_auc_score(true_classes, predictions[:, 1])
        recall = sklearn.metrics.recall_score(true_classes, predictions.argmax(axis=1))
        precision = sklearn.metrics.precision_score(true_classes, predictions.argmax(axis=1))
        f1_score = sklearn.metrics.f1_score(true_classes, predictions.argmax(axis=1))

        print("AUC:", auc)
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1-score:", f1_score)
        # Evaluate the classifier model
        results = self.classifier_model.evaluate(self.test_ds)

        # Unpack only loss and accuracy
        loss, accuracy = results[:2]
        print(f'Tensorflow test method: Loss: {loss}; ACCURACY: {accuracy}')
        # Uncomment to see the Training and Validation Loss & Training and Validation Accuracy
        # plt.plot(history.history['loss'], label='Training Loss')
        # plt.plot(history.history['val_loss'], label='Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()
        # plt.plot(history.history['accuracy'], label='Training Accuracy')
        # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        # plt.title('Training and Validation Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()

itm = ITM_Classifier()


#Total params: 138,262,402
#Trainable params: 138,209,282
#Non-trainable params: 53,120

#TEST accuracy=0.802756
#Balanced accuracy: 0.7803089919218849
#model tested!
#AUC: 0.8687005155218654
#Recall: 0.814139110604333
#Precision: 0.9083969465648855
#F1-score: 0.8586891160553217
#73/73 [==============================] - 6s 79ms/step - loss: 0.9971 - accuracy: 0.7967 - precision: 0.7967 - recall: 0.7967 - auc: 0.8605 - tp: 925.0000 - fp: 236.0000 - tn: 925.0000 - fn: 236.0000
#Tensorflow test method: Loss: 0.9971243739128113; ACCURACY: 0.7967269420623779
