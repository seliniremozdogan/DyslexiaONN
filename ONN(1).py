import pandas as pd
from IPython.display import display
import numpy as np
import os
import json
import ast
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras import layers, models, regularizers # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# Example structure in SelfONN.py
from fastonn.SelfONN import SelfONN1d  # Check if this matches your module structure
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras import backend as K # type: ignore
from keras.callbacks import ModelCheckpoint, Callback # type: ignore


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_accuracy', mode='max', verbose=1):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = -np.Inf if mode == 'max' else np.Inf
        self.best_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        current_loss = logs.get('val_loss')

        if current is None:
            return

        if self.mode == 'max':
            if current > self.best or (current == self.best and current_loss < self.best_loss):
                self.best = current
                self.best_loss = current_loss
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: {self.monitor} improved to {current}, saving model to {self.filepath}')
                self.model.save_weights(self.filepath)
        else:
            if current < self.best or (current == self.best and current_loss < self.best_loss):
                self.best = current
                self.best_loss = current_loss
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: {self.monitor} improved to {current}, saving model to {self.filepath}')
                self.model.save_weights(self.filepath)

import tensorflow as tf
from tensorflow.keras import layers # type: ignore

class SelfONN1dKeras(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, q=1, padding_mode='zeros', mode='fast', dropout=None, **kwargs):
        super(SelfONN1dKeras, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.q = q
        self.padding_mode = padding_mode
        self.mode = mode
        self.dropout = dropout

        self.conv = layers.Conv1D(out_channels, kernel_size, strides=stride, padding=padding, dilation_rate=dilation, groups=groups, use_bias=bias)
    
    def call(self, inputs):
        x = tf.concat([tf.math.pow(inputs, i) for i in range(1, self.q + 1)], axis=-1)
        x = self.conv(x)
        return x

    def get_config(self):
        config = super(SelfONN1dKeras, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
            'q': self.q,
            'padding_mode': self.padding_mode,
            'mode': self.mode,
            'dropout': self.dropout
        })
        return config

def create_multi_input_cnn_model(input_shapes, num_classes=1):
    # Model for series1
    input_layer_1 = layers.Input(shape=input_shapes['series1'], name='input_series1')
    conv1_1 = SelfONN1dKeras(in_channels=input_shapes['series1'][1], out_channels=16, kernel_size=5, q=3)(input_layer_1)
    batchnorm1_1 = layers.BatchNormalization()(conv1_1)
    maxpool1_1 = layers.MaxPooling1D(7)(batchnorm1_1)
    dropout1_1 = layers.Dropout(0.4)(maxpool1_1)  # Dropout added
    
    conv2_1 = SelfONN1dKeras(in_channels=16, out_channels=32, kernel_size=3, q=3)(dropout1_1)
    batchnorm2_1 = layers.BatchNormalization()(conv2_1)
    maxpool2_1 = layers.MaxPooling1D(7)(batchnorm2_1)
    dropout2_1 = layers.Dropout(0.4)(maxpool2_1)  # Dropout added

    flat_1 = layers.Flatten()(dropout2_1)
    
    # Model for series2
    input_layer_2 = layers.Input(shape=input_shapes['series2'], name='input_series2')
    conv1_2 = SelfONN1dKeras(in_channels=input_shapes['series2'][1], out_channels=16, kernel_size=5, q=3)(input_layer_2)
    batchnorm1_2 = layers.BatchNormalization()(conv1_2)
    maxpool1_2 = layers.MaxPooling1D(7)(batchnorm1_2)
    dropout1_2 = layers.Dropout(0.4)(maxpool1_2)  # Dropout added

    conv2_2 = SelfONN1dKeras(in_channels=16, out_channels=32, kernel_size=3, q=3)(dropout1_2)
    batchnorm2_2 = layers.BatchNormalization()(conv2_2)
    maxpool2_2 = layers.MaxPooling1D(7)(batchnorm2_2)
    dropout2_2 = layers.Dropout(0.4)(maxpool2_2)  # Dropout added

    flat_2 = layers.Flatten()(dropout2_2)
    
    # Model for vector data
    input_layer_3 = layers.Input(shape=input_shapes['vector'], name='input_vector')
    flat_3 = layers.Flatten()(input_layer_3)
    dense_3 = layers.Dense(32, activation='tanh')(flat_3)
    batchnorm3 = layers.BatchNormalization()(dense_3)
    dropout3 = layers.Dropout(0.3)(batchnorm3)  # Dropout added
    scaled_dropout3 = layers.Lambda(lambda x: x * 2.0)(dropout3)

    # Concatenate the outputs
    concatenated = layers.Concatenate()([flat_1, flat_2, scaled_dropout3])
    
    # Regularized fully connected layer
    dense4 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l1(0.01))(concatenated)
    dense5 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.01))(dense4)
    dense6 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01))(dense5)
    final_dense = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01))(dense6)
    batchnorm_dense = layers.BatchNormalization()(final_dense)
    dropout_dense = layers.Dropout(0.5)(batchnorm_dense)  # Dropout added

    output = layers.Dense(1, activation='sigmoid')(dropout_dense)

    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2, input_layer_3], outputs=output)

    return model

def BinarySearch(cumulated_data_df):
    sum = 0
    i = 0
    while sum < 30.0:
        sum += float(cumulated_data_df['calculated_time'].iloc[i])
        i += 1
    return i

def read_data(json_files):
    ID_list = {}
    vector_list = {}
    for json_file in json_files:
        file_path = os.path.join(dataset_dir, json_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            student_id = data['StudentID']
            student_id = str(student_id)
            student_class = data['StudentClass']

            point_data_df = pd.DataFrame([item for item in data['point_data']])
            cumulated_data_df = pd.DataFrame([item for item in data['cumulated_data']])
            
            point_data_df['duration'] = point_data_df['duration'].astype(float)
            point_data_df['hit_start'] = point_data_df['hit_start'].astype(np.int64)
            point_data_df['hit_end'] = point_data_df['hit_end'].astype(np.int64)
            hit_start_df = point_data_df[['hit_start']].copy()
            hit_end_df = point_data_df[['hit_end']].copy()
            hit_start_df = hit_start_df[hit_start_df['hit_start'] != 0]
            hit_end_df = hit_end_df[hit_end_df['hit_end'] != 0]
            time_spent = hit_end_df['hit_end'].iloc[-1] - hit_start_df['hit_start'].iloc[0]
            
            idx = BinarySearch(cumulated_data_df)
            
            point_data_df = pd.DataFrame([item for item in data['point_data'] if item['coord'] != "(0, 0)"])
            point_data_df['duration'] = point_data_df['duration'].astype(float)
            point_data_df['hit_start'] = point_data_df['hit_start'].astype(np.int64)
            point_data_df['hit_end'] = point_data_df['hit_end'].astype(np.int64)


            point_data_df['coord'] = point_data_df['coord'].apply(ast.literal_eval)

            # Flatten coordinate tuples into separate columns
            point_data_df[['X', 'Y']] = pd.DataFrame(point_data_df['coord'].tolist(), index=point_data_df.index)
            point_data_df.drop('coord', axis=1, inplace=True)

            # Calculate distance (türev) from previous point
            point_data_df['yol'] = (point_data_df['X'] - point_data_df['X'].shift(1))
            point_data_df['yol'] = point_data_df['yol'].fillna(0)
            max_yol = point_data_df['yol'].max()
            min_yol = point_data_df['yol'].min()
            avg_yol = point_data_df['yol'].mean()

            coord_series = point_data_df[['X','yol']]

            x_threshold = 10
            y_threshold = 10

            # Initialize counters and flags
            x_counter = 0
            y_counter = 0
            x_decreasing_flag = False

            # Iterate through points to update counters
            for i in range(1, len(point_data_df)):
                # X coordinate counter
                if (point_data_df.loc[i, 'X'] - point_data_df.loc[i-1, 'X']) < -x_threshold:
                    if not x_decreasing_flag:
                        x_counter += 1
                        x_decreasing_flag = True
                elif (point_data_df.loc[i, 'X'] - point_data_df.loc[i-1, 'X']) > 0:
                    x_decreasing_flag = False
                
                # Y coordinate counter
                if abs(point_data_df.loc[i, 'Y'] - point_data_df.loc[i-1, 'Y']) > y_threshold:
                    y_counter += 1
            
            
            #total_length = point_data_df['türev'].sum()
            std_x = point_data_df['X'].std()
            
            text_no = data.get('TextID')


            if student_id not in ID_list:
                vector = [None,None]
                ID_list[student_id] = coord_series
                vector_list[student_id] = vector

                if text_no == 1:
                    vector_list[student_id][0] = [std_x, y_counter, x_counter, time_spent, max_yol, min_yol, avg_yol, idx, student_class]
                elif text_no == 3:
                    vector_list[student_id][1] = [std_x, y_counter, x_counter, time_spent, max_yol, min_yol, avg_yol, idx, student_class]
            else:
                existing_value = ID_list[student_id]
                if isinstance(existing_value, pd.DataFrame):
                    existing_value = [existing_value]
                if text_no == 1:
                    existing_value.insert(0,coord_series)
                    vector_list[student_id][0] = [std_x, y_counter, x_counter, time_spent, max_yol, min_yol, avg_yol, idx, student_class]
                elif text_no == 3:
                    existing_value.append(coord_series)
                    vector_list[student_id][1] = [std_x, y_counter, x_counter, time_spent, max_yol, min_yol, avg_yol, idx, student_class]

                ID_list[student_id] = existing_value

    return ID_list, vector_list

#Data Hazırlama
##############################################################################################

status_df = pd.read_csv('Veriler_2025_02_17.csv')

dataset_dir = 'Veri/EslestirilenVeriler'

json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

#Get the point dataframe for each individual student
Students, vector = read_data(json_files=json_files)

for key, value in Students.items():
    if len(value) != 2:
        print(f"{key}: {len(Students[key])} values found.")

arrays = [np.array(v) for v in vector.values()]

vector = np.array(arrays)

status_df.drop_duplicates(subset=['StudentId'], inplace=True)

status_df['Coords'] = pd.Series()

status_df['StudentId'] = status_df['StudentId'].astype(str)


X_data = []
y_labels = []
student_ids = []

for student_id in Students.keys():
    if student_id in status_df['StudentId'].values:
        status_knowledge = status_df[status_df['StudentId'] == student_id]['IkiliStatus'].values[0]
        if status_knowledge != 1:
            status_knowledge = 2
        y_labels.append(status_knowledge)
        student_ids.append(student_id)
    else:
        print(f"Student ID {student_id} not found in status_df.")

label_encoder = LabelEncoder()

y_all_encoded = label_encoder.fit_transform(y_labels)


#Veri Temizleme
lengths_series1 = [len(pair[0]) for pair in Students.values()]
lengths_series2 = [len(pair[1]) for pair in Students.values()]

def outlier_cutoff(lengths):
    q1, q3 = np.percentile(lengths, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

lower1, upper1 = outlier_cutoff(lengths_series1)
lower2, upper2 = outlier_cutoff(lengths_series2)

non_outlier_lengths1 = [l for l in lengths_series1 if lower1 <= l <= upper1]
non_outlier_lengths2 = [l for l in lengths_series2 if lower2 <= l <= upper2]

max_non_outlier_length1 = max(non_outlier_lengths1)
max_non_outlier_length2 = max(non_outlier_lengths2)

padded_series = {}
for key, (series1, series2) in Students.items():
    padded_series[key] = (pad_sequences([series1], maxlen=max_non_outlier_length1, padding='post', truncating='post')[0],
                          pad_sequences([series2], maxlen=max_non_outlier_length2, padding='post', truncating='post')[0])

first_elements = []
second_elements = []

for key, value in padded_series.items():
    if len(value) == 2:  
        first_elements.append(value[0])
        second_elements.append(value[1])

first_series_array = np.array(first_elements)
second_series_array = np.array(second_elements)


X_data = np.concatenate([first_series_array, second_series_array], axis=1)
y_all_encoded = np.array(y_all_encoded)
print(y_all_encoded)
print(X_data.shape)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

input_shape_series1 = first_series_array.shape[1:]
input_shape_series2 = second_series_array.shape[1:]

print(input_shape_series1, input_shape_series2)

best_accuracy = 0
best_model = None

all_results = [] 


for fold, (train_index, test_index) in enumerate(kf.split(X_data, y_all_encoded)):
    X_train_split, X_test_split = X_data[train_index], X_data[test_index]
    y_train, y_test = y_all_encoded[train_index], y_all_encoded[test_index]
    vector_train, vector_test = vector[train_index], vector[test_index]
    student_ids_test = np.array(student_ids)[test_index]  # Keep track of student IDs for the test set


    model = create_multi_input_cnn_model({'series1': input_shape_series1, 'series2': input_shape_series2, 'vector': vector_train[1].shape})
    model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
    
    # Define a checkpoint callback to save the model with the best validation accuracy and lowest validation loss in case of ties
    checkpoint_filepath = f'checkpoint_fold_{fold + 1}.weights.h5'
    checkpoint_callback = CustomModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', verbose=1)
    
    history = model.fit([X_train_split[:, :input_shape_series1[0]], X_train_split[:, input_shape_series1[0]:], vector_train], y_train,
              epochs=200, batch_size=32,
              validation_data=([X_test_split[:, :input_shape_series1[0]], X_test_split[:, input_shape_series1[0]:], vector_test], y_test),
              callbacks=[checkpoint_callback])

    # Find the best epoch based on the highest validation accuracy and lowest validation loss in case of ties
    best_epoch = np.argmax(history.history['val_accuracy'])
    if history.history['val_accuracy'][best_epoch] > best_accuracy or \
       (history.history['val_accuracy'][best_epoch] == best_accuracy and history.history['val_loss'][best_epoch] < best_loss):
        best_accuracy = history.history['val_accuracy'][best_epoch]
        best_loss = history.history['val_loss'][best_epoch]
        best_fold = fold + 1
        best_model = model
        # Load the best weights
        model.load_weights(checkpoint_filepath)
        
        test_loss, test_accuracy = model.evaluate([X_test_split[:, :input_shape_series1[0]], X_test_split[:, input_shape_series1[0]:], vector_test], y_test)
        y_pred_prob = model.predict([X_test_split[:, :input_shape_series1[0]], X_test_split[:, input_shape_series1[0]:], vector_test])
        
        predicted_classes = (y_pred_prob > 0.5).astype(int)
        converted_classes = [0 if cls == 0 else 1 for cls in predicted_classes]

        formatted_pred_probs = np.round(y_pred_prob, 2)

        precision = precision_score(y_test, converted_classes, average='weighted')
        recall = recall_score(y_test, converted_classes, average='weighted')
        f1 = f1_score(y_test, converted_classes, average='weighted')


        results_df = pd.DataFrame({
            'StudentID': student_ids_test,
            'Predicted Label': converted_classes,
            'Actual Label': y_test,
            'Disleksi ihtimali':  np.round(1 - y_pred_prob[:, 0], 4),
            'Fold': fold + 1 

        })
            
        display(results_df)




if best_model:
    best_model.summary()
    best_model.save(f'Modeller/ONN modeller/ONN3_weighted2.keras')
    print(f'Best model saved with accuracy on derivative: {best_accuracy}')
    with open('results.txt', 'a') as f:
        f.write(f'Model Name: ONN3 weighted\n')
        f.write(f'CNN only x\n Inputs: X, Yol and vector\n')
        f.write(f'Model information: \n')
        f.write(f'    Filters: 16, 32\n')
        f.write(f'    Max Pooling Size: 2\n')
        f.write(f'    Kernel Size: 7, 5\n')
        f.write(f'    Batch Size: 32\n')
        f.write(f'    Optimizer: Adam\n')
        f.write(f'    Loss: SparseCategoricalCrossentropy\n')
        f.write(f'Fold {best_fold}: \n')
        f.write(f'    Accuracy: {best_accuracy:.4f}\n')
        f.write(f'    Precision: {precision:.4f}\n')
        f.write(f'    Recall: {recall:.4f}\n')
        f.write(f'    F1 Score: {f1:.4f}\n\n')
        f.write(f'********************************\n\n')