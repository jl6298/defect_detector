import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from keras_tuner import RandomSearch
# 假设数据集已加载到变量X和y中
# X是输入数据，形状为(num_samples, 50)
# y是标签，形状为(num_samples, )

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化输入数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义一个函数用于创建模型，这在Keras Tuner中是必需的
def build_model(hp):
    model = Sequential()
    # Layer 1
    model.add(Dense(hp.Int('units_1', min_value=32, max_value=128, step=32), input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Layer 2
    model.add(Dense(hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(hp.Int('units_3', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Layer 3
    model.add(Dense(hp.Int('units_4', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_4', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Output Layer for multiclass (adjust units based on number of classes)
    model.add(Dense(len(np.unique(y)), activation='softmax'))  # num_classes is the number of unique labels in y
    
    # Compile the model with categorical_crossentropy for multiclass classification
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy',  # or 'categorical_crossentropy'
                  metrics=['accuracy'])
    
    return model



# 使用Keras Tuner进行超参数搜索
tuner = RandomSearch(build_model,
                     objective='val_accuracy',
                     max_trials=20,
                     executions_per_trial=2,
                     directory='hyperparam_tuning',
                     project_name='slm_3d_printing')

# 提前停止回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 运行超参数搜索
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# 获取最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# 使用最佳超参数构建并训练模型
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 在测试集上评估模型
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# 打印分类报告和准确率
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
