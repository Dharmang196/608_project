import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2  # Corrected import for l2 regularization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from io import StringIO

# Normality test
def normality_test(data):
    stat, p = stats.shapiro(data)
    return f'Statistics={stat:.3f}, p={p:.3f}'

# Preprocess data for regression or classification model
def preprocess_data(data, independent_vars, dependent_var):
    # Identify categorical columns
    categorical_cols = data[independent_vars].select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply one-hot encoding to categorical columns
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
    X_transformed = ct.fit_transform(data[independent_vars])

    # Normalize the features
    scaler_X = StandardScaler()
    X_transformed = scaler_X.fit_transform(X_transformed)

    # Handle wide range target variable for regression
    transformer = PowerTransformer(method='yeo-johnson')
    y_transformed = transformer.fit_transform(data[[dependent_var]].values.reshape(-1, 1)).flatten()

    return X_transformed, y_transformed, scaler_X, transformer

# Determine if the task is regression or classification
def determine_task(y):
    if len(np.unique(y)) > 2 or np.issubdtype(y.dtype, np.inexact):
        return 'regression'
    return 'classification'

# Function to create the ANN model with added complexity and regularization
def create_ann_model(X_train, task):
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    if task == 'regression':
        model.add(Dense(1, activation='linear'))
    else:
        model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    return model

# Main function for the Streamlit app
def main():
    st.title('Data Analysis and Model Building')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Select dependent and independent variables
        dependent_var = st.selectbox("Select the dependent variable", data.columns)
        independent_vars = st.multiselect("Select independent variables", data.columns)

        # Run Normality Test
        if st.button('Run Normality Test'):
            if dependent_var:
                st.write(normality_test(data[dependent_var]))
            else:
                st.error("Please select a dependent variable.")

        # Build and Train ANN Model
        if st.button('Build and Train ANN Model'):
            if dependent_var and independent_vars:
                X, y, scaler_X, transformer = preprocess_data(data, independent_vars, dependent_var)
                task = determine_task(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Save model to session state
                st.session_state['ann_model'] = create_ann_model(X_train, task)

                callbacks = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
                history = st.session_state['ann_model'].fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)
                y_pred = st.session_state['ann_model'].predict(X_test)

                if task == 'regression':
                    # Inverse transform the predictions and targets if necessary
                    y_pred = transformer.inverse_transform(y_pred)
                    y_test = transformer.inverse_transform(y_test.reshape(-1, 1)).flatten()

                    test_mse = mean_squared_error(y_test, y_pred)
                    test_mae = mean_absolute_error(y_test, y_pred)
                    test_r2 = r2_score(y_test, y_pred)

                    st.write("ANN Regression Model Summary:")
                    st.write(f"Test MSE: {test_mse}")
                    st.write(f"Test MAE: {test_mae}")
                    st.write(f"Test R²: {test_r2}")
                else:
                    test_accuracy = history.history['val_accuracy'][-1]  # Assuming binary classification for simplicity
                    st.write("ANN Classification Model Summary:")
                    st.write(f"Test Accuracy: {test_accuracy}")

                st.session_state['history'] = history
                st.line_chart(history.history['loss'])
                if task == 'classification':
                    st.line_chart(history.history['accuracy'])
                st.line_chart(history.history['val_loss'])
                if task == 'classification':
                    st.line_chart(history.history['val_accuracy'])

    # Add a button to display the model architecture
    if st.button('Display Model Architecture'):
        if 'ann_model' in st.session_state:
            model_summary = StringIO()
            st.session_state['ann_model'].summary(print_fn=lambda x: model_summary.write(x + '\n'))
            model_summary.seek(0)
            st.text(model_summary.read())
        else:
            st.error('The model has not been built yet. Please build the model first by pressing the "Build and Train ANN Model" button.')

if __name__ == "__main__":
    main()
