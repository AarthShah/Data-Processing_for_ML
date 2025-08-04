import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    df1 = pd.read_csv(
        'C:\\Users\\Aarth Shah\\OneDrive\\Desktop\\data-processing\\airlines_flights_data.csv')
    df = pd.DataFrame(df1)
    print("Data loaded successfully")
    process_data(df)


def process_data(df):
    print(df.head(25))
    print("Processing data...")
    rows, cols = df.shape
    print(f"Data shape: {rows} rows, {cols} columns")

    df_min = df['price'].min()
    df_max = df['price'].max()

    print("Normalizing data...")

    df['price'] = (df['price']-df_min)/(df_max-df_min)
    #print(df[:25])
    print("Data processing complete")

    print("starting to retreat x and y values")

    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)
    # print(train_data.head(25))
    x_train = train_data['duration'].to_numpy().reshape(-1, 1)
    y_train = train_data['price'].to_numpy().reshape(-1, 1)

    x_test = test_data['duration'].to_numpy().reshape(-1, 1)
    y_test = test_data['price'].to_numpy().reshape(-1, 1)

    train_model(x_train, y_train, df_min, df_max, x_test, y_test)
    # print(x_test[:25], y_test[:25])


def train_model(x_train, y_train, df_min, df_max, x_test, y_test):
    print("Training model...")
    np.random.seed(42) 
    w = np.random.rand(1).reshape(-1, 1)
    b = np.random.rand(1)
    n = 0.001
    epochs = 10001
    for epoch in range(epochs):
        y_pred = np.dot(x_train, w)+b
        error = y_pred-y_train
        w_grad = (2/len(x_train))*np.dot(x_train.T, error)
        b_grad = (2/len(x_train))*np.sum(error)
        w = w-n*w_grad
        b = b-n*b_grad

        if epoch % 100 == 0:
            loss = np.mean(error**2)
            print(
                f"Epoch {epoch}, Loss: {loss:.6f}, w: {w.flatten()[0]:.6f}, b: {b[0]:.6f}")
            original_error = loss * (df_max - df_min)
            print(f"Original Error: {original_error:.6f}")
    print("Model trained successfully")
    test_model(x_test, y_test, w, b, df_min, df_max)
   


def test_model(x_test, y_test, w, b, df_min, df_max):
    print("Testing model...")
    y_pred = np.dot(x_test, w)+b
    error = y_pred-y_test
    loss = np.mean(error**2)
    original_error = loss * (df_max - df_min)
    print(f"Test Loss: {loss:.6f}, Original Error: {original_error:.6f}")
    print(f"Model parameters: w={w.flatten()[0]:.6f}, b={b[0]:.6f}")
    print("Model testing complete")
    plt.scatter(x_test, y_test, color='blue', label='Actual Prices')
    plt.scatter(x_test, y_pred, color='red', label='Predicted Prices')
    plt.xlabel('Flight Duration (minutes)')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.show()
    predict_price(w, b, df_min, df_max, x_test)


def predict_price(w, b, df_min, df_max, duration):
    x = float(input("Enter the duration of the flight in minutes: "))
    z = x*w[0][0]+b[0]
    y_pred = (z*(df_max-df_min))+df_min
    print(f"Predicted price for a flight of {x} hours is : {y_pred}")


load_data()
