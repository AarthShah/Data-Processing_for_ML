import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data loaded successfully from", file_path)
    print(df.head(10))
    return df

def process_data(df):
    number_col=df.select_dtypes(include=['number']).columns.tolist()
    print("Numerical columns in the dataset:", number_col)

    # fill missing values in numerical columns with their mean
    for col in number_col:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in {col} with mean: {mean_value}")

    # normalization of numerical columns
    max_value={}
    min_value={}
    for col in number_col:
        max_value[col] = df[col].max()
        min_value[col] = df[col].min()
        df[col]=(df[col]-min_value[col])/(max_value[col]-min_value[col])
    
    print("Normalization of numerical columns completed.",df.head(10))

    # spilit the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.7, random_state=42)
    return train_df, test_df


df= load_data('C:\\Users\\Aarth Shah\\OneDrive\\Desktop\\data-processing\\Data-Processing_or_ML\\Titanic Dataset.csv')
process_data(df)


