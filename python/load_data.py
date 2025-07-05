import pandas as pd

def load_and_prepare_data():
    # Load dataset
    try: 
        train_df = pd.read_csv('data/multiclass_kdd_train.csv')
        test_df = pd.read_csv('data/multiclass_kdd_test.csv')
    except FileNotFoundError:
        print("Dataset files not found. Please ensure the files are in the 'data' directory.")
        return None, None, None, None
    
    # Separate the data
    y_train = train_df['attack_label']
    x_train = train_df.drop('attack_label', axis=1)
    y_test = test_df['attack_label']
    x_test = test_df.drop('attack_label', axis=1)

    print("Data loaded successfully.")
    return x_train, y_train, x_test, y_test

# Test data
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    if x_train is not None:
        print(f"\nTraining features (x_train) shape: {x_train.shape}")
        print(f"Training labels (y_train) shape: {y_train.shape}")
        print(f"Test features (x_test) shape: {x_test.shape}")
        print(f"Test labels (y_test) shape: {y_test.shape}")