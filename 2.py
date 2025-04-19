import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import joblib

class HotelBookingOOP:
    def __init__(self, file_path, random_state=42):
        self.file_path = file_path
        self.random_state = random_state
        self.df = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.rf_model = RandomForestClassifier(random_state=self.random_state)

    def load_and_clean_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df = self.df.dropna()
        self.df = self.df.drop(columns=['Booking_ID'])

    def encode_target(self):
        self.df['booking_status'] = self.label_encoder.fit_transform(self.df['booking_status'])

    def split_data(self):
        self.x = self.df.drop(columns=['booking_status'])
        self.y = self.df['booking_status']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=self.random_state
        )

    def encode_categoricals(self):
        cols_to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        encoded_train = self.encoder.fit_transform(self.x_train[cols_to_encode])
        encoded_test = self.encoder.transform(self.x_test[cols_to_encode])

        encoded_train_df = pd.DataFrame(encoded_train, columns=self.encoder.get_feature_names_out(cols_to_encode))
        encoded_test_df = pd.DataFrame(encoded_test, columns=self.encoder.get_feature_names_out(cols_to_encode))

        self.x_train = self.x_train.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)

        self.x_train = pd.concat([self.x_train.drop(columns=cols_to_encode), encoded_train_df], axis=1)
        self.x_test = pd.concat([self.x_test.drop(columns=cols_to_encode), encoded_test_df], axis=1)

    def encode_arrival_year(self):
        if 'arrival_year' in self.x_train.columns:
            self.x_train['arrival_year'] = self.label_encoder.fit_transform(self.x_train['arrival_year'])
            self.x_test['arrival_year'] = self.label_encoder.transform(self.x_test['arrival_year'])

    def scale_features(self):
        cols_to_scale = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights',
                         'required_car_parking_space','lead_time','arrival_month','arrival_date',
                         'repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled',
                         'avg_price_per_room','no_of_special_requests']

        self.x_train[cols_to_scale] = self.scaler.fit_transform(self.x_train[cols_to_scale])
        self.x_test[cols_to_scale] = self.scaler.transform(self.x_test[cols_to_scale])

    def train_model(self):
        self.rf_model.fit(self.x_train, self.y_train)

    def evaluate(self):
        predictions = self.rf_model.predict(self.x_test)
        print("Accuracy:", accuracy_score(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))

    def save_model(self, model_file='rf_model_compressed.pkl', model_path='./'):
        model_full_path = os.path.join(model_path, model_file)
        joblib.dump(self.rf_model, model_full_path, compress=3)

if __name__ == '__main__':
    path = 'C:\\Users\\Asus\\Documents\\Code\\.vscode\\UTS ModelDeployment\\Dataset_B_hotel.csv'
    model = HotelBookingOOP(path)
    model.load_and_clean_data()
    model.encode_target()
    model.split_data()
    model.encode_categoricals()
    model.encode_arrival_year()
    model.scale_features()
    model.train_model()
    model.evaluate()
    save_model_path = 'C:\\Users\\Asus\\Documents\\Code\\.vscode\\UTS ModelDeployment'
    model.save_model(model_file='rf_model.pkl', model_path=save_model_path)