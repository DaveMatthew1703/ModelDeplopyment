import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class BookingPredictor:
    def __init__(self, model_file, enc_info, feature_list):
        self.model = joblib.load(model_file)
        self.enc_info = enc_info
        self.feature_list = feature_list
        self.encoders = {}

    def transform_input(self, df_input):
        le = LabelEncoder()
        for col in self.enc_info['label']:
            df_input[col] = le.fit_transform(df_input[col])

        # One-hot Encoding
        ohe_frames = []
        for col in self.enc_info['one_hot']:
            if col not in self.encoders:
                enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc.fit(df_input[[col]])
                self.encoders[col] = enc

            enc = self.encoders[col]
            transformed = enc.transform(df_input[[col]])
            cols = enc.get_feature_names_out([col])
            cols = [col.replace("Room_Type", "Room_Type ") for col in cols]  
            ohe_frames.append(pd.DataFrame(transformed, columns=cols))

        df_input = df_input.reset_index(drop=True)
        df_input = pd.concat([df_input.drop(columns=self.enc_info['one_hot'])] + ohe_frames, axis=1)

        df_input = df_input.reindex(columns=self.feature_list, fill_value=0)
        return df_input



    def predict(self, df_input):
        prepped = self.transform_input(df_input)
        return self.model.predict(prepped)

def main():
    st.title('Prediksi Pembatalan Booking Hotel')

    # Form Input
    user_input = {
        'no_of_adults': st.number_input('Jumlah Dewasa', 0, 10),
        'no_of_children': st.number_input('Jumlah Anak', 0, 10),
        'no_of_weekend_nights': st.number_input('Weekend Nights', 0, 7),
        'no_of_week_nights': st.number_input('Week Nights', 0, 20),
        'type_of_meal_plan': st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
        'required_car_parking_space': st.selectbox('Parkir Dibutuhkan', [0, 1]),
        'room_type_reserved': st.selectbox('Tipe Kamar', [
            'Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 
            'Room Type 5', 'Room Type 6', 'Room Type 7'
        ]),
        'lead_time': st.number_input('Lead Time', 0, 1000),
        'arrival_year': st.number_input('Tahun Kedatangan', 2000, 2023, value=2017),
        'arrival_month': st.number_input('Bulan Kedatangan', 1, 12),
        'arrival_date': st.number_input('Tanggal Kedatangan', 1, 31),
        'market_segment_type': st.selectbox('Segment Pasar', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online']),
        'repeated_guest': st.selectbox('Tamu Berulang', [0, 1]),
        'no_of_previous_cancellations': st.number_input('Pembatalan Sebelumnya', 0, 20),
        'no_of_previous_bookings_not_canceled': st.number_input('Booking Sukses Sebelumnya', 0, 100),
        'avg_price_per_room': st.number_input('Harga Rata-rata Kamar', 0.0, 100000.0),
        'no_of_special_requests': st.number_input('Permintaan Khusus', 0, 5)
    }

    if st.button('Prediksi'):
        df_input = pd.DataFrame([user_input])

        model_path = 'rf_model.pkl'
        encoding_setup = {
            'label': ['arrival_year'],
            'one_hot': ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        }
        expected_features = [
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
            'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month',
            'arrival_date', 'repeated_guest', 'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests',
            'type_of_meal_plan_Meal Plan 1', 'type_of_meal_plan_Meal Plan 2', 
            'type_of_meal_plan_Meal Plan 3', 'type_of_meal_plan_Not Selected',
            'room_type_reserved_Room_Type 1', 'room_type_reserved_Room_Type 2',
            'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4',
            'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6',
            'room_type_reserved_Room_Type 7', 'market_segment_type_Aviation',
            'market_segment_type_Complementary', 'market_segment_type_Corporate',
            'market_segment_type_Offline', 'market_segment_type_Online'
        ]


        predictor = BookingPredictor(model_path, encoding_setup, expected_features)
        hasil_prediksi = predictor.predict(df_input)

        label = "Not Canceled" if hasil_prediksi[0] == 1 else "Canceled"
        print(df_input.iloc[0])
        st.success(f"Hasil Prediksi: {label}")

if __name__ == '__main__':
    main()