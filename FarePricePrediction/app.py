import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle
from flask import Flask, render_template, request


app = Flask(__name__)

df=pd.read_csv('dataset/CabFirmCaseStudyMerged.csv')
unique_cities = df['City'].unique().tolist()
unique_companies = df['Company'].unique().tolist()
unique_genders = df['Gender'].unique().tolist()

le_city = LabelEncoder()
le_company = LabelEncoder()
le_gender = LabelEncoder()

df['City_encoded'] = le_city.fit_transform(df['City'])
df['Company_encoded'] = le_company.fit_transform(df['Company'])
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])


x = df[['KM_Travelled', 'City_encoded', 'Company_encoded', 'Gender_encoded']]
y = df['Price_Charged']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed
rf_regressor.fit(x_train, y_train)



# Save the trained model as a pickle file
pickle.dump(rf_regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
#joblib.dump(rf_regressor, '/content/drive/My Drive/OUTPUT_CSV/random_forest_regressor_model.pkl')


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html', unique_cities=unique_cities, unique_companies=unique_companies, unique_genders=unique_genders,
                           error="Please enter a value for Km to Travel.")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    city = request.form['city']
    km_to_travel = float(request.form['km_to_travel'])
    gender = request.form['gender']
    company = request.form['company']
    
    # Encoding categorical variables
    city_encoded = le_city.transform([city])[0]
    gender_encoded = le_gender.transform([gender])[0]
    company_encoded = le_company.transform([company])[0]
    
    # Make prediction
    prediction = model.predict([[km_to_travel, city_encoded, company_encoded, gender_encoded]])
    
    # Render the prediction result template with the predicted price
    return render_template('index.html', prediction=prediction[0],unique_cities=unique_cities, unique_companies=unique_companies, unique_genders=unique_genders,
                           city=city, km_to_travel=km_to_travel, gender=gender, company=company)
prediction = None

if __name__ == '__main__':
    app.run(debug=True)
