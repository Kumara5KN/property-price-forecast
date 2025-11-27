🏠 Bangalore Home Price Prediction (Streamlit App)

A modern, interactive Streamlit web application that predicts home prices in Bangalore based on location, square footage, number of bedrooms (BHK), and bathrooms.
The app uses a trained machine learning model, cleaned dataset, and custom-designed UI with dark mode styling.

1.Features
----------

Predict Bangalore home prices in Indian Lakhs

Fully interactive form:

Square Feet

BHK selection

Bathrooms

Location dropdown

Machine Learning Model integrated from artifacts
(columns.json & banglore_home_prices_model.pickle)

Beautiful dark-mode UI with custom CSS

Background with blurred overlay

Optimized with @st.cache_resource for faster loading

Error handling for missing model/data files
<br>
2.How It Works
--------------<br>

Loads saved model artifacts from the ./artifacts folder

Converts inputs into the correct numeric format

Matches location to one-hot encoded column

Generates prediction using the ML model

Displays the result with styled UI
<br>

▶️ Run Locally
--------------<br>
1️⃣ Install dependencies
pip install streamlit numpy pickle5

2️⃣ Run the app
streamlit run app.py

3️⃣ Visit the app in your browser
http://localhost:8501


