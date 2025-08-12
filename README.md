🌟 AtmoPredict: Weather Classification
A Streamlit web application that predicts precipitation type (Rain, Snow, Sunny) using a historical weather dataset. The app allows users to compare the performance of three popular machine learning models: Naive Bayes, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).

🎨 Features
Model Comparison: Train and evaluate Naive Bayes, KNN, and SVM on a weather dataset.

Interactive UI: A user-friendly interface built with Streamlit for easy model selection and execution.

Performance Metrics: Displays accuracy scores, classification reports, and confusion matrices to evaluate model performance.

Educational Content: Provides a clear explanation of each algorithm, including its formula, advantages, and disadvantages.

⚙️ How It Works
The application uses a historical weather dataset with the following features:

Apparent Temperature (°C)

Humidity

Wind Speed (km/h)

The goal is to predict the Precipitation Type (Rain, Snow, or Sunny). The data is preprocessed, scaled, and then used to train the selected machine learning model.

🖼️ Demo
Here's a quick look at the application in action:

🚀 Getting Started
Follow these steps to run the project locally.

1. Prerequisites
Make sure you have Python and pip installed.

2. Clone the Repository

git clone https://github.com/<your-username>/atmo-predict-app.git
cd atmo-predict-app

3. Install Dependencies
Install the required Python libraries using the requirements.txt file.
pip install -r requirements.txt

5. Run the App
Launch the Streamlit application from your terminal.
streamlit run app.py
The app will open automatically in your web browser.

📂 File Structure
app.py: The main Streamlit application file.

weatherHistory.csv: The dataset used for training the models.

requirements.txt: A list of all Python dependencies.

.gitignore: Specifies files and folders to be ignored by Git.

🤝 Contributing
Feel free to fork the repository, open an issue, or submit a pull request if you have any suggestions or improvements.
