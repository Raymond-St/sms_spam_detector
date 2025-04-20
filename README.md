# DU AI Bootcamp Course Project #21 "SMS Spam Detector"
(AKA "sms_spam_detector")

# SMS Spam Classification
A machine learning application to classify SMS messages as spam or ham (non-spam) with an interactive Gradio interface.

## Project Overview
This project implements a text classification system that can detect whether an SMS message is spam or legitimate (ham). The system uses natural language processing techniques and machine learning algorithms to analyze the content of text messages and make predictions based on patterns found in a training dataset.
Test it here (for one Week, until 4-26-2025):  https://9eca2a4ec93f59566a.gradio.live

## Features
- **Multiple Classification Algorithms:** Compares Naive Bayes, Random Forest, and Support Vector Machines to automatically select the best performing model
- **Text Preprocessing:** Includes lowercase conversion, special character removal, and number filtering
- **TF-IDF Vectorization**: Transforms text data into numerical features
- **Interactive Web Interface**: Built with Gradio for easy testing
- **Pre-trained Model**: Includes examples for immediate testing
- **Simple API**: Can be integrated with other applications

## Technologies Used
- Python 3.12
- scikit-learn
- pandas
- numpy
- Gradio
- matplotlib
- seaborn

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/Raymond-St/sms_spam_detector.git
   ```
2. Install the required packages:
   ```
   pip install pandas, numpy, scikit-learn, gradio, matplotlib, & seaborn
   ```
3. Download the SMS Spam Collection dataset and place it in the `Resources` folder.

## Usage
1. Run the Jupyter notebook:
   ```
   jupyter notebook sms-classification-project.ipynb
   ```
2. Execute all cells to train the model and launch the Gradio interface.
3. Once the interface is running, you can:
   - Enter your own text messages to classify
   - Click on example messages to see how they are classified
   - View confidence scores for each prediction

## Project Structure
```
project/
├── Resources/
│   └── SMSSpamCollection.csv         # Dataset file
├── 21-gradio_sms_text_classification/
│   ├── data/                         # Processed data
│   └── models/                       # Saved model files
└── sms-classification-project.ipynb  # Main notebook file
```

## How It Works
1. **Data Preparation**: The SMS dataset is loaded and processed, including text normalization, special character removal, and conversion to lowercase..
2. **Feature Extraction**: Text messages are converted to numerical features using TF-IDF vectorization.
3. **Model Comparison & Selection**: Multiple classification algorithms (Naive Bayes, Random Forest, and Support Vector Machines) are trained and evaluated, with the best performing model automatically selected.
4. **Prediction**: New messages are classified based on the trained model.
5. **Interface**: Results are displayed through a user-friendly Gradio interface.

## Examples
The application includes several example messages for testing such as:
- "You are a lucky winner of $5000!" (Likely spam)
- "You won 2 free tickets to the Super Bowl." (May be classified as spam)
- "You won 2 free tickets to the Super Bowl. Text us to claim your prize." (Likely spam)
- "Hey, what time should we meet for dinner tonight?" (Likely ham)
- "Don't forget to pick up milk on your way home." (Likely ham)

## Model Performance
Multiple models are evaluated with the following configuration:
Test size: 33% of the dataset
Feature extraction: TF-IDF with up to 5000 features and unigrams/bigrams
Evaluated algorithms:
Multinomial Naive Bayes
Random Forest (100 estimators)
Linear Support Vector Machine
**NOTE:
**The best performing model is automatically selected** based on classification accuracy and used for the final application.

## Future Improvements
- Add hyperparameter tuning to further optimize model performance
- Implement ensemble methods combining multiple classifiers
- Add feature importance visualization
- Create a deployable web application
- Add user feedback collection to improve the model
- Implement cross-validation for better performance measurement

## License
MIT license - copyright R_Stover for Educational purposes only

## Contributors
R_Stover - sole contributor for "Project 21 - SMS Spam Detector"

## Acknowledgments
- The SMS Spam Collection dataset used in this project provided by the DU AI Bootcamp Instructional Team
- Scikit-learn and Gradio libraries for machine learning and UI implementation
- The DU AI Bootcamp Instructional Team
