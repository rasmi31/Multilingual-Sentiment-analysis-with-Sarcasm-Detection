Multilingual Sentiment Analysis with Sarcasm Detection 🚀
Project Status: Active Development 🛠️
🌟 Project Overview
Traditional sentiment analysis models often struggle with sarcasm, as a sarcastically positive statement (e.g., "Oh, what a fantastic day! My car broke down and it's raining!") is frequently misclassified as positive. This project tackles this challenge head-on by building a joint model that performs two tasks simultaneously:

Sentiment Classification: Categorizing text as Positive, Neutral, or Negative.

Sarcasm Detection: Identifying if a given text is Sarcastic or Not Sarcastic.

The model leverages the powerful capabilities of the XLM-RoBERTa architecture, which is pre-trained on a massive multilingual corpus, making it highly effective for cross-lingual tasks.

✨ Key Features
--Multilingual Capability: The model is built on XLM-RoBERTa, enabling it to process and understand reviews in various languages.

--Multi-Task Learning: It combines sentiment analysis and sarcasm detection into a single, efficient model.

--Deep-Learning-Based: Utilizes a state-of-the-art transformer architecture for high performance.

--Comprehensive Evaluation: Includes detailed classification reports and confusion matrices for both tasks.

--Optimized for Accessibility: The code is optimized to run on systems with limited resources by freezing the encoder layers and using mixed precision training.

--Model and Methodology
The core of the model is the xlm-roberta-base transformer. A MultiTaskModel class is implemented with a shared encoder and two separate classification heads: one for sentiment and one for sarcasm.

--Shared Encoder: XLMRobertaModel captures the deep semantic and contextual representations of the input text.

--Sentiment Head: A linear layer with 3 output neurons (for Positive, Neutral, Negative).

--Sarcasm Head: A linear layer with 1 output neuron, followed by a sigmoid activation for binary classification (Sarcastic or Not Sarcastic).

The model is trained using a combined loss function that minimizes both the sentiment classification error (using CrossEntropyLoss) and the sarcasm detection error (using BCEWithLogitsLoss).

📁 Dataset
The model is trained and evaluated on the final_project_data.xlsx - Sheet1.csv dataset, which contains customer reviews with columns for REVIEW, LANG_CODE, RATING, SENTIMENT, and SARCASM. The RATING and SARCASM columns are used to generate the ground truth labels for the training process.

📈 Results and Evaluation
The model demonstrates strong performance in both tasks, as evidenced by the generated confusion matrices and classification reports.

Sentiment vs. Sarcasm Confusion Matrix
Sentiment Classification Confusion Matrix
Sarcasm Detection Confusion Matrix
🛠️ How to Run the Code
Clone this repository: git clone [your-repo-url]

Install the required dependencies (see below).

Place the final_project_data.xlsx - Sheet1.csv file in the same directory as the Jupyter notebook.

Open and run the final_code_project.ipynb notebook in a Jupyter environment like JupyterLab or Google Colab. The notebook will handle all data loading, model training, and evaluation automatically.

📦 Dependencies
You will need the following Python libraries:

pandas

torch

transformers

scikit-learn

seaborn

matplotlib

openpyxl (to read the .xlsx file)

You can install them using pip:
pip install pandas torch transformers scikit-learn seaborn matplotlib openpyxl

🔮 Future Work
Fine-tune the entire model: The current implementation freezes the transformer encoder to conserve memory. Fine-tuning the entire model could lead to further performance gains.

Implement a web interface: Create a simple web application using Flask or Streamlit to allow users to input text and get real-time sentiment and sarcasm predictions.

Expand the dataset: Collect and label a larger and more diverse dataset to improve model generalization.

⚖️ License
This project is licensed under the MIT License - see the LICENSE.md file for details.
