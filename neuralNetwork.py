import numpy as np
import random
import pandas as pd
import tabula
import PyPDF2
import json
import os

# --------------------------
# Utility Functions
# --------------------------

def convert_percentage_to_float(column):
    """
    Converts percentage strings (e.g., "80%") to float values (0.8).
    If the value is already numeric, it is returned unchanged.
    """
    return column.apply(lambda x: float(x.rstrip('%')) / 100 if isinstance(x, str) and x.endswith('%') else x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --------------------------
# Neural Network Class
# --------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        return self.output

    def backward(self, X, y, learning_rate):
        output_error = y - self.output
        output_gradient = output_error * sigmoid_derivative(self.output)
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * sigmoid_derivative(self.hidden_layer_output)

        # Clip gradients to avoid exploding gradients
        output_gradient = np.clip(output_gradient, -1, 1)
        hidden_gradient = np.clip(hidden_gradient, -1, 1)

        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_gradient) * learning_rate
        self.bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate
        self.bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = mse_loss(y, self.output)
                print(f"Epoch {epoch}, Loss: {loss}")

# --------------------------
# Evaluation Functions & Constants
# --------------------------

# Sample constants for evaluation criteria
act_english = 19
act_math = 22
acte = "ENG 101"
actm = "Math 128"

sat_english = 480
sat_math = 530
sate = "English 101"
satm = "Math 128"

teasWeight = 0.4
min_gpa = 2.0

lincenseType = ""
immunization = True
residency = True
gen_eds = True
completedCourses = ["ENG 101", "PSYC 101", "HEAL 109", "BIO 250", "BIO 251", "BIO 240", "COMM 101", "NURS 100"]
Ati_Teas = True
Act_Sat = True
Cna = True
waitlist = False

def evalStudent(gpa, teasScore, cna):
    gTeas = False
    gGPA = False
    gCNA = False
    aTeas = False
    if teasScore >= 82:
        gTeas = True
    elif teasScore >= 60:
        aTeas = True

    if gpa >= min_gpa:
        gGPA = True

    if cna:
        gCNA = True

    if gTeas and gGPA and gCNA:
        print("Accepted - Guaranteed")
        return "Accepted - Guaranteed"
    elif aTeas and gGPA and gCNA:
        print("Accepted - Regular")
        return "Accepted - Regular"
    else:
        print("Rejected")
        return "Rejected"

def eval_Student(name, teas, gpa):
    name = str(name)
    teas = int(teas)
    gpa = float(gpa)
    gTeas = False
    gGPA = False
    gCNA = False
    aTeas = False
    cna = True

    if teas >= 82:
        gTeas = True
    elif teas >= 60:
        aTeas = True

    if gpa >= min_gpa:
        gGPA = True

    if cna:
        gCNA = True

    if gTeas and gGPA and gCNA:
        print("Accepted - Guaranteed")
        return "Accepted - Guaranteed"
    elif aTeas and gGPA and gCNA:
        print("Accepted - Regular")
        return "Accepted - Regular"
    else:
        print("Rejected")
        return "Rejected"

# --------------------------
# Data Processing Classes
# --------------------------

# Global header list for dynamic header management
headers = []

def update_headers(new_headers):
    global headers
    print("Updating headers from:", new_headers)
    for header in new_headers:
        if header not in headers:
            headers.append(header)
    print("Current headers:", headers)

def randomizerGPA():
    return np.round(np.random.uniform(0.0, 4.0), 2)

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text_pypdf2(self):
        text = ""
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def extract_text_pdfminer(self):
        from pdfminer.high_level import extract_text
        return extract_text(self.file_path)

    def extract_tables_tabula(self):
        tables = tabula.read_pdf(self.file_path, pages='all', multiple_tables=True)
        if tables:
            for table in tables:
                if isinstance(table, pd.DataFrame):
                    update_headers(table.columns.tolist())
                else:
                    print("Error: Extracted table is not in DataFrame format.")
            return tables
        return None

    def extract_data(self):
        text = self.extract_text_pypdf2()
        tables = self.extract_tables_tabula()
        return text, tables

    def read_table(self, df):
        # Assumes the first column is 'Name' and the third column is 'Teas'
        return pd.DataFrame({'Name': df.iloc[:, 0], 'Teas': df.iloc[:, 2]})

    def read_tableFull(self, df):
        # Assumes specific column order:
        # Column 1: Name, 3: Teas, 4: Reading, 5: Math, 6: Science, 7: English
        name = df.iloc[:, 0]
        teas = df.iloc[:, 2]
        reading = df.iloc[:, 3]
        math = df.iloc[:, 4]
        science = df.iloc[:, 5]
        english = df.iloc[:, 6]
        ndf = pd.DataFrame({
            'Name': name, 
            'Teas': teas, 
            'Reading': reading, 
            'Math': math, 
            'Science': science, 
            'English': english, 
            'GPA': [randomizerGPA() for _ in range(len(name))]
        })
        return ndf

class CSVProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_csv(self):
        data = pd.read_csv(self.file_path)
        update_headers(data.columns.tolist())
        return data

class JSONProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_json(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        if data:
            if isinstance(data, list):
                update_headers(list(data[0].keys()))
            else:
                update_headers(list(data.keys()))
        return data

class TXTProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_txt(self):
        with open(self.file_path, 'r') as file:
            data = file.read()
        update_headers(['Text Content'])
        return data

def combine_df(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

def combine_df1(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

# --------------------------
# Main Data Processing Function
# --------------------------

def main():
    # Initialize empty DataFrames for basic and full data extraction
    df_basic = pd.DataFrame(columns=['Name', 'Teas', 'GPA'])
    df_full = pd.DataFrame()
    
    # Prompt user for the file name (including extension)
    file_name = input("Enter the file name (including extension): ")
    base_path = os.path.join("C:\\Users\\Lucien\\Desktop\\data science\\NursingLabDatabaseProject\\static\\testData")
    file_path = os.path.join(base_path, file_name)

    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return pd.DataFrame()

    if file_name.endswith('.pdf'):
        processor = PDFProcessor(file_path)
        text, tables = processor.extract_data()
        for table in tables:
            new_df = processor.read_table(table)
            new_df_full = processor.read_tableFull(table)
            df_basic = combine_df(df_basic, new_df)
            df_full = combine_df1(df_full, new_df_full)
    elif file_name.endswith('.csv'):
        processor = CSVProcessor(file_path)
        data = processor.read_csv()
        print(data)
        return data
    elif file_name.endswith('.json'):
        processor = JSONProcessor(file_path)
        data = processor.read_json()
        print(data)
        return pd.DataFrame(data)
    elif file_name.endswith('.txt'):
        processor = TXTProcessor(file_path)
        data = processor.read_txt()
        print(data)
        return pd.DataFrame()
    else:
        print("Unsupported file format.")
        return pd.DataFrame()

    return df_full

# --------------------------
# Execution: Data Processing and Neural Network Training
# --------------------------

if __name__ == "__main__":
    # Process the file and extract the full table data
    df = main()
    
    if not df.empty:
        # Convert percentage columns to numeric values (if they are strings with '%')
        for col in ['Teas', 'Reading', 'Math', 'Science', 'English']:
            if col in df.columns:
                df[col] = convert_percentage_to_float(df[col])
        
        # Normalize GPA values (assumes original GPA is on a 4.0 scale)
        df['GPA'] = df['GPA'] / 4.0
        
        # Prepare training data for the neural network
        X = df[['Teas', 'Reading', 'Math', 'Science', 'English']].values
        y = df['GPA'].values.reshape(-1, 1)
        
        # Initialize and train the neural network
        nn = NeuralNetwork(input_size=5, hidden_size=4, output_size=1)
        print("Training Neural Network...")
        nn.train(X, y, epochs=10000, learning_rate=0.01)
        
        # Generate predictions (rescaled back to the 4.0 GPA scale)
        predicted_gpa = nn.forward(X) * 4.0
        print("Predicted GPA:")
        print(predicted_gpa)
        df['Predicted_GPA'] = predicted_gpa
        
        # Export results to Excel
        df.to_excel("Name&Teas.xlsx", sheet_name='Name & Tease Scores')
    else:
        print("No data to process.")
