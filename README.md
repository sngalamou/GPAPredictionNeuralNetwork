# Neural Network Grade Prediction Model (Demonstration Project)

## Overview

🚀 **Welcome to the Neural Network Grade Prediction Model demonstration project!** 🚀

This project showcases a Python-based system that combines neural network techniques with data processing to predict student acceptance decisions.  **It is important to note that this project uses *sample* data for demonstration purposes only.**  It is *not* intended for use with real student data without explicit permission from the relevant authorities and adherence to data privacy regulations.  The model evaluates student performance using simulated TEAS scores, GPA, and additional criteria, and is designed to process data from various file formats—including PDF, CSV, JSON, and TXT.

**Key Features (Demonstration):**
- **Hybrid Evaluation System:** Integrates a neural network model with traditional evaluation functions to simulate student readiness assessment.
- **Multi-Format Data Integration:** Demonstrates the capability to extract and process data from PDFs, CSVs, JSON files, and text files.
- **Neural Network Implementation:** Utilizes a simple neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron. The model performs both forward and backward passes, updating weights via gradient descent.
- **Automated Grade Prediction:**  Demonstrates the automated generation of *sample* test data, evaluation of simulated student records, and export of results for analysis.

---

## Features

- **Neural Network Architecture**  
  - **Structure:** 2 input neurons, 4 hidden neurons, and 1 output neuron.
  - **Forward Pass:** Processes inputs through layers using weights, biases, and a sigmoid activation function.
  - **Backward Pass:** Calculates gradients of the loss with respect to weights and biases and updates them via gradient descent.

- **Comprehensive Data Processing**  
  - **PDF Extraction:** Uses `PyPDF2` and `tabula-py` to extract text and table data.
  - **Multi-Format Support:** Includes dedicated processors for CSV, JSON, and TXT files.
  - **Dynamic Header Management:** Automatically updates and maintains data headers during processing.

- **Grade Prediction & Evaluation (Demonstration):**  
  - Simulates student acceptance evaluation based on *sample* TEAS scores, GPA, and predefined criteria.
  - Generates synthetic test data for demonstration and testing purposes.
  - Exports processed results to an Excel file (`Name&Teas.xlsx`) for easy review.

- **Modular & Extensible Design**  
  - Organized into clear, dedicated classes (`PDFProcessor`, `CSVProcessor`, `JSONProcessor`, `TXTProcessor`) for easy maintenance and future expansion.
  - Seamlessly combines multiple data sources into unified DataFrames.

---

## Tech Stack

- **Python 3.10+**
- **numpy** for numerical operations
- **pandas** for data manipulation
- **tabula-py** for extracting tables from PDFs
- **PyPDF2** for PDF text extraction
- **pdfminer.six** (as an alternative PDF text extraction tool)
- **json** and **os** for file handling
- **random** for test data generation

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/sngalamou/NeuralNetworkGradePrediction.git
   cd NeuralNetworkGradePrediction
   ```

2. **Create & Activate a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   *Ensure that `requirements.txt` includes all necessary packages: numpy, pandas, tabula-py, PyPDF2, pdfminer.six, etc.*

---

## Usage

1. **Prepare Your Data File**  
   Place your data file (PDF, CSV, JSON, or TXT) in the designated directory (e.g., `static/testData`).  **Note: This project is intended for use with *sample* or *synthetic* data only.  Do not use real student data without proper authorization.**

2. **Run the Notebook or Script**  
   - **Jupyter Notebook:**  
     Launch the notebook to interact with the code:
     ```bash
     jupyter notebook neuralNetwork.ipynb
     ```
   - **Python Script:**  
     Alternatively, run the script directly if available:
     ```bash
     python neuralNetwork.py
     ```

3. **Follow On-Screen Prompts**  
   The program will prompt you to enter the file name (including its extension). It will then process the file, extract the relevant data, evaluate each student’s record, and decide on their acceptance status.

4. **Review the Output**  
   Processed data is saved as an Excel file named `Name&Teas.xlsx` with a sheet titled "Name & Tease Scores". Open this file to review the processed student data and evaluation outcomes.

---

## Project Structure

```
.
├── neuralNetwork.ipynb         # Main Jupyter Notebook with the implementation and explanations
├── neuralNetwork.py            # Python script version (if applicable)
├── requirements.txt            # List of Python dependencies
├── static
│   └── testData                # Folder containing sample data files (PDF, CSV, JSON, TXT)
├── README.md                   # Project documentation
```

---

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**  
   ```bash
   git checkout -b feature/my-awesome-feature
   ```
3. **Commit Your Changes**  
   ```bash
   git commit -m "Add awesome feature"
   ```
4. **Push to the Branch**  
   ```bash
   git push origin feature/my-awesome-feature
   ```
5. **Open a Pull Request** on GitHub

---

## License

This project is not licensed for public use. Please do not use or adapt the code without permission.  **Furthermore, this code should not be used with real student data without explicit consent and adherence to all applicable data privacy laws and regulations.**

---

## Acknowledgments

- **Supervisor:**
  I would like to thank my supervisor, Paul Schroeder, for his support, guidance, and the autonomy he granted me, which were essential to the successful completion of this project.
- **Libraries & Tools:**  
  Special thanks to the developers behind `numpy`, `pandas`, `tabula-py`, `PyPDF2`, and `pdfminer.six` for their indispensable libraries.
- **Feedback & Inspiration:**  
  Gratitude to colleagues and mentors who provided valuable feedback during development.
- **You:**  
  Thank you for exploring this project and considering its use for your data processing and neural network applications!

---

*Follow me on [LinkedIn](#) for more updates and projects.*
