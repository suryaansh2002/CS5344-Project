# Hate Detection in MMH150K Dataset

## About Project

### Background
The proliferation of hate speech on social media platforms poses significant challenges to
maintaining respectful online environments. While traditional hate speech detection methods rely solely on textual analysis, which makes them ineffective in identifying cyberbullying that emerges from the combination of text and images. In many cases, neither the text nor the image alone is explicitly hateful, but together they convey harmful intent. This is especially common on platforms like Twitter, where users often craft multimodal content to bypass moderation. Therefore, detecting cyberbullying in multimodal publications is crucial, as it enables more accurate and comprehensive identification of harmful content, protecting users from subtle and context-dependent abuse.

### Target
The project aims to develop a classifier capable of detecting hate
speech in social media posts by integrating both textual and visual data. By leveraging multimodal
data analytics, we seek to enhance the accuracy of hate speech detection and provide
comprehensive insights into the nature of such content

## Key Features
- Full data preprocessing pipeline for the MMHS150K dataset
- Class balancing and multimodal data augmentation techniques
- Two fusion-based architectures:
  - EfficientNet-B0 + BiLSTM
  - MobileNetV2 + RoBERTa
- Comparative evaluation showing significant improvements over text-only baselines

## Dataset
We used the MMHS150K dataset https://gombru.github.io/2019/10/09/MMHS/, which contains approximately 150,000 tweet-image pairs labeled across multiple hate categories:
- NotHate
- Racist
- Sexist
- Homophobe
- Religion
- OtherHate

For our project, we converted these into a binary classification task (Hate vs NotHate).


## Pipeline
### Data Preprocessiong
In the data preprocessing phase, we conducted three major steps to transform raw data into model-ready format:
- **Label Transformation**: Implemented majority voting system to handle inconsistent annotations and converted complex multiclass labels into binary format for simplified classification:
  - 0 = NotHate
  - 1 = Hate (including Racist, Sexist, Homophobe, Religion, OtherHate)
- **Text Cleaning**: Lowercased all text and removed URLs, mentions,hashtags, numbers, punctuation
- **Text Tokenization**: Employed Hugging Face's BERT tokenizer ('bert-base-uncased') for advanced language understanding
Applied standardization techniques with Padding and Truncation to ensured consistent input lengths for model processing

### Data Augmentation
To address class imbalance and enhance model robustness, we implemented comprehensive augmentation strategies:

- **Text Augmentation for Hate Class**: Applied two random techniques per record to create diverse text variations:
  - Synonym Replacement using WordNet
  - Random Word Deletion (word-level dropout)
  - Random Word Swap
- **Image Augmentation for Hate Class**: Utilized torchvision to apply one random technique per image:
    - Rotation transformations
    - Brightness adjustment
    - Horizontal/vertical flips
- **Dataset Balancing**: Achieved perfect balance through targeted sampling:
    - NotHate: Undersampled to 75,000 samples
    - Hate: Oversampled to 75,000 samples using combined text and image augmentation
### Multimodal Fusion Models
We implemented two distinct fusion models, each with unique architectural choices and fusion strategies:
- **EfficientNet**(for Image Branch) + **BiLSTM**(for Text Branch)
- **MobileNetV2**(for Image Branch) + **RoBERTa**(for Text Branch)
  
## Models
### EfficientNet-B0 + BiLSTM
- Image feature extraction with EfficientNet-B0
- Text processing through BiLSTM
- Learned fusion mechanism via concatenation and fully connected layers
- Best performance: 72.06% accuracy, 72.17% F1-score

### MobileNetV2 + RoBERTa
- Lightweight image processing with MobileNetV2
- Text encoding with RoBERTa
- Alpha-weighted fusion strategy
- Performance: 70.50% accuracy, 68.62% F1-score

## Results
Our best model (EfficientNet + BiLSTM) achieved a 6.27% F1-score improvement over text-only baselines, demonstrating the value of multimodal approaches for hate speech detection.



## Project Setup Instructions

This document outlines the steps to set up your local environment for this project.

## Prerequisites

* **Git:** You need Git installed on your system to clone the repository. You can check if you have it by running `git --version` in your terminal. If not, you can download it from [https://git-scm.com/downloads](https://git-scm.com/downloads).
* **Python 3:** Python 3 must be installed on your system. It's highly recommended to use a recent version of Python 3. You can check your Python version by running `python --version` or `python3 --version` in your terminal. If not installed, you can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).
* **pip:** pip is the package installer for Python. It usually comes bundled with Python installations. You can check if you have it by running `pip --version` or `pip3 --version` in your terminal.

## Steps

1.  **Clone the Repository:**

    Open your terminal or command prompt and navigate to the directory where you want to clone the project. Then, use the `git clone` command followed by the repository URL:

    ```bash
    git clone https://github.com/suryaansh2002/CS5344-Project.git
    ```

2.  **Navigate to the Project Directory:**

    Once the repository is cloned, you need to change your current directory to the newly created project directory. Use the `cd` (change directory) command followed by the repository name:

    ```bash
    cd CS5344-Project
    ```

3.  **Create a Virtual Environment:**

    It's best practice to create a virtual environment to isolate the project's dependencies from your global Python installation. This helps avoid conflicts with other projects. Use the following command to create a virtual environment named `venv`:

    ```bash
    python -m venv venv
    ```

    or, if you are using Python 3 specifically:

    ```bash
    python3 -m venv venv
    ```

    This command will create a directory named `venv` within your project directory containing a copy of the Python interpreter and necessary supporting files.

4.  **Activate the Virtual Environment:**

    You need to activate the virtual environment to use the Python interpreter and installed packages within it. The activation command depends on your operating system:

    * **Linux/macOS:**

        ```bash
        source venv/bin/activate
        ```

        Your terminal prompt should now be prefixed with `(venv)`, indicating that the virtual environment is active.

    * **Windows (Command Prompt):**

        ```bash
        venv\Scripts\activate
        ```

        Your command prompt should now be prefixed with `(venv)`.

    * **Windows (PowerShell):**

        ```powershell
        .\venv\Scripts\Activate.ps1
        ```

        Your PowerShell prompt should now be prefixed with `(venv)`.

5.  **Install Dependencies from `requirements.txt`:**

    The `requirements.txt` file usually lists all the necessary Python packages and their versions that the project depends on. Once the virtual environment is activated, you can install these dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    This command will read the `requirements.txt` file and install all the listed packages into your active virtual environment.

## Next Steps

After completing these steps, your environment should be set up correctly, and you can proceed with running the project's scripts. Remember to activate the virtual environment every time you start working on this project in a new terminal session.

1.    Download the MMH150K Dataset, save into 'DATA' folder from [https://gombru.github.io/2019/10/09/MMHS/](https://gombru.github.io/2019/10/09/MMHS/). A manually annotated multimodal hate speech dataset formed by 150,000 tweets, each one of them containing text and an image.

2.    Run the EfficientNet-biLSTM model.
```bash
python scripts/run_efficientnet_bilstm.py
```
3.    Run the MobileNet-roBERTa model.
```bash
python scripts/run_mobilenet_roberta.py
```


