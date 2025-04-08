# Hate Detection in MMH150K Dataset


# Project Setup Instructions

This document outlines the steps to set up your local environment for this project.

## Prerequisites

* **Git:** You need Git installed on your system to clone the repository. You can check if you have it by running `git --version` in your terminal. If not, you can download it from [https://git-scm.com/downloads](https://git-scm.com/downloads).
* **Python 3:** Python 3 must be installed on your system. It's highly recommended to use a recent version of Python 3. You can check your Python version by running `python --version` or `python3 --version` in your terminal. If not installed, you can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).
* **pip:** pip is the package installer for Python. It usually comes bundled with Python installations. You can check if you have it by running `pip --version` or `pip3 --version` in your terminal.

## Steps

1.  **Clone the Repository:**

    Open your terminal or command prompt and navigate to the directory where you want to clone the project. Then, use the `git clone` command followed by the repository URL:

    ```bash
    git clone [https://github.com/suryaansh2002/CS5344-Project.git](https://github.com/suryaansh2002/CS5344-Project.git)
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

After completing these steps, your environment should be set up correctly, and you can proceed with running the project's scripts.
- Download the MMH150K Dataset, save into 'DATA' folder.
```bash
python scripts/run_efficientnet_bilstm.py
```
```bash
python scripts/run_mobilenet_roberta.py
```


Remember to activate the virtual environment every time you start working on this project in a new terminal session.
