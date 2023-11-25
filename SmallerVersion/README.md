# Open EMIT

Application that uses machine learning algorithms to predict and forecast upcoming sand and dust storms  and give adaptive recommendations per mineral substances in SDS. 

## Technologies and Techniques used:
* [Steamlit](https://streamlit.io/): (v 1.22.0) Streamlit turns data scripts into shareable web apps in minutes.
All in pure Python. No front‑end experience required.
* [TensorFlow](https://www.tensorflow.org/?hl=end): TensorFlow is an open-source machine learning framework developed by the Google Brain team. It is designed to facilitate the creation of machine learning models, particularly deep learning models, and enable tasks such as neural network training, natural language processing, image recognition, and more.
* [Pandas](https://pandas.pydata.org/): (v 1.5.3) Pandas is an open-source data manipulation and analysis library for the Python programming language. It provides data structures and functions for working with structured data, making it a fundamental tool for data scientists, analysts, and engineers who work with data in Python.
* Other dependencies are found at requirements.txt
* Classification Machine Learning technique + K-Nearest Neighbors (KNN) algorithm to develop a prediction model.

g
## Installation
* If you wish to run your own build, first ensure you have python globally installed in your computer. If not, you can get python [here](https://www.python.org").
* After doing this, confirm that you have installed virtualenv globally as well. If not, run this:
    ```bash
        $ pip install virtualenv
    ```
* Then, Git clone this repo to your PC
    ```bash
        $ git clone https://github.com/oneku16/nasa-hackathon.git
    ```

* #### Dependencies
    1. cd into your the cloned repo as such:
        ```bash
            $ cd nasa-hackathon
        ```
    2. Create and fire up your virtual environment:
        ```bash
            $ virtualenv venv -p python3
            $ source venv/bin/activate
        ```
    3. Install the dependencies needed to run the app:
        ```bash
            $ pip install -r requirements.txt
        ```

* #### Run It
    Fire up the server using this one simple command:
    ```bash
        $ streamlit run myApp.py
    ```
    You can now access the file api service on your browser by using
    ```
        http://192.168.2.188:8501/
    ```
