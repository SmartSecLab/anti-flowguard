# Anti-flowguard : Practically disproving the claim made in Flowguard paper

In this repository, we have provided the replication package to disprove the claim made by the [`FlowGuard: An Intelligent Edge Defense Mechanism Against IoT DDoS Attacks`](https://ieeexplore.ieee.org/document/9090824).
This project aims to train and evaluate several machine learning models using a given dataset. The primary focus is on classification tasks using different classifiers such as Decision Tree, Naive Bayes, and Random Forest. Additionally, the project includes feature importance analysis and visualization.


## Requirements

- Python 3.8+
- Required libraries are listed in the `requirements.txt` file.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/SmartSecLab/anti-flowguard.git
    cd anti-flowguard
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

The configuration file `config.yaml` should be placed in the root directory. It should include the following keys:
- `data`: Path to the dataset (CSV file).
- `split`: Dictionary containing the `test_size` for train-test split.

## Usage

Ensure that the configuration file config.yaml is correctly set up.
Run the script:

    python main.py


## Classifiers

The script includes the following classifiers:

    Decision Tree Classifier
    Naive Bayes Classifier
    Random Forest Classifier


## Outputs

    classification_report.txt: Contains the classification report for the evaluated models.
    figure/feature_importances.png: Visualization of feature importances.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or fixes.