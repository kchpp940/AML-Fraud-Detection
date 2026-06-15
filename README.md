# Anti Money Laundering (AML) Fraud Detection
**Executive Summary**

The "Anti-Money Laundering Fraud Detection" project aimed to detect fraudulent transactions using classification machine learning algorithms, including Random Forest, AdaBoost, and XGBoost. These machine learning models were trained using preprocessed data and evaluated with cross-validation technique to optimize the bias-variance trade-off. The models' hyperparameters were tuned using GridSearch and RandomSearch. Predictions were made using these models on both training and unseen (test) data, with the best-performing model selected based on recall (True Positive Rate) metrics.

The best model was deployed on AWS using Docker, with ECR (Elastic Container Registry) utilized for containerization and an EC2 instance used for cloud deployment. The trained model was integrated into a CI/CD pipeline via GitHub Actions for continuous updates and real-time monitoring. MLflow was utilized for experiment tracking and performance comparison. Additionally, an interactive web interface was developed using Streamlit, enabling user-friendly interaction with the model and its visualizations.


## Project Overview
This project focuses on developing a machine learning system to identify potentially fraudulent transactions, addressing the critical challenge of money laundering, a multi-billion dollar problem. This project is essential for financial institutions aiming to improve their Anti-Money Laundering (AML) systems. Traditional automated algorithms have a high false positive rate: legitimate transactions incorrectly flagged as laundering. The converse is also a major problem -- false negatives, i.e. undetected laundering transactions. The project seeks to reduce both of these issues, enhancing the accuracy and effectiveness of AML systems.

### Solution Approach
Machine Learning: ML Classification Algorithms

## Project Lifecycle
		1. Understanding the Problem Statement
		2. Data Collection
		3. Data Checks to perform
		4. Exploratory Data Analysis (EDA) 
		5. Data Pre-Processing and Feature Engineering
		6. Models cross-validation, tunning and training 
		7. Choose Best Model based on desired metrics
		8. Model deployment with CI/CD pipeline
		9. Web interactive interface
		10. Outcomes

### 1. Understanding the Problem Statement
The problem involves detecting money laundering activities, which are challenging because most automated algorithms have a high false positive rate: legitimate transactions incorrectly flagged as laundering. The converse is also a major problem -- false negatives, i.e. undetected laundering transactions. 

### 2. Data Collection
Data was sourced from Kaggle, providing a comprehensive dataset of transactions labeled as fraudulent (1) or legitimate (0).

Data Source: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data

### 3. Data Checks to Perform
Initial data validation included checking for missing values, inconsistencies, and anomalies to ensure data integrity.

### 4. Exploratory Data Analysis (EDA)
In-depth EDA was conducted to understand the distribution of features, detect patterns, and identify correlations between variables. Visualization techniques were employed to uncover hidden insights.

### 5. Data Pre-Processing and Feature Engineering
Data was cleaned and transformed to prepare it for model training. This included:
- **Handling Missing Values:** Addressed any gaps in the data to ensure completeness.
- **Encoding Categorical Variables:** Converted categorical variables into numerical formats, as machine learning models require numerical input.
- **Normalizing Numerical Features:** Scaled numeric variables to ensure uniformity and improve model performance.

Feature engineering was employed to create new variables that could enhance model performance. This process involved:
- **Scaling Numeric Variables:** Ensured that numeric features were on a similar scale to prevent any one feature from dominating the model.
- **Encoding Categorical Variables:** Transformed categorical data into numerical representations, enabling the model to process these features effectively.

#### Multicollinearity Check
- **Numerical Features:** Variance Inflation Factor (VIF) was calculated to identify multicollinearity among numerical features. Features with high VIF values were considered for removal or transformation.
- **Categorical Features:** Chi-squared tests were conducted to evaluate the relationship between categorical features and the target variable. The null hypothesis (no association between features) and the alternative hypothesis (association exists) were tested to assess feature importance and avoid redundancy.


### 6. Models Cross-Validation, Tunning and Training
Multiple machine learning models were built using classification algorithms: Random Forest, AdaBoost, and XGBoost. These models were cross-validated (`StratifiedKFold`) amd  hyperparameter tuned using `GridSearchCV`, and then trained using training dataset to find the best hyperparemeters for each model.  Now, each model was then configured with the best hyperparameters and finally trained using the training dataset. The trained models were then used to make predictions on both the training and test data.

### 7. Model Selection - choosing best model
The previous trained models with best hyperparamters were evaluated based on accuracy, precision, recall, and F1-score metrics. For this project, `Recall` (True positive Rate), was chosen as the final evaluation metric since minimizing `false negatives` is crucial for detecting fraudulent transactions. The best model was selected based on its recall performance.

### 8. Deployment
The final model was containerized using Docker and deployed on AWS. Continuous Integration/Continuous Deployment (CI/CD) pipelines were set up using GitHub Actions and AWS services. The deployment process involved the following steps:

**Notes**: 
Finally run the following command to test in local after buling prediction pipeline for web interface using FastAPI or streamlit
```bash
python app.py
or
streamlit run app_streamlit.py
```
```bash
open up you local host and port
```


		
**Docker** Notes

Build an Image from a Dockerfile
```bash
docker build -t aml-streamlit-app .
```
List local images
```bash
docker images
```
Delete an image
```bash
docker rmi <image_name>
```
Remove all unused images
```bash
docker image prune
```
Run a container with and publish a container’s port(s) to the host.
```bash
docker run -p 8501:8501 aml-streamlit-app
or
docker run -p 5000:5000 app_name
```
```bash
open up your local host and port
```


1. **CI/CD Setup with GitHub Actions:**
   - Configured GitHub Actions workflows in the `.github/workflows` directory, the `main.yaml` file, to automate the build, test, and deployment stages.

2. **AWS Console Setup:**
   - **Login to AWS Console:** Accessed the AWS Management Console to set up required resources.
   - **IAM User Creation:** Created an IAM user with permissions for deployment:
     - `AmazonEC2ContainerRegistryFullAccess`: Full access to Amazon Elastic Container Registry (ECR).
     - `AmazonEC2FullAccess`: Full access to Amazon EC2.

3. **Elastic Container Registry (ECR) Setup:**
   - Created an ECR repository for storing Docker images.
   - **ECR Repo URI:** `767397970670.dkr.ecr.us-east-1.amazonaws.com/aml_fraud_detector-container`

4. **EC2 Instance Setup:**
   - Create EC2 (Ubuntu): Virtual machine in the AWS cloud
   - Launched an EC2 instance (Ubuntu) to run the application.
   - **Connect to EC2 Instance:** Built Docker image, pushed it to ECR, and launched it on EC2.
		- **Description**: About the deployment
		1. Build docker image of the source code
		2. Push  docker image to ECR
		3. Launch EC2 
		4. Pull  image from ECR in EC2
		5. Launch docker image in EC2

		- Docker setup in EC2
		```bash
		# optional
		sudo apt-get update -y
		sudo apt-get upgrade

		# Required
		curl -fsSL https://get.docker.com -o get-docker.sh
		sudo sh get-docker.sh
		sudo usermod -aG docker ubuntu
		newgrp docker
		```

5. Configure EC2 as self-hosted runner in GitHub
	- Now, Go to GitHub 
```bash
	setting > actions > runner > new self hosted runner > choose os (Linux) > then run command one by one
```
	
6. Setup github secrets:
```bash
	setting > Secrets and variables > actions > New repository secret (in main screen)	

		AWS_ACCESS_KEY_ID=
		AWS_SECRET_ACCESS_KEY=
		AWS_REGION = us-east-1
		AWS_ECR_LOGIN_URI = 767397970670.dkr.ecr.us-east-1.amazonaws.com
		ECR_REPOSITORY_NAME = aml_fraud_detector-container			
```



## Model Version Management

Each training run automatically generates a `model_metadata.json` file and an `artifact_manifest.json` in the artifacts directory, providing full traceability and integrity validation from training to deployment.

### Generated Files

#### `artifacts/model_metadata.json`

| Field | Description |
|---|---|
| `model_version` | Auto-incremented integer, starting from 1 |
| `training_time` | ISO 8601 timestamp of when training completed |
| `data_file` | Absolute path of the source data file |
| `data_file_digest` | SHA-256 hash of the source data file |
| `feature_schema_version` | Version string from `feature_metadata.json` (`contract_version`) |
| `best_model_name` | Name of the best-performing model |
| `best_model_params` | Hyperparameters of the best model |
| `selection_metric` | Metric used for model selection (e.g., Recall) |
| `best_metric_value` | Value of the selection metric for the best model |
| `all_model_metrics` | Precision / Recall / F1 for every candidate model |
| `artifact_path` | Absolute path to the saved model artifact (`model.pkl`) |

#### `artifacts/artifact_manifest.json`

This manifest is the source of truth for artifact integrity. It records the path, size, mtime and SHA-256 digest of four core artifacts:

| Key | File |
|---|---|
| `model_pkl` | `artifacts/model.pkl` |
| `preprocessor_pkl` | `artifacts/preprocessor.pkl` |
| `feature_metadata_json` | `artifacts/feature_metadata.json` |
| `model_metadata_json` | `artifacts/model_metadata.json` (self-included) |

### How It Works

1. **Training**: After the training pipeline finishes, `save_model_metadata()` writes `model_metadata.json`, then `save_artifact_manifest()` captures digests for all four core artifacts into `artifact_manifest.json`. Both paths are recorded in the `TrainingSummary` and the resolved config output.
2. **Prediction Startup**: `PredictionPipeline.__init__()` calls `validate_artifacts()`, which runs the following checks:
   - Each of the four artifacts exists on disk and its SHA-256 digest matches the manifest
   - `model_metadata.artifact_path` resolves to the same file as `artifacts/model.pkl`
   - `model_metadata.data_file_digest` matches a fresh digest of the training data (data file still accessible)
   - `model_metadata.feature_schema_version` matches `feature_metadata.json.contract_version`
   - On any **error**, `validation_ok = False`, `model_metadata` is cleared (the old metadata is never silently shown) and all errors/warnings are exposed on the pipeline
3. **Web UI**: Flask and Streamlit only display model version info when `validation_ok` is true. Otherwise a prominent red banner lists every validation failure, instructing the operator to re-run training. Flask additionally exposes `GET /model-info` returning `{validation_ok, validation_errors, validation_warnings, metadata}`.

### Deployment Checklist

Before going live, make sure the `artifacts/` directory ships with **all** of:

```
artifacts/
├── artifact_manifest.json   ← integrity source of truth
├── model_metadata.json      ← version & metrics
├── model.pkl                ← trained model
├── preprocessor.pkl         ← feature preprocessor
└── feature_metadata.json    ← feature schema
```

If any of these files is missing, replaced by an older version, or modified on disk, `PredictionPipeline` will reject the metadata and surface the mismatch on the web UI.

### Troubleshooting

- **"MODEL ARTIFACT VALIDATION FAILED" on web UI**: Inspect the error list shown. The most common causes are:
  - `artifact_manifest.json missing or unreadable` — the manifest was not shipped with the model. Re-run training or copy the file from the training artifacts.
  - `<file> digest mismatch` — the file on disk differs from what was recorded at training time (wrong version, partial copy, corruption). Re-deploy the exact artifact set produced by the latest training run.
  - `model_metadata artifact_path mismatch` — `model_metadata.json` points to a different `model.pkl` than the one on disk.
  - `Feature schema version mismatch` — `feature_metadata.json.contract_version` no longer matches the value recorded in `model_metadata.json`.
- **Warnings instead of errors**: These are non-blocking (e.g., training data no longer accessible so digest can't be re-verified, or size-only drift) but should still be investigated.
- **Version not incrementing**: Delete `artifacts/model_metadata.json` and retrain. The version starts from 1 if the file does not exist.
- **Data digest mismatch**: Compare the `data_file_digest` field with a fresh hash of the data file to verify the deployed model was trained on the expected dataset:
  ```bash
  python -c "
  import hashlib, sys
  h = hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
  print(f'sha256:{h}')
  " <path_to_csv>
  ```
- **Programmatic validation check**:
  ```bash
  python -c "
  from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline
  p = PredictionPipeline()
  print('OK' if p.validation_ok else 'FAILED')
  print('errors:', p.validation_errors)
  print('warnings:', p.validation_warnings)
  "
  ```

## Web Interfaces
Two web interfaces were developed to interact with the model:
- **FastAPI:** A high-performance web framework for building APIs with Python, used to create an API endpoint for real-time fraud detection.
- **Streamlit:** An open-source app framework for machine learning and data science projects, used to build an interactive web application for visualizing results and interacting with the model.

## Tools and Technologies
- **Data Processing:** Python, Pandas, NumPy, Scikit-learn
- **Modeling:** Random Forest, XGBoost, AdaBoost
- **Hyperparameter Tuning:** Grid Search, Cross-Validation
- **Deployment:** Docker, AWS (EC2, ECR), GitHub Actions
- **Web Interfaces:** FastAPI, Streamlit
- **Version Control and Experiment Tracking:** GitHub, DVC, MLflow

## Outcome

The project resulted in a robust machine learning model that significantly enhanced the detection of fraudulent transactions, effectively reducing both false positives and false negatives. The model was successfully deployed using Docker on AWS, leveraging ECR and EC2 instances, and was integrated into a CI/CD pipeline with GitHub Actions to ensure continuous updates and monitoring.

MLflow was used for experiment tracking, allowing the team to log and compare different model versions and their performance metrics. It provided valuable insights into model evaluation by tracking parameters, metrics, and artifacts, which aided in making informed decisions during model tuning and selection.

The web interfaces enabled users to easily determine whether a transaction is fraudulent or legitimate based on its details. These interfaces were developed using FastAPI for API-based access and Streamlit for an interactive, web-based experience, ensuring they were user-friendly and flexible.



```bash
Author: Robins Yadav
Data Scientist

```



#### Project setup Notes
- Setup the GitHub repository (creating repo and cloning repo locally)
	by creating a new repository on GitHub or by creating a new directory, and initialize it
	```
	echo "# projectname" >> README.md or touch README.md
	git init
	git add README.md
	git commit -m "first commit"
	git branch -M main
	git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
	git push -u origin main
	```
- Template (Deployable code) using `template.py`
	``` 
	- aml_fraud_detector
		- components
			- data_ingestion.py
			- data_validation.py
			- data_transformation.py
			- model_trainer.py
			- model_evaluation.py
			- model_pusher.py
		- configuration
		- constant
		- entity
			- config_entity.py
			- artifact_entity.py
		- logger
		- pipeline
			- training_pipeline.py
			- prediction_pipeline.py
		- utils
			- main_utils.py
	
	```
- Requirements
```bash
conda create -p venv python=3.8 -y
```
```bash
conda activate venv
```
```bash
pip install -r requirements.txt
Note: -e . at the end in requirements.txt file -> This is for `setup.py` file
```
```bash
python app.py
or
streamlit run app_streamlit.py
```
```bash
open up you local host and port
```
- Database setup - MongoDB
	- create project `aml_fraud_detector_mongoDB`
		- create a cluster `cluster-aml-fraud-detector`
		- Setup Network Access
	- Downloaded data from kaggle and insert or upload the data to the MongoDB database (Atlas)

#### Logger and Exceptions setup
- Create directories for logger and exception: The logger and exception will be imported from here throughout the projects
	- aml_fraud_detector
		- logger
			- __init__.py -> logger related codes
		- exception
			- __init__.py -> expception related codes



#### Workflows 

In progress
1. constants
2. entity


1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml



#### MLflow
- [Documentation](https://mlflow.org/docs/latest/index.html)
```bash
- mlflow ui
```

#### DagsHub
[DagsHub](https://dagshub.com/) 

MLflow is used for tracking experiments, logging metrics, parameters, and artifacts. Integrating DagsHub with MLflow allows you to track machine learning experiments and their associated data and models in one place.

Run this to export as env variables:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/robinyUArizona/DL-EndToEnd-Project.mlflow

export MLFLOW_TRACKING_USERNAME=robinyUArizona 

export MLFLOW_TRACKING_PASSWORD=
```
```bash
python script.py
```



#### DVC 
[DVC](https://dvc.org/)
Open-source version control system for Data Science and Machine Learning projects. Git-like experience to organize your data, models, and experiments.
```bash
1. dvc init
2. dvc repro
3. dvc dag
```


```bash
Author: Robins Yadav
Data Scientist

```