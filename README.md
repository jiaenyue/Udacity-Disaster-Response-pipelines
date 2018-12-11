# Disaster Response Pipeline Project

### File Description

```
|-- app
|   |-- run.py
|   |-- static
|   |   |-- images
|   |   |   `-- 8781383171_47ea51ec1b_h.jpg
|   |   `-- stylesheet
|   |       `-- bootstrap.min.css
|   |-- templates
|   |   |-- go.html
|   |   `-- master.html
|   `-- tranformer.py
|-- data
|   |-- categories.csv
|   |-- disaster_categories.csv
|   |-- disaster_messages.csv
|   |-- DisasterResponse.db
|   |-- messages.csv
|   `-- process_data.py
|-- models
|   |-- classifier.pkl
|   |-- train_classifier.py
|   `-- tranformer.py
|-- README.md
`-- requirments.txt
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - Install required python packages
        `pip install -r requirements.txt`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
