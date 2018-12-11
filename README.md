# Disaster Response Pipeline Project
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### File Description

```
|-- app
|   |-- run.py                                  # Main flask app file
|   |-- static                                  
|   |   |-- images
|   |   |   `-- 8781383171_47ea51ec1b_h.jpg     # Backgroud image
|   |   `-- stylesheet
|   |       `-- bootstrap.min.css               # Bootstrap theme
|   |-- templates
|   |   |-- go.html                             # Classify message result page template
|   |   `-- master.html                         # Main page template
|   `-- tranformer.py                           # TextLengthExtractor tranformer class for sklearn pipeline
|-- data
|   |-- disaster_categories.csv                 # Categories data
|   |-- disaster_messages.csv                   # Messages data
|   |-- DisasterResponse.db                     # Sqlite db for save pre-processed data
|   `-- process_data.py                         # Data pre process script
|-- models
|   |-- classifier.pkl                          # Saved model file
|   |-- train_classifier.py                     # Model training script
|   `-- tranformer.py                           # TextLengthExtractor tranformer class for sklearn pipeline
|-- README.md                                   
`-- requirments.txt                             # Used by pip to install required python packages
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
