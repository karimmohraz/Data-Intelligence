# SAP Data Intelligence
Content for implementing iris flower classification

Blog [Train and Deploy a Tensorflow Pipeline in SAP Data Intelligence](https://blogs.sap.com/?p=865553&preview=true&preview_id=865553)

* iris.csv: raw data
* keras.ipynb: jupyter notebook for preprocessing, training and evaluating a keras network
* train-graph.json, inference-graph.json: training pipelines
* inference-graph: pipeline for deploying iris classification service
* train-operator.py, inference-operator.py: python / tensorflow code for training and serving iris model
* iris-swagger.json: predict and predict_classes endpoints
* client.ipynb: notebook for testing the deployed service
