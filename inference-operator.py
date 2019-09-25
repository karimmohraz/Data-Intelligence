import json
import io
import tensorflow as tf
import numpy as np

# Global vars to keep track of model status
model = None
model_ready = False

# Validate input data is JSON
def is_json(data):
  try:
    json_object = json.loads(data)
  except ValueError as e:
    return False
  return True

# When Model Blob reaches the input port
def on_model(model_blob):
    global model
    global model_ready

    model = model_blob
    model_ready = True
    api.logger.info("Model Received & Ready")

# Client POST request received
def on_input(msg):
    error_message = ""
    success = False
    try:
        attr = msg.attributes
        request_id = attr['message.request.id']
        
        api.logger.info("POST request received from Client - checking if model is ready")
        if model_ready:
            api.logger.info("Model Ready")
            api.logger.info("Received data from client - validating json input")
            
            user_data = msg.body.decode('utf-8')
            # Received message from client, verify json data is valid
            if is_json(user_data):
                api.logger.info("Received valid json data from client - ready to use")
                
                # apply your model
                model_stream = io.BytesIO(model)
                model_stream.read()
                model_iris = tf.keras.models.load_model(model_stream)
                model_stream.close()
                api.logger.info('load_model')

                # obtain your results
                feed = json.loads(user_data)
                iris_data = np.array(feed['data'])
                api.logger.info(str(iris_data))
                
                # check path
                op_id = attr['openapi.operation_id']
                api.logger.info('operation_id: ' + op_id )
                if 'predict_classes' in op_id:
                    prediction = model_iris.predict_classes(iris_data).tolist()
                else:
                    prediction = model_iris.predict(iris_data).tolist()
                api.logger.info(str(prediction))

                success = True
            else:
                api.logger.info("Invalid JSON received from client - cannot apply model.")
                error_message = "Invalid JSON provided in request: " + user_data
                success = False
        else:
            api.logger.info("Model has not yet reached the input port - try again.")
            error_message = "Model has not yet reached the input port - try again."
            success = False
    except Exception as e:
        api.logger.error(e)
        error_message = "An error occurred: " + str(e)
    
    if success:
        # apply carried out successfully, send a response to the user
        result = json.dumps({'Results': prediction})
    else:
        result = json.dumps({'Error': error_message})
    
    request_id = msg.attributes['message.request.id']
    response = api.Message(attributes={'message.request.id': request_id}, body=result)
    api.send('output', response)

    
api.set_port_callback("model", on_model)
api.set_port_callback("input", on_input)
