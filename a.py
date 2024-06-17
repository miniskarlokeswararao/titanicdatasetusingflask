from flask import Flask, render_template, request

import pickle5
# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
with open('C:\\Users\\user\\Downloads\\titanic_model.pkl', 'rb') as model_file:
        loaded_model = pickle5.load(model_file)
        print("Model loaded successfully.")
@app.route('/')
def form():
    return render_template('index.html',title='home')


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()
    