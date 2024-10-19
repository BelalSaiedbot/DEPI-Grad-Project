import mlflow
import mlflow.pyfunc
from transformers import pipeline

# Define a class for the sentiment analysis model
class SentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = pipeline('text-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment')

    def predict(self, context, model_input):
        return self.model(model_input)

# Set the experiment name
experiment_name = "sentiment_analysis_class"
mlflow.set_experiment(experiment_name)

# Start the MLflow run
with mlflow.start_run() as run:
    # Initialize the sentiment analysis pipeline
    sa = pipeline('text-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment')

    # List of sentences to analyze
    sentences = ['هذا جيد', 'هذا سئ']

    # Get the predictions
    results = sa(sentences)

    # Print the results
    for result in results:
        print(f"Label: {result['label']}, Score: {result['score']}")

    # Log the model
    mlflow.pyfunc.log_model("sentiment_model", python_model=SentimentModel())

    print("Model logged to MLflow!")

# Optionally, you can print the run ID to track your experiment
print(f"MLflow run ID: {run.info.run_id}")
