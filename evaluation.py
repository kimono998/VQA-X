# Description: This file contains the functions to evaluate the model
import evaluate




def metric_computation(prediction, references, metric):
    metric = evaluate.load(metric)
    result = metric.compute(predictions=prediction,
                            references=references)

    return result

