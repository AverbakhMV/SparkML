'''
Обучение модели, выбор лучшей модели и подбор гиперпараметров
Сохранение модели в файл
'''
import operator
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.pipeline import PipelineModel

MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'

def fit_func(classifier_type, training_data, test_data):
    evaluator = MulticlassClassificationEvaluator(labelCol="is_bot", predictionCol="prediction", metricName="accuracy")
    classifier = classifier_type(labelCol="is_bot", featuresCol="features")
    model = classifier.fit(training_data)
    prediction = model.transform(test_data)
    accuracy = evaluator.evaluate(prediction)
    return [classifier, accuracy]

def best_model (model, accuracy, best_m, max_accuracy=0):
    if accuracy>max_accuracy:
        max_accuracy = accuracy
        best_m = model
    return (best_m, max_accuracy)

def process(spark, data_path, model_path):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param model_path: путь сохранения обученной модели
    """
    df = spark.read.parquet(data_path)
    user_type_index = StringIndexer(inputCol='user_type', outputCol="user_type_index")
    platform_index = StringIndexer(inputCol='platform', outputCol="platform_index")
    df = user_type_index.fit(df).transform(df)
    df = platform_index.fit(df).transform(df)
    features = ['duration', 'item_info_events', 'select_item_events',
                'make_order_events', 'events_per_min', 'user_type_index', 'platform_index']
    feature = VectorAssembler(inputCols=features, outputCol="features")
    feature_vector = feature.transform(df)
    training_data, test_data = feature_vector.randomSplit([0.8, 0.2], seed=42)
    best = best_model(*fit_func(DecisionTreeClassifier, training_data, test_data), DecisionTreeClassifier)
    best = best_model(*fit_func(RandomForestClassifier, training_data, test_data), *best)
    best = best_model(*fit_func(GBTClassifier, training_data, test_data), *best)
    classifier = best[0]
    df = spark.read.parquet(data_path)
    user_type_indexer = StringIndexer(inputCol='user_type', outputCol="user_type_index")
    platform_indexer = StringIndexer(inputCol='platform', outputCol="platform_index")
    pipeline = Pipeline(stages=[user_type_indexer, platform_indexer, feature, classifier])
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, [2, 3, 4]) \
        .addGrid(classifier.maxBins, [4, 5, 6]) \
        .addGrid(classifier.minInfoGain, [0.05, 0.1, 0.15]) \
        .build()
    evaluator = MulticlassClassificationEvaluator(labelCol="is_bot", predictionCol="prediction", metricName="accuracy")
    tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               trainRatio=0.8)
    model = tvs.fit(df)
    best_mod = model.bestModel
    best_mod.write().overwrite().save(model_path)


def main(data_path, model_path):
    spark = _spark_session()
    process(spark, data_path, model_path)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)
