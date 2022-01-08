import pandas as pd
import pyspark
import mrmr

spark_session = pyspark.sql.SparkSession(pyspark.context.SparkContext())

columns = ["target_classif", "target_regression", "some_null", "feature_a", "constant", "feature_b"]
target_column_classif = "target_classif"
target_column_regression = "target_regression"
features = ["some_null", "feature_a", "constant", "feature_b"]

data = [
    ('a', 1.0, 1.0,          2.0, 7.0, 3.0),
    ('a', 2.0, float('NaN'), 2.0, 7.0, 2.0),
    ('b', 3.0, float('NaN'), 3.0, 7.0, 1.0),
    ('b', 4.0, 4.0,          3.0, 7.0, 2.0),
    ('b', 5.0, 5.0,          4.0, 7.0, 3.0),
]

df_pandas = pd.DataFrame(data=data, columns=columns)
df_spark = spark_session.createDataFrame(data=data, schema=columns)


def test_consistency_f_classif():
    f_classif_pandas = mrmr.pandas.f_classif(X=df_pandas.loc[:, features], y=df_pandas.loc[:, target_column_classif])
    f_classif_spark = mrmr.spark.f_classif(df=df_spark, target_column=target_column_classif, features=features)

    assert set(f_classif_pandas.index) == set(f_classif_spark.index)
    assert ((f_classif_pandas - f_classif_spark[f_classif_pandas.index]).abs() < .001).all()


def test_consistency_correlation():
    correlation_pandas = mrmr.pandas.correlation(target_column=target_column_regression, features=features, X=df_pandas)
    correlation_spark = mrmr.spark.correlation(target_column=target_column_regression, features=features, df=df_spark)

    assert set(correlation_pandas.index) == set(correlation_spark.index)
    assert ((correlation_pandas - correlation_spark[correlation_pandas.index]).abs() < .001).all()


def test_consistency_f_regression():
    f_regression_pandas = mrmr.pandas.f_regression(X=df_pandas.loc[:, features], y=df_pandas.loc[:, target_column_regression])
    f_regression_spark = mrmr.spark.f_regression(df=df_spark, target_column=target_column_regression, features=features)

    assert set(f_regression_pandas.index) == set(f_regression_spark.index)
    assert ((f_regression_pandas.drop('some_null') - f_regression_spark[f_regression_pandas.index.drop('some_null')]).abs() < .001).all()
