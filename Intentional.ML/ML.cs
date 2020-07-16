using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace Intentional.ML
{
    public enum MLType
    {
        TextFeaturizingEstimator,
        LightGbm
    }

    public class ML<T, T2>
        where T : class, new()
    {
        public IEnumerable<T> data { get; set; }

        private bool isTaught = false;
        private bool isLoaded = false;

        public string labelName { get; set; } = "Label";

        public string inputName { get; set; } = "Text";

        public string scoreName { get; set; } = "Score";

        public string featuresName { get; set; } = "Features";

        public bool isFeaturesIncluded { get; set; } = false;

        public string modelName { get; set; } = "_model";

        public MLType type { get; set; }

        public T thisType { get; set; }

        ITransformer mlModel;
        MLContext mlContext = new MLContext();
        ITransformer trainedModel;
        object modelMetrics; //CalibratedBinaryClassificationMetrics

        public ML() { this.thisType = new T(); }

        public ML<T, T2> WhereResultIs(string labelName)
        {
            this.labelName = labelName;
            return this;
        }

        public ML<T, T2> WhereInputIs(string inputName)
        {
            this.inputName = inputName;
            return this;
        }

        public ML<T, T2> WhereScoreIs(string scoreName)
        {
            this.scoreName = scoreName;
            return this;
        }

        public ML<T, T2> WhereFeaturesIs(string featuresName)
        {
            this.featuresName = featuresName;
            this.isFeaturesIncluded = true;
            return this;
        }

        public ML<T, T2> IncludingFeatures()
        {
            this.isFeaturesIncluded = true;
            return this;
        }

        public ML<T, T2> LearnFrom(IEnumerable<T> data)
        {
            this.data = data;
            return this;
        }

        public ML<T, T2> Using(MLType type)
        {
            this.type = type;
            return this;
        }

        public ML<T, T2> Run()
        {
            IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(this.data);

            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data
                .TrainTestSplit(trainingDataView, testFraction: 0.2);

            switch(this.type)
            {
                case MLType.TextFeaturizingEstimator:
                {
                    TextFeaturizingEstimator dataProcessPipeline = mlContext.Transforms.Text
                        .FeaturizeText(outputColumnName: "Features", inputColumnName: this.inputName);

                    SdcaLogisticRegressionBinaryTrainer sdcaRegressionTrainer = mlContext.BinaryClassification.Trainers
                        .SdcaLogisticRegression(labelColumnName: this.labelName, featureColumnName: "Features");

                    EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);

                    trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);
                    mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, this.modelName);
                    IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);

                    this.modelMetrics = mlContext.BinaryClassification
                        .Evaluate(data: testSetTransform,
                                  labelColumnName: this.labelName,
                                  scoreColumnName: this.scoreName);
                    break;
                }
                case MLType.LightGbm:
                {
                    var fields = this.thisType
                        .GetType()
                        .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                        .Select(p => p.Name);
                    //mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")

                    var featurePipeline = this.isFeaturesIncluded
                        ? null
                        : mlContext.Transforms.Concatenate("Features", fields.ToArray());

                    var trainer = mlContext.Regression.Trainers
                        .LightGbm(new LightGbmRegressionTrainer.Options()
                        {
                            NumberOfIterations = 100,
                            LearningRate = 0.3227682f,
                            NumberOfLeaves = 55,
                            MinimumExampleCountPerLeaf = 10,
                            UseCategoricalSplit = false,
                            HandleMissingValue = true,
                            UseZeroAsMissingValue = false,
                            MinimumExampleCountPerGroup = 50,
                            MaximumCategoricalSplitPointCount = 32,
                            CategoricalSmoothing = 20,
                            L2CategoricalRegularization = 5,
                            Booster = new GradientBooster.Options() { L2Regularization = 0, L1Regularization = 0.5 },
                            LabelColumnName = this.labelName,
                            FeatureColumnName = "Features"
                        });

                    var pipeline2 = featurePipeline == null ? null : featurePipeline.Append(trainer);

                    if(pipeline2 == null)
                    {
                        trainedModel = trainer.Fit(dataSplit.TrainSet);
                        mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, this.modelName);

                        IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);

                        var crossValidationResults = mlContext.Regression
                            .CrossValidate(trainingDataView, trainer, numberOfFolds: 5, labelColumnName: this.labelName);
                    }
                    else
                    {
                        trainedModel = pipeline2.Fit(dataSplit.TrainSet);
                        mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, this.modelName);

                        IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);

                        var crossValidationResults = mlContext.Regression
                            .CrossValidate(trainingDataView,
                                           pipeline2,
                                           numberOfFolds: 5,
                                           labelColumnName: this.labelName);
                    }

                    //this.modelMetrics = mlContext.Regression
                    //.Evaluate(data: testSetTransform,
                    //          labelColumnName: this.labelName,
                        //          scoreColumnName: this.scoreName);
                        break;
                }
            }

            //var msg = $"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}" +
            //    $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}" +
            //    $"{Environment.NewLine}" +
            //    $"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}" +
            //    $"F1Score: {modelMetrics.F1Score:P2}{Environment.NewLine}" +
            //    $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
            //    $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}";

            this.isTaught = true;
            return this;
        }

        public ML<T, T2> Load(string modelName = null)
        {
            modelName = modelName ?? this.modelName;
            using(var stream = new FileStream(modelName, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }

            this.isTaught = true;
            this.isLoaded = true;
            return this;
        }

        public PredictionOutput<T2> PredictWithOutput(T input)
        {
            if(!isTaught)
            {
                this.Run();
            }

            if(!this.isLoaded)
            {
                Load(this.modelName);
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<T, PredictionOutput<T2>>(mlModel);

            var prediction = predictionEngine.Predict(input);
            return prediction;
        }

        public PredictionOutput2 PredictWithOutput2(T input)
        {
            if(!isTaught)
            {
                this.Run();
            }

            if(!this.isLoaded)
            {
                Load(this.modelName);
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<T, PredictionOutput2>(mlModel);

            var prediction = predictionEngine.Predict(input);
            return prediction;
        }

        public T2 Predict(T input)
        {
            var prediction = this.PredictWithOutput(input);
            return prediction.Prediction;
        }

        public T PredictAndSet(T input)
        {
            var prediction = this.PredictWithOutput(input);
            input.GetType().GetProperty(this.labelName).SetValue(input, prediction.Score);
            return input;
        }

        public ML<T, T2> Save(string modelName)
        {
            this.modelName = modelName;

            if(!isTaught)
            {
                this.Run();
            }

            return this;
        }
    }

    public class PredictionOutput<T2>
    {
        [ColumnName("Score")]
        public float Score { get; set; }

        [ColumnName("PredictedLabel")]
        public T2 Prediction { get; set; }

        public float Probability { get; set; }
    }

    public class PredictionOutput2
    {
        public float Score { get; set; }
    }

    public class MLFeatureAttribute : Attribute
    {
        private bool isFeature;

        public MLFeatureAttribute(bool isFeature) { this.isFeature = isFeature; }
    }
}
