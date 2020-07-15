using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Intentional.ML
{
    public class ML<T, T2>
        where T : class, new()
    {
        public IEnumerable<T> data { get; set; }

        private bool isTaught = false;
        private bool isLoaded = false;

        public string labelName { get; set; } = "Label";

        public string inputName { get; set; } = "Text";

        public string scoreName { get; set; } = "Score";

        public string modelName { get; set; } = "_model";

        ITransformer mlModel;
        MLContext mlContext = new MLContext();
        ITransformer trainedModel;
        CalibratedBinaryClassificationMetrics modelMetrics;

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

        public ML<T, T2> LearnFrom(IEnumerable<T> data)
        {
            this.data = data;
            return this;
        }

        public ML<T, T2> Run()
        {
            IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(this.data);

            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data
                .TrainTestSplit(trainingDataView, testFraction: 0.2);

            TextFeaturizingEstimator dataProcessPipeline = mlContext.Transforms.Text
                .FeaturizeText(outputColumnName: "Features", inputColumnName: this.inputName);

            SdcaLogisticRegressionBinaryTrainer sdcaRegressionTrainer = mlContext.BinaryClassification.Trainers
                .SdcaLogisticRegression(labelColumnName: this.labelName, featureColumnName: "Features");

            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);

            trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, this.modelName);
            IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            this.modelMetrics = mlContext.BinaryClassification
                .Evaluate(data: testSetTransform, labelColumnName: this.labelName, scoreColumnName: this.scoreName);

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

        public ML<T, T2> Load(string modelName)
        {
            using(var stream = new FileStream(modelName, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }

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

        public T2 Predict(T input)
        {
            var prediction = this.PredictWithOutput(input);
            return prediction.Prediction;
        }

        public T PredictAndSet(T input)
        {
            var prediction = this.PredictWithOutput(input);
            input.GetType().GetProperty(this.labelName).SetValue(input, prediction.Prediction);
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
        [ColumnName("PredictedLabel")]
        public T2 Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
