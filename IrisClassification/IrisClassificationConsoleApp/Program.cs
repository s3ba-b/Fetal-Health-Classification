using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MulticlassClassification_Iris.DataStructures;

namespace MulticlassClassification_Iris
{
    public static partial class Program
    {
        #region data paths
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../../Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/fetalhealth-train.csv";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/fetalhealth-test.csv";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static string BaseModelsRelativePath = @"../../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/FetalHealthClassificationModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);
        #endregion
        private static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);

            //1.
            BuildTrainEvaluateAndSaveModel(mlContext);

            //2.
            TestSomePredictions(mlContext);

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        private static void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<FetalHealthData>(TrainDataPath, hasHeader: false, separatorChar: ',');
            var testDataView = mlContext.Data.LoadFromTextFile<FetalHealthData>(TestDataPath, hasHeader: false, separatorChar: ',');
            

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(FetalHealthData.FetalHealth))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(FetalHealthData.BaselineValue),
                        nameof(FetalHealthData.Accelerations),
                        nameof(FetalHealthData.FetalMovement),
                        nameof(FetalHealthData.UterineContractions),
                        nameof(FetalHealthData.LightDecelerations),
                        nameof(FetalHealthData.SevereDecelerations),
                        nameof(FetalHealthData.ProlonguedDecelerations),
                        nameof(FetalHealthData.AbnormalShortTermVariability),
                        nameof(FetalHealthData.MeanValueOfShortTermVariability),
                        nameof(FetalHealthData.PercentageOfTimeWithAbnormalLongTermVariability),
                        nameof(FetalHealthData.MeanValueOfLongTermVariability),
                        nameof(FetalHealthData.HistogramWidth),
                        nameof(FetalHealthData.HistogramMin),
                        nameof(FetalHealthData.HistogramMax),
                        nameof(FetalHealthData.HistogramNumberOfPeaks),
                        nameof(FetalHealthData.HistogramNumberOfZeroes),
                        nameof(FetalHealthData.HistogramMode),
                        nameof(FetalHealthData.HistogramMean),
                        nameof(FetalHealthData.HistogramMedian),
                        nameof(FetalHealthData.HistogramVariance),
                        nameof(FetalHealthData.HistogramTendency))
                                                                       .AppendCacheCheckpoint(mlContext)); 
                                                                       // Use in-memory cache for small/medium datasets to lower training time. 
                                                                       // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets. 

            // STEP 3: Set the training algorithm, then append the trainer to the pipeline  
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: nameof(FetalHealthData.FetalHealth) , inputColumnName: "KeyColumn"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);

            // STEP 5: Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "FetalHealth", "Score");

            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {trainer.ToString()} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy.ToString(CultureInfo.CurrentCulture)}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy.ToString(CultureInfo.CurrentCulture)}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss.ToString(CultureInfo.CurrentCulture)}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0].ToString(CultureInfo.CurrentCulture)}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1].ToString(CultureInfo.CurrentCulture)}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2].ToString(CultureInfo.CurrentCulture)}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        private static void TestSomePredictions(MLContext mlContext)
        {
            //Test Classification Predictions with some hard-coded samples 
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
        
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<FetalHealthData, FetalHealthPrediction>(trainedModel);
        
            // During prediction we will get Score column with 3 float values.
            // We need to find way to map each score to original label.
            // In order to do that we need to get TrainingLabelValues from Score column.
            // TrainingLabelValues on top of Score column represent original labels for i-th value in Score array.
            // Let's look how we can convert key value for PredictedLabel to original labels.
            // We need to read KeyValues for "PredictedLabel" column.
            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();
        
            // Since we apply MapValueToKey estimator with default parameters, key values
            // depends on order of occurence in data file. Which is "Iris-setosa", "Iris-versicolor", "Iris-virginica"
            // So if we have Score column equal to [0.2, 0.3, 0.5] that's mean what score for
            // Iris-setosa is 0.2
            // Iris-versicolor is 0.3
            // Iris-virginica is 0.5.
            //Add a dictionary to map the above float values to strings. 
            Dictionary<float, string> HealthClasses = new Dictionary<float, string>();
            HealthClasses.Add(1, "Normal");
            HealthClasses.Add(2, "Suspect");
            HealthClasses.Add(3, "Pathological");
        
            Console.WriteLine("=====Predicting using model====");
            //Score sample 1
            var resultprediction1 = predEngine.Predict(SampleFetalHealthData.Fetal1);
        
            Console.WriteLine($"Actual: Suspect.     Predicted label and score:  {HealthClasses[labelsArray[0]]}: {resultprediction1.Score[0].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[1]]}: {resultprediction1.Score[1].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[2]]}: {resultprediction1.Score[2].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine();
        
            //Score sample 2
            var resultprediction2 = predEngine.Predict(SampleFetalHealthData.Fetal2);
        
            Console.WriteLine($"Actual: Suspect.     Predicted label and score:  {HealthClasses[labelsArray[0]]}: {resultprediction2.Score[0].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[1]]}: {resultprediction2.Score[1].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[2]]}: {resultprediction2.Score[2].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine();
        
            //Score sample 3
            var resultprediction3 = predEngine.Predict(SampleFetalHealthData.Fetal3);
        
            Console.WriteLine($"Actual: Suspect.     Predicted label and score:  {HealthClasses[labelsArray[0]]}: {resultprediction3.Score[0].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[1]]}: {resultprediction3.Score[1].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[2]]}: {resultprediction3.Score[2].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine();
            
            //Score sample 4
            var resultprediction4 = predEngine.Predict(SampleFetalHealthData.Fetal4);
        
            Console.WriteLine($"Actual: Suspect.     Predicted label and score:  {HealthClasses[labelsArray[0]]}: {resultprediction4.Score[0].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[1]]}: {resultprediction4.Score[1].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[2]]}: {resultprediction4.Score[2].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine();
            
            //Score sample 5
            var resultprediction5 = predEngine.Predict(SampleFetalHealthData.Fetal5);
        
            Console.WriteLine($"Actual: Normal.     Predicted label and score:  {HealthClasses[labelsArray[0]]}: {resultprediction5.Score[0].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[1]]}: {resultprediction5.Score[1].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine($"                                                {HealthClasses[labelsArray[2]]}: {resultprediction5.Score[2].ToString(CultureInfo.CurrentCulture)}");
            Console.WriteLine();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
