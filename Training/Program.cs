using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Model;
using System;
using System.Threading.Tasks;

namespace Training
{
    class Program
    {
        const string trainingDataFile = @".\Data\pima-indians-diabetes-Train.csv";
        const string testDataFile = @".\Data\pima-indians-diabetes-Test.csv";
        const string modelPath = @".\Data\Model.zip";

        public static void Main(string[] args)
        {
            Task.Run(async () =>
            {
                // Get a model trained to use for evaluation
                Console.WriteLine("Training Data Set");
                Console.WriteLine("-----------------");
                var model = await TrainAsync(trainingDataFile, modelPath);

                Console.WriteLine();
                Console.WriteLine("Evaluating Training Results");
                Console.WriteLine("---------------------------");
                Evaluate(model, testDataFile);

                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Trains a model using the configured data file in <paramref name="trainingDataFile"/>
        /// and outputs a model as configured in the <paramref name="modelPath"/> parameter.
        /// </summary>
        /// <param name="trainingDataFile"></param>
        /// <param name="modelPath"></param>
        /// <returns></returns>
        internal static async Task<PredictionModel<PimaDiabeteModel, ClassPrediction>>
            TrainAsync(string trainingDataFile, string modelPath)
        {
            // LearningPipeline allows you to add steps in order to keep everything together 
            // during the learning process.  
            var pipeline = new LearningPipeline();

            // The TextLoader loads a dataset with comments and corresponding postive or negative sentiment. 
            // When you create a loader, you specify the schema by passing a class to the loader containing
            // all the column names and their types. This is used to create the model, and train it. 
            pipeline.Add(new TextLoader(trainingDataFile).CreateFrom<PimaDiabeteModel>(useHeader: true, separator: ','));

            pipeline.Add(new Dictionarizer("Label"));

            // TextFeaturizer is a transform that is used to featurize an input column. 
            // This is used to format and clean the data.
            pipeline.Add(new ColumnConcatenator("Features", "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // Train the pipeline based on the dataset that has been loaded, transformed.
            PredictionModel<PimaDiabeteModel, ClassPrediction> model =
                                pipeline.Train<PimaDiabeteModel, ClassPrediction>();

            await model.WriteAsync(modelPath);

            return model;
        }

        /// <summary>
        /// Evaluates the trained model for quality assurance against a
        /// second independent test data set.
        /// 
        /// Loads the test dataset.
        /// Creates the binary evaluator.
        /// Evaluates the model and create metrics.
        /// 
        /// Displays the metrics.
        /// </summary>
        /// <param name="model"></param>
        internal static void Evaluate(
            PredictionModel<Model.PimaDiabeteModel, ClassPrediction> model,
            string testDatafile)
        {
            // loads the new test dataset with the same schema.
            // You can evaluate the model using this dataset as a quality check.
            var testData = new TextLoader(testDatafile).CreateFrom<PimaDiabeteModel>(useHeader:true,separator: ',');

            // Computes the quality metrics for the PredictionModel using the specified dataset.
            var evaluator = new ClassificationEvaluator();

            // The BinaryClassificationMetrics contains the overall metrics computed by binary
            // classification evaluators. To display these to determine the quality of the model,
            // you need to get the metrics first.
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            // Displaying the metrics for model validation
            // Displaying the metrics for model validation
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine("  Accuracy Macro: {0:P2}", metrics.AccuracyMacro);
            Console.WriteLine("  Accuracy Micro: {0:P2}", metrics.AccuracyMicro);
            Console.WriteLine("   Top KAccuracy: {0:P2}", metrics.TopKAccuracy);

            Console.WriteLine("         LogLoss: {0:P2}", metrics.LogLoss);

            Console.WriteLine("");
            Console.WriteLine(" PerClassLogLoss:");
            for (int i = 0; i < metrics.PerClassLogLoss.Length; i++)
                Console.WriteLine("       Class: {0} - {1:P2}", i, metrics.PerClassLogLoss[i]);
        }
    }
}
