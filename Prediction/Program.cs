using Microsoft.ML;
using Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Prediction
{
    class Program
    {
        static readonly IEnumerable<PimaDiabeteModel> predictClassData = new[]
       {
            new PimaDiabeteModel
            {
                Age = 50,
                BloodPressure = 72,
                BMI = 33.6f,
                DiabetesPedigreeFunction = 0,
                Glucose = 148,
                Insulin = 0,
                Pregnancies = 6,
                SkinThickness = 35
               
            }
        };

        const string modelPath = @".\Learned\Model.zip";

        public static void Main(string[] args)
        {
            Task.Run(async () =>
            {
                var model = await PredictAsync(modelPath, predictClassData);

                Console.WriteLine();
                Console.WriteLine("Please enter another string to classify or just <Enter> to exit the program.");
                Console.WriteLine("Sample string (without quotes): '6,80,66,30,0,26.2,0.313,41'");

                var input = string.Empty;

                while (string.IsNullOrEmpty(input = Console.ReadLine()) == false)
                {
                    try
                    {
                        var inputObj = readLine(input);

                        IEnumerable<PimaDiabeteModel> predictInput = new[]
                        {
                            inputObj
                        };

                        model = await PredictAsync(modelPath, predictInput, model);
                    }
                    catch
                    {
                        Console.WriteLine("Syntax error. Please input a value string...");
                    }
                }

                Console.WriteLine("Press any key to end program...");
                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        internal static PimaDiabeteModel readLine(string input)
        {
            if (string.IsNullOrEmpty(input) == true)
                return null;

            string[] commaSepList = input.Split(',');

            if (commaSepList == null)
                return null;

            if (commaSepList.Length != 8)
                return null;

            return new PimaDiabeteModel
            {
                Pregnancies = float.Parse(commaSepList[0]),
                Glucose = float.Parse(commaSepList[1]),
                BloodPressure = float.Parse(commaSepList[2]),
                SkinThickness = float.Parse(commaSepList[3]),
                Insulin = float.Parse(commaSepList[4]),
                BMI = float.Parse(commaSepList[5]),
                DiabetesPedigreeFunction = float.Parse(commaSepList[6]),
                Age = float.Parse(commaSepList[7])
                
            };
        }

        /// <summary>
        /// Predicts the test data outcomes based on a model that can be
        /// loaded via path or be given via parameter to this method.
        /// 
        /// Creates test data.
        /// Predicts classification based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="model"></param>
        internal static async Task<PredictionModel<PimaDiabeteModel, ClassPrediction>> PredictAsync(
            string modelPath,
            IEnumerable<PimaDiabeteModel> predicts = null,
            PredictionModel<PimaDiabeteModel, ClassPrediction> model = null)
        {
            if (model == null)
            {
                model = await PredictionModel.ReadAsync<PimaDiabeteModel, ClassPrediction>(modelPath);
            }

            if (predicts == null) // do we have input to predict a result?
                return model;

            // Use the model to predict the classification of the data.
            IEnumerable<ClassPrediction> predictions = model.Predict(predicts);

            Console.WriteLine();
            Console.WriteLine("Classification Predictions");
            Console.WriteLine("--------------------------");

            // Builds pairs of (input, prediction)
            IEnumerable<(PimaDiabeteModel input, ClassPrediction prediction)> inputsAndPredictions =
                predicts.Zip(predictions, (input, prediction) => (input, prediction));

            foreach (var item in inputsAndPredictions)
            {
                Console.WriteLine("    Pregnancies: {0}", item.input.Pregnancies);
                Console.WriteLine("    Glucose: {0}", item.input.Glucose);
                Console.WriteLine("    BloodPressure: {0}", item.input.BloodPressure);
                Console.WriteLine("    SkinThickness: {0}", item.input.SkinThickness);
                Console.WriteLine("    Insulin: {0}", item.input.Insulin);
                Console.WriteLine("    BMI: {0}", item.input.BMI);
                Console.WriteLine("    DiabetesPedigreeFunction: {0}", item.input.DiabetesPedigreeFunction);
                Console.WriteLine("    Age: {0}", item.input.Age);

                Console.WriteLine("Predicted Diabete: {0}", item.prediction.Class == 1 ? "Positivo":"Negativo");
            }
            Console.WriteLine();

            return model;
        }
    }
}
