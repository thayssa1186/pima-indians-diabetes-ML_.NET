using System;
using Microsoft.ML.Runtime.Api;

namespace Model
{
    public class PimaDiabeteModel
    {
        [Column(ordinal: "0")]
        public float Pregnancies;

        [Column(ordinal: "1")]
        public float Glucose;

        [Column(ordinal: "2")]
        public float BloodPressure;

        [Column(ordinal: "3")]
        public float SkinThickness;

        [Column(ordinal: "4")]
        public float Insulin;

        [Column(ordinal: "5")]
        public float BMI;

        [Column(ordinal: "6")]
        public float DiabetesPedigreeFunction;

        [Column(ordinal: "7")]
        public float Age;

        [Column(ordinal: "8", name: "Label")]
        public float Outcome;
    }
}
