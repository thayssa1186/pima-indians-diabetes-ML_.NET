using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace Model
{
    public class ClassPrediction
    {
        [ColumnName("PredictedLabel")]
        public float Class;
    }
}
