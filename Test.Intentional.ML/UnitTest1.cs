using Intentional.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Linq;

namespace Test.Intentional.ML
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestTextPredictWithOutput()
        {
            var intention = new ML<MyTextRow, bool>();

            IEnumerable<MyTextRow> data = GetTextData();
            var msg = "Parking is terrible";

            var result = intention.Using(MLType.TextFeaturizingEstimator)
                .LearnFrom(data)
                .PredictWithOutput(new MyTextRow { Text = msg });
            Assert.IsTrue(result.Prediction);
        }

        [TestMethod]
        public void TestTextPredict()
        {
            var intention = new ML<MyTextRow, bool>();

            IEnumerable<MyTextRow> data = GetTextData();
            var msg = "Parking is terrible";

            var result = intention.Using(MLType.TextFeaturizingEstimator)
                .LearnFrom(data)
                .Predict(new MyTextRow { Text = msg });
            Assert.IsTrue(result);
        }

        [TestMethod]
        public void TestTextPredictAndSet()
        {
            var intention = new ML<MyTextRow, bool>();

            IEnumerable<MyTextRow> data = GetTextData();
            var msg = "Parking is terrible";

            var result = intention.Using(MLType.TextFeaturizingEstimator)
                .LearnFrom(data)
                .PredictAndSet(new MyTextRow { Text = msg });
            Assert.IsTrue(result.Label);
        }

        public IEnumerable<MyTextRow> GetTextData()
        {
            var data = "0   Great Pizza\n" +
                "0   Awesome customer service\n" +
                "1   Dirty floors\n" +
                "1   Very expensive\n" +
                "0   Toppings are good\n" +
                "1   Parking is terrible\n" +
                "0   Bathrooms are clean\n" +
                "1   Management is unhelpful\n" +
                "0   Lighting and atmosphere are romantic\n" +
                "1   Crust was burnt\n" +
                "0   Pineapple was freshv\n" +
                "1   Lack of garlic cloves is upsetting\n" +
                "0   Good experience, would come back\n" +
                "0   Friendly staff\n" +
                "1   Rude customer service\n" +
                "1   Waiters never came back\n" +
                "1   Could not believe the napkins were $10!\n" +
                "0   Supersized Pizza is a great deal\n" +
                "0   $5 all you can eat deal is good\n" +
                "1   Overpriced and was shocked that utensils were an upcharge";

            return data.Split(new char[] { '\n' }, System.StringSplitOptions.RemoveEmptyEntries)
                .Select(line => new MyTextRow { Label = line.StartsWith("1"), Text = line.Substring(4) });
        }

        [TestMethod]
        public void TestLinearPredictWithOutput()
        {
            var intention = new ML<MyLinearRow, float>();

            IEnumerable<MyLinearRow> data = GetFloatData();

            var result = intention.Using(MLType.LightGbm)
                .WhereScoreIs("Score")
                .LearnFrom(data)
                .PredictWithOutput(new MyLinearRow { Val1 = 1, Val2 = 0, Val3 = 0 });
            var i = 0;
        }

        public IEnumerable<MyLinearRow> GetFloatData()
        {
            var data = "1 0 0 1; 1 1 1 1; 0 1 1 0; 0 0 0 0; 0 1 0 0; 0 0 1 0; 1 0 1 1; 1 1 0 1";
            return data.Split(new char[] { ';' }, System.StringSplitOptions.RemoveEmptyEntries)
                .Select(line =>
                {
                    var parts = line.Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                    return new MyLinearRow
                    {
                        Val1 = float.Parse(parts[0]),
                        Val2 = float.Parse(parts[1]),
                        Val3 = float.Parse(parts[2]),
                        Label = float.Parse(parts[3])
                    };
                });
        }
    }

    public class MyTextRow
    {
        public bool Label { get; set; }

        public string Text { get; set; }
    }

    public class MyLinearRow
    {
        public float Val1 { get; set; }

        public float Val2 { get; set; }

        public float Val3 { get; set; }

        public float Label { get; set; }

        public float Score { get; set; }
    }
}
