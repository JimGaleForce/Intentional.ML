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
        public void TestPredictWithOutput()
        {
            var intention = new ML<MyRow, bool>();

            IEnumerable<MyRow> data = GetTextData();
            var msg = "Parking is terrible";

            var result = intention.LearnFrom(data).PredictWithOutput(new MyRow { Text = msg });
            Assert.IsTrue(result.Prediction);
        }

        [TestMethod]
        public void TestPredict()
        {
            var intention = new ML<MyRow, bool>();

            IEnumerable<MyRow> data = GetTextData();
            var msg = "Parking is terrible";

            var result = intention.LearnFrom(data).Predict(new MyRow { Text = msg });
            Assert.IsTrue(result);
        }

        [TestMethod]
        public void TestPredictAndSet()
        {
            var intention = new ML<MyRow, bool>();

            IEnumerable<MyRow> data = GetTextData();
            var msg = "Parking is terrible";

            var result = intention.LearnFrom(data).PredictAndSet(new MyRow { Text = msg });
            Assert.IsTrue(result.Label);
        }

        public IEnumerable<MyRow> GetTextData()
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
                .Select(line => new MyRow { Label = line.StartsWith("1"), Text = line.Substring(4) });
        }
    }

    public class MyRow
    {
        public bool Label { get; set; }

        public string Text { get; set; }
    }
}
