using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace TfConsoleApp
{
    public class TensorFlowHelp
    {
        #region TestMethod1
        public static void TestMethod1()
        {
            using (var session = new TFSession())
            {
                var graph = session.Graph;
                Console.WriteLine(TFCore.Version);

                var a = graph.Const(2);
                var b = graph.Const(3);
                Console.WriteLine("a=2 b=3");
            }            
        }
        #endregion

        #region BasicOperation
        /// <summary>
        /// 基础常量运算，演示了常量的使用
        /// </summary>
        public static void BasicOperation()
        {
            using (var s = new TFSession())
            {
                var g = s.Graph;

                //建立两个TFOutput，都是常数
                var v1 = g.Const(1.5);
                var v2 = g.Const(0.5);

                //建立一个相加的运算
                var add = g.Add(v1, v2);

                //获得runner
                var runner = s.GetRunner();

                //相加
                var result = runner.Run(add);

                //获得result的值2
                Console.WriteLine($"相加的结果:{result.GetValue()}");

                // 相加的结果:2
            }
        }
        #endregion

        #region BasicPlaceholderOperation
        /// <summary>
        /// 基础占位符运算
        /// </summary>
        public static void BasicPlaceholderOperation()
        {
            using (var s = new TFSession())
            {
                var g = s.Graph;

                //占位符 - 一种不需要初始化，在运算时再提供值的对象
                //1*2的占位符
                var v1 = g.Placeholder(TFDataType.Double, new TFShape(2));
                var v2 = g.Placeholder(TFDataType.Double, new TFShape(2));

                //建立一个相乘的运算
                var add = g.Mul(v1, v2);

                //获得runner
                var runner = s.GetRunner();

                //相加
                //在这里给占位符提供值
                var data1 = new double[] { 0.3, 0.5 };
                var data2 = new double[] { 0.4, 0.8 };

                var result = runner
                    .Fetch(add)
                    .AddInput(v1, new TFTensor(data1))
                    .AddInput(v2, new TFTensor(data2))
                    .Run();

                var dataResult = (double[])result[0].GetValue();

                //获得result的值
                Console.WriteLine($"相乘的结果: [{dataResult[0]}, {dataResult[1]}]");

                // 相乘的结果: [0.12, 0.4]
            }
        }
        #endregion

        #region BasicMatrixOperation
        /// <summary>
        /// 基础矩阵运算
        /// </summary>
        public static void BasicMatrixOperation()
        {
            using (var s = new TFSession())
            {
                var g = s.Graph;

                //1x2矩阵
                var matrix1 = g.Const(new double[,] { { 1, 2 } });

                //2x1矩阵
                var matrix2 = g.Const(new double[,] { { 3 }, { 4 } });

                var product = g.MatMul(matrix1, matrix2);
                var result = s.GetRunner().Run(product);
                Console.WriteLine("矩阵相乘的值：" + ((double[,])result.GetValue())[0, 0]);

                // 矩阵相乘的值：11
            };
        }
        #endregion

    }
}
