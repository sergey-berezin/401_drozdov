using SixLabors.ImageSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;
using SixLabors.Fonts;
using System.Net;

namespace YoloPackage
{
    public static class Yolo
    {
        private const string ModelURL = "https://storage.yandexcloud.net/dotnet4/tinyyolov2-8.onnx";
        private const string ModelFilename = "yolomodel.onnx";
        private const int TargetSize = 416;
        private const int CellCount = 13;
        private const int BoxCount = 5;
        private const int ClassCount = 20;
        public static readonly string[] Labels = new string[] {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };
        private static readonly (double, double)[] Anchors = new (double, double)[] {
            (1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52)
        };
        private static InferenceSession? Session = null;
        private static int timesToReload = 10; 
        private static SemaphoreSlim SessionLock = new SemaphoreSlim(1, 1);

        private static void DownloadModel(CancellationToken token)
        {
            WebClient client = new();
            for (int i = 0; i < timesToReload; ++i)
            {
                if (!File.Exists(ModelFilename))
                {
                    client.DownloadFile(ModelURL, ModelFilename);
                    break;
                }
            }

            if (!File.Exists(ModelFilename))
            {
                throw new Exception("Unable to download model!");
            }

            try
            {
                Session = new InferenceSession(ModelFilename);
            }
            catch
            {
                throw new Exception("Session is null!");
            }
        }

        private static Tensor<float> Inference(List<NamedOnnxValue> inputs)
        {
            SessionLock.Wait();
            if (Session == null)
            {
                throw new Exception("Current session is not exist!");
            }
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = Session.Run(inputs);
            SessionLock.Release();
            return outputs[0].AsTensor<float>();
        }

        private static List<ObjectBox> GetBoxes(Tensor<float> outputs)
        {
            List<ObjectBox> objects = new();
            int cellSize = TargetSize / CellCount;

            for (var row = 0; row < CellCount; row++)
                for (var col = 0; col < CellCount; col++)
                    for (var box = 0; box < BoxCount; box++)
                    {
                        var rawX = outputs[0, (5 + ClassCount) * box, row, col];
                        var rawY = outputs[0, (5 + ClassCount) * box + 1, row, col];

                        var rawW = outputs[0, (5 + ClassCount) * box + 2, row, col];
                        var rawH = outputs[0, (5 + ClassCount) * box + 3, row, col];

                        var x = (float)((col + Sigmoid(rawX)) * cellSize);
                        var y = (float)((row + Sigmoid(rawY)) * cellSize);

                        var w = (float)(Math.Exp(rawW) * Anchors[box].Item1 * cellSize);
                        var h = (float)(Math.Exp(rawH) * Anchors[box].Item2 * cellSize);

                        var conf = Sigmoid(outputs[0, (5 + ClassCount) * box + 4, row, col]);

                        if (conf > 0.5)
                        {
                            var classes
                            = Enumerable
                            .Range(0, ClassCount)
                            .Select(i => outputs[0, (5 + ClassCount) * box + 5 + i, row, col])
                            .ToArray();
                            objects.Add(
                                new ObjectBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf, IndexOfMax(Softmax(classes))));
                        }
                    }

            // Removing duplicates
            for (int i = 0; i < objects.Count; i++)
            {
                var o1 = objects[i];
                for (int j = i + 1; j < objects.Count;)
                {
                    var o2 = objects[j];
                    if (o1.Class == o2.Class && o1.IoU(o2) > 0.6)
                    {
                        if (o1.Confidence < o2.Confidence)
                        {
                            objects[i] = o1 = objects[j];
                        }
                        objects.RemoveAt(j);
                    }
                    else
                    {
                        j++;
                    }
                }
            }
            return objects;
        }

        private static void Annotate(Image<Rgb24> target, List<ObjectBox> objects)
        {
            foreach (var objbox in objects)
            {
                target.Mutate(ctx =>
                {
                    ctx.DrawPolygon(
                        Pens.Solid(Color.Blue, 2),
                        new PointF[] {
                            new PointF((float)objbox.XMin, (float)objbox.YMin),
                            new PointF((float)objbox.XMin, (float)objbox.YMax),
                            new PointF((float)objbox.XMax, (float)objbox.YMax),
                            new PointF((float)objbox.XMax, (float)objbox.YMin)
                        });

                    ctx.DrawText(
                        $"{Labels[objbox.Class]}",
                        SystemFonts.Families.First().CreateFont(16),
                        Color.Blue,
                        new PointF((float)objbox.XMin, (float)objbox.YMax));
                });
            }
        }

        private static Image<Rgb24> ImageResizing(Image<Rgb24> image)
        {
            return image.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetSize, TargetSize),
                    Mode = ResizeMode.Pad
                });
            });
        }

        private static List<NamedOnnxValue> CreateInputs(Image<Rgb24> image)
        {
            var input = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });
            image.ProcessPixelRows(pa =>
            {
                for (int y = 0; y < TargetSize; y++)
                {
                    Span<Rgb24> pixelSpan = pa.GetRowSpan(y);
                    for (int x = 0; x < TargetSize; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].R;
                        input[0, 1, y, x] = pixelSpan[x].G;
                        input[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            });
            return new List<NamedOnnxValue>
            {
               NamedOnnxValue.CreateFromTensor("image", input)
            };
        }

        public static async Task<SegResult> ObjectDetection(Image<Rgb24> image, CancellationToken token)
        {
            var downloadTask = Task.Run(() => DownloadModel(token), token);
            var preprocessingTask = Task.Run(() => ImageResizing(image), token);
            var createInputsTask = Task.Run(() => CreateInputs(preprocessingTask.Result), token);

            await Task.WhenAll(new Task[] { downloadTask, createInputsTask });

            var inferenceTask = Task.Run(() => Inference(createInputsTask.Result), token);

            var getBoxesTask = Task.Run(() => GetBoxes(inferenceTask.Result), token);
            var annotatingTask = Task.Run(
                () =>  Annotate(preprocessingTask.Result, getBoxesTask.Result), token
            );

            await annotatingTask;
            return new SegResult(preprocessingTask.Result, getBoxesTask.Result);
        }

        public static float Sigmoid(float value)
        {
            var e = (float)Math.Exp(value);
            return e / (1.0f + e);
        }

        public static float[] Softmax(float[] values)
        {
            var exps = values.Select(v => Math.Exp(v));
            var sum = exps.Sum();
            return exps.Select(e => (float)(e / sum)).ToArray();
        }

        public static int IndexOfMax(float[] values)
        {
            int idx = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[idx])
                    idx = i;
            return idx;
        }
    }

    public record ObjectBox(double XMin, double YMin, double XMax, double YMax, double Confidence, int Class)
    {
        public double IoU(ObjectBox b2) =>
            (Math.Min(XMax, b2.XMax) - Math.Max(XMin, b2.XMin)) * (Math.Min(YMax, b2.YMax) - Math.Max(YMin, b2.YMin)) /
            ((Math.Max(XMax, b2.XMax) - Math.Min(XMin, b2.XMin)) * (Math.Max(YMax, b2.YMax) - Math.Min(YMin, b2.YMin)));
    }

    public class SegResult
    {
        public Image<Rgb24>? image;

        public List<ObjectBox>? boxes;
        public SegResult(Image<Rgb24> image, List<ObjectBox> boxes) 
        {
            this.image = image;
            this.boxes = boxes;
        }
    }

}