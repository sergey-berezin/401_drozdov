using SixLabors.ImageSharp.Drawing.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SixLabors.Fonts;
using YoloModel;
using YoloConstant;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
namespace Yolo
{
    public interface IYoloService
    {
        Task<ProcessedImage> ProcessImageAsync(string path, CancellationToken token);
    }

    public class YoloService : IYoloService
    {

        private static SemaphoreSlim _sessionLock = new SemaphoreSlim(1, 1);
        private async Task DownloadModel(CancellationToken token)
        {
            using var client = new HttpClient();
            using var data = await client.GetStreamAsync(Constant.MODEL_URL, token);
            using var fileStream = new FileStream(Constant.MODEL_NAME, FileMode.OpenOrCreate);
            await data.CopyToAsync(fileStream, token);
        }

        public async Task<InferenceSession> GetModel(CancellationToken token)
        {
            _sessionLock.Wait();

            InferenceSession? ans = null;
            var options = new SessionOptions()
            {
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            while (ans == null)
            {
                try
                {
                    ans = new InferenceSession(Constant.MODEL_NAME, options);
                }
                catch (Exception) { }
                if (ans == null)
                    await DownloadModel(token);
            }

            _sessionLock.Release();

            return ans;
        }

        private void Annotate(Image<Rgb24> target, IEnumerable<ObjectBox> objects)
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
                        $"{Constant.LABELS[objbox.Class]}",
                        SystemFonts.Families.First().CreateFont(16),
                        Color.Blue,
                        new PointF((float)objbox.XMin, (float)objbox.YMax));
                });
            }
        }



        private float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        private int IndexOfMax(float[] values)
        {
            int idx = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[idx])
                    idx = i;
            return idx;
        }

        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private void CheckIfImage(string path)
        {
            try
            {
                var imageInfo = Image.Identify(path);
            }
            catch (ArgumentNullException)
            {
                throw new Exception("Не указан путь к изображению.");
            }
            catch (InvalidImageContentException)
            {
                throw new Exception("Файл не является изображением.");
            }
            catch (UnknownImageFormatException)
            {
                throw new Exception("Неизвестный формат изображения.");
            }
        }

        private List<NamedOnnxValue> GetInputs(Image<Rgb24> img)
        {
            img.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(Constant.TARGET_SIZE),
                Mode = ResizeMode.Pad
            }));

            var input = new DenseTensor<float>(new[] { 1, 3, Constant.TARGET_SIZE, Constant.TARGET_SIZE });
            img.ProcessPixelRows(accessor =>
                {
                    Rgb24 transparent = Color.Transparent;
                    for (int y = 0; y < accessor.Height; y++)
                    {
                        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                        for (int x = 0; x < pixelSpan.Length; x++)
                        {
                            input[0, 0, y, x] = pixelSpan[x].R;
                            input[0, 1, y, x] = pixelSpan[x].G;
                            input[0, 2, y, x] = pixelSpan[x].B;
                        }
                    }
                });

            return new List<NamedOnnxValue>
            {
               NamedOnnxValue.CreateFromTensor("image", input),
            };
        }

        private List<ObjectBox> GetObjectBoxes(Tensor<float> outputs)
        {
            var objects = new List<ObjectBox>();
            for (var row = 0; row < Constant.CELL_COUNT; row++)
                for (var col = 0; col < Constant.CELL_COUNT; col++)
                    for (var box = 0; box < Constant.BOXES_PER_CELL; box++)
                    {
                        var rawX = outputs[0, (5 + Constant.CLASS_COUNT) * box, row, col];
                        var rawY = outputs[0, (5 + Constant.CLASS_COUNT) * box + 1, row, col];

                        var rawW = outputs[0, (5 + Constant.CLASS_COUNT) * box + 2, row, col];
                        var rawH = outputs[0, (5 + Constant.CLASS_COUNT) * box + 3, row, col];

                        var x = (float)((col + Sigmoid(rawX)) * Constant.CELL_SIZE);
                        var y = (float)((row + Sigmoid(rawY)) * Constant.CELL_SIZE);

                        var w = (float)(Math.Exp(rawW) * Constant.ANCHORS[box * 2] * Constant.CELL_SIZE);
                        var h = (float)(Math.Exp(rawH) * Constant.ANCHORS[box * 2 + 1] * Constant.CELL_SIZE);

                        var conf = Sigmoid(outputs[0, (5 + Constant.CLASS_COUNT) * box + 4, row, col]);

                        if (conf > 0.5)
                        {
                            var classes
                            = Enumerable
                            .Range(0, Constant.CLASS_COUNT)
                            .Select(i => outputs[0, (5 + Constant.CLASS_COUNT) * box + 5 + i, row, col])
                            .ToArray();
                            objects.Add(new ObjectBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf, IndexOfMax(Softmax(classes))));
                        }
                    }
            return objects;
        }

        private void GetDistinct(ref List<ObjectBox> objects)
        {
            for (int i = 0; i < objects.Count; i++)
            {
                var o1 = objects[i];
                for (int j = i + 1; j < objects.Count;)
                {
                    var o2 = objects[j];
                    Console.WriteLine($"IoU({i},{j})={o1.IoU(o2)}");
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
        }

        private string GetFileName(string path)
        {
            return path.Split('/').Last();
        }

        public async Task<ProcessedImage> ProcessImageAsync(string path, CancellationToken token)
        {
            CheckIfImage(path);
            using Image<Rgb24> img = Image.Load<Rgb24>(path);
            var session = await GetModel(token);
            var imgName = GetFileName(path);
            var inputs = GetInputs(img);
            using var results = session.Run(inputs);
            var outputs = results.First().AsTensor<float>();
            var objects = GetObjectBoxes(outputs);
            GetDistinct(ref objects);
            var csvStrings = objects.Select(
                obj => Environment.NewLine + string.Join(',',
                    imgName,
                    Constant.LABELS[obj.Class],
                    (int)obj.XMin,
                    (int)obj.YMin,
                    (int)(obj.XMax - obj.XMin),
                    (int)(obj.YMax - obj.YMin)
                           )
                       );
            var annotated = img.Clone();
            Annotate(annotated, objects);
            return new ProcessedImage(annotated, imgName, csvStrings);
        }
    }
}