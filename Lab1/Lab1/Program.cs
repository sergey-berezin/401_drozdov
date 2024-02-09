using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using YoloPackage;

namespace ConsoleApp
{
    public class Program
    {
        private static CancellationTokenSource cts = new CancellationTokenSource();

        public static async Task Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Wrong program arguments sequence! Try program_name image_path");
                return;
            }

            string ImagePath = args[0];
            List<CSVElement> objects = new();
            SemaphoreSlim addLock = new SemaphoreSlim(1, 1);

            var ProcessingTask = new Task(async () => {
                    try
                    {
                        var image = Image.Load<Rgb24>(ImagePath);
                        var processingTask = Yolo.ObjectDetection(image, cts.Token);

                        var filename = Path.GetFileName(ImagePath);
                        Directory.CreateDirectory("result");
                        var path = $"result/{filename}";

                        processingTask.Wait();
                        image = processingTask.Result.image;
                        List<ObjectBox>? boundingBoxes = processingTask.Result.boxes;
                        var saveImageTask = image.SaveAsJpegAsync(path, cts.Token);

                        addLock.Wait();
                        objects.AddRange(boundingBoxes.Select(
                            bb => new CSVElement(
                                ImagePath,
                                Yolo.Labels[bb.Class],
                                (int)bb.XMin,
                                (int)bb.YMin,
                                (int)(bb.XMax - bb.XMin),
                                (int)(bb.YMax - bb.YMin)
                            )
                        ));
                        addLock.Release();
                        await saveImageTask;
                    }
                    catch (Exception e)
                    {
                        throw new Exception(e.Message);
                    }
                }, cts.Token);

            Console.CancelKeyPress += delegate {
                cts.Cancel();
            };

            ProcessingTask.Start();

            try
            {
                await ProcessingTask;
            }
            catch (Exception e)
            {
                throw new Exception(e.Message);
            }
            SaveToCSV(objects);
        }

        private static void SaveToCSV(List<CSVElement> objects)
        {
            StreamWriter writer = File.CreateText(PathToCSV);
            writer.WriteLine(CSVHead);
            foreach (var obj in objects)
            {
                writer.WriteLine(obj.ToString());
            }
            writer.Close();
        }

        const string PathToCSV = "result/bboxes.csv";
        const string CSVHead = "File name,Class name,X,Y,W,H";
    }

    public class CSVElement 
    {
        private readonly string? filename;
        private readonly string? classname;
        private readonly int? X;
        private readonly int? Y;

        private readonly int? W;
        private readonly int? H;

        public CSVElement(string? filename, string? classname, int? x, int? y, int? w, int? h)
        {
            this.filename = filename;
            this.classname = classname;
            X = x;
            Y = y;
            W = w;
            H = h;
        }

        public override string ToString()
        {
            return $"\"{filename}\",\"{classname}\",{X},{Y},{W},{H}";
        }
    }
}