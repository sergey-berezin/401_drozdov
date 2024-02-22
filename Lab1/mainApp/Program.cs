using Yolo;
using System.Text;
using SixLabors.ImageSharp;

class Program
{
    private static CancellationTokenSource cts = new CancellationTokenSource();
    private static IYoloService? _yoloService;
    static async Task Main(string[] args)
    {
        _yoloService = new YoloService();
        var csvFilePath = $"logs/{DateTime.UtcNow:yyyy-MM-dd-HH-mm-ss}.csv";
        var stringBuilder = new StringBuilder("filename,class,X,Y,W,H");
        var imageProcessTasks = Task.WhenAll(args.Select(a => _yoloService.ProcessImageAsync(a, cts.Token)));
        var processedImages = await imageProcessTasks;
        await Task.WhenAll(processedImages.Select(a => a.img.SaveAsJpegAsync("processedImgs/" + a.imgName)));

        foreach (var processedImage in processedImages)
            stringBuilder.AppendJoin(string.Empty, processedImage.csvStrings);

        using (StreamWriter sw = File.CreateText(csvFilePath))
            sw.WriteLine(stringBuilder);
    }

}