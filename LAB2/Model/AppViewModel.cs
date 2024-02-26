using LAB2.Command;
using Ookii.Dialogs.Wpf;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Bmp;
using SixLabors.ImageSharp.Processing;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using Yolo;
using YoloModel;

namespace LAB2.Model
{
    public class ObjectVm {
        public string Class { get; set; }
        public double Confidence { get; set; }
        private Image _img { get; set; }
        public BitmapImage Img { get { return _img.ImageSharpToBitmap(); } }
        private Image _imgCropped { get; set; }
        public BitmapImage ImgCropped { get { return _imgCropped.ImageSharpToBitmap(); } }

        public ObjectVm(Image img, Image cropped,  string _class, double confidence) {
            _imgCropped = cropped;
            _img = img;
            Confidence = confidence;
            Class = _class;
        } 
    
    }

    public static  class BitmapImageUtil{
        public static BitmapImage ImageSharpToBitmap(this SixLabors.ImageSharp.Image img)
        {
            if (img == null) return new BitmapImage();
            BitmapImage bitmap = new BitmapImage();

            using (MemoryStream stream = new MemoryStream())
            {
                img.Save(stream, BmpFormat.Instance);

                stream.Seek(0, SeekOrigin.Begin);

                bitmap.BeginInit();
                bitmap.StreamSource = stream;
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();
            }

            return bitmap;
        }
    }

    public class AppViewModel : INotifyPropertyChanged
    {
        private SemaphoreSlim _sessionLock = new SemaphoreSlim(1, 1);
        private bool _loading = false;
        private IYoloService? _yoloService;
        private CancellationTokenSource _cts { get; set; }
        private string _imgsDir { get; set; } = string.Empty;
        public ICommand selectImgsDir { get; set; }
        public ICommand processImgs { get; set; }
        public ICommand cancel { get; private set; }
        public event PropertyChangedEventHandler PropertyChanged;

        public IEnumerable<ObjectVm> objects { get; private set; }

        public string imgsDir
        {
            get => _imgsDir;
            set
            {
                if (value != null && value != _imgsDir)
                {
                    _imgsDir = value;
                    OnPropertyChanged(nameof(imgsDir));
                }
            }
        }

        public AppViewModel()
        {
            _yoloService = new YoloService();
            objects = Enumerable.Empty<ObjectVm>();
            selectImgsDir = new RelayCommand(OnSelectFolder, x => !_loading);
            processImgs = new AsyncRelayCommand(ProcessImagesAsync, x => imgsDir != string.Empty && !_loading);
            cancel = new RelayCommand(OnRequestCancellation, x => _loading);
        }

        public void OnPropertyChanged([CallerMemberName] string prop = "")
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(prop));
        }

        


        private void OnSelectFolder(object arg)
        {
            VistaFolderBrowserDialog dialog = new VistaFolderBrowserDialog();
            if (dialog.ShowDialog() == true && !string.IsNullOrEmpty(dialog.SelectedPath))
                imgsDir = dialog.SelectedPath;
        }

        public string[]? GetFiles()
        {
            try
            {
                if (!string.IsNullOrEmpty(imgsDir))
                    return Directory.GetFiles(imgsDir, "*.jpg", SearchOption.AllDirectories);
                else throw new Exception("Не выбрана папка с изображениями.");
            }
            catch (DirectoryNotFoundException)
            {
                MessageBox.Show("Папка с таким путем не найдена или не существует.");
                return null;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
                return null;
            }
        }

        public async Task ProcessImagesAsync(object arg) {
            objects = Enumerable.Empty<ObjectVm>();
            OnPropertyChanged(nameof(objects));
            try
            {
                _loading = true;
                _cts = new CancellationTokenSource();
                var imageProcessTasks = GetFiles().Select(a => Task.Run(async () => {
                    var task = await _yoloService.ProcessImageAsync(a, _cts.Token, false);
                    foreach (var obj in task.objectsProps)
                    {
                        var img = task.img.Clone();
                        double xMax, yMax, xMin, yMin;
                        Image croped;

                        xMax = (obj.xMax < img.Width) ? obj.xMax : img.Width;
                        yMax = (obj.yMax < img.Height) ? obj.yMax : img.Height;
                        xMin = (obj.xMin > 0) ? obj.xMin : 0;
                        yMin = (obj.yMin > 0) ? obj.yMin : 0;
                            
                        croped = img.Clone(i => i.Crop(new SixLabors.ImageSharp.Rectangle((int)xMin, (int)yMin, (int)(xMax - xMin), (int)(yMax - yMin))));
                       
                        _sessionLock.Wait();
                        _yoloService.Annotate(img, new ObjectBox[] { new ObjectBox(xMin, yMin, xMax, yMax, 0,0)}, false);
                        objects = objects.Append(new ObjectVm(img, croped, obj.Class, obj.confidence));
                        _sessionLock.Release();

                    }
                    OnPropertyChanged(nameof(objects));

                }));
                await Task.WhenAll(imageProcessTasks);
            }
            catch(Exception ex) {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                _loading = false;
            }
        }

        public void OnRequestCancellation(object arg)
        {
            _cts.Cancel();
        }
    }
}
