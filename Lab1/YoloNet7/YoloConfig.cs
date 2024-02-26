namespace YoloConstant
{
    public static class Constant
    {
        public static readonly int TARGET_SIZE = 416;
        public static readonly int CELL_COUNT = 13;
        public static readonly int CHANNEL_COUNT = 125;
        public static readonly int BOXES_PER_CELL = 5;
        public static readonly int BOX_INFO_FEATURE_COUNT = 5;
        public static readonly int CLASS_COUNT = 20;
        public static readonly float CELL_WIDTH = 32;
        public static readonly float CELL_HEIGHT = 32;
        public static readonly int CELL_SIZE = TARGET_SIZE / CELL_COUNT;
        public static string MODEL_URL = "https://storage.yandexcloud.net/dotnet4/tinyyolov2-8.onnx";
        public static string MODEL_NAME = "OnnxModel.onnx";
        public static readonly string[] LABELS = new string[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        public static readonly float[] ANCHORS = new float[]
        {
            1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
        };
    }
}