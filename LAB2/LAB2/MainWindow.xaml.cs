using LAB2.Model;
using System.Windows;


namespace LAB2
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        public MainWindow()
        {
            InitializeComponent();
            DataContext = new AppViewModel();
        }


        
    }
}