using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;


namespace MyWpfApp
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        CancellationTokenSource myCts;
        string text;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void loadButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*\";";

            if (openFileDialog.ShowDialog() == true)
            {
                string fileName = openFileDialog.FileName;
                text = File.ReadAllText(fileName);

                chatTextBox.Text += "Text is loaded.\n";
                chatTextBox.Text += "--------------------------------------------------------------------------.\n";
                chatTextBox.Text += text + "\n";
                chatTextBox.Text += "--------------------------------------------------------------------------.\n";

                //sendButton.IsEnabled = true;
            }
        }

        private void sendButton_Click(object sender, RoutedEventArgs e)
        {
            string input_qшestion = 'What text is about?';
            NuPack.MyPackedNetwork.AnsweringAsync(text, input_qшestion, CancellationTokenSource myCts;);

        }

        private void cancelButton_Click(object sender, RoutedEventArgs e)
        {

        }

        private void chatTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void questionTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {

        }
    }
}
