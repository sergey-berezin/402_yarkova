using Microsoft.Win32;
using System.IO;
using System;
using System.Collections.Generic;
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
using static System.Net.Mime.MediaTypeNames;
using Microsoft.ML.OnnxRuntime;
using static System.Collections.Specialized.BitVector32;
using System.Xml;

namespace MyNuPackApp
{
    public class DialogEntry
    {
        public string Question { get; set; }
        public string Answer { get; set; }
        public string FileText { get; set; }
    }
   
    public partial class MainWindow : Window
    {
        NuPack.MyPackedNetwork bertNetwork;
        CancellationTokenSource cts;
        string text;
        CancellationTokenSource bertCts;

        public MainWindow()
        {
            InitializeComponent();
            cts = new CancellationTokenSource();
            bertNetwork = new NuPack.MyPackedNetwork(cts.Token);
            ModelLoadAsync();
            text = null;
        }
        private async void ModelLoadAsync()
        {
            sendButton.IsEnabled = false;
            cancelButton.IsEnabled = false;
            try
            {
                await bertNetwork.MakeSession();
                chatTextBox.Text += "Берт готов к работе.\n";
                
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка: {ex.Message}");
            }
           
        }

        private async void sendButton_Click(object sender, RoutedEventArgs e)
        {
            cancelButton.IsEnabled = true;
            if (text == null)
            {
                chatTextBox.Text += $"Подгрузите, пожалуйста, текст\n";
                cancelButton.IsEnabled = false;
                sendButton.IsEnabled = true;
                return;
            }

            string quest = questionTextBox.Text;
            bertCts = new CancellationTokenSource();

            try
            {
                var ans = await NuPack.MyPackedNetwork.AnsweringAsync(text, quest, bertCts.Token);

                chatTextBox.Text += $"Ваш вопрос: {quest}\n";
                chatTextBox.Text += $"Ответ Берт: {ans}\n";
            }
            catch (Exception ex)
            {
                if (bertCts.Token.IsCancellationRequested)
                {
                    chatTextBox.Text += "Отмена. Сворачиваемся\n";
                    cancelButton.IsEnabled = false;
                    sendButton.IsEnabled = true;
                    return;
                }
                MessageBox.Show($"Ошибка: {ex.Message}");
            }
            cancelButton.IsEnabled = false;
        }
        
        private void loadButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*\";";

            if (openFileDialog.ShowDialog() == true)
            {
                string fileName = openFileDialog.FileName;
                text = File.ReadAllText(fileName);

                chatTextBox.Text += "Ваш текст:\n";
                chatTextBox.Text += text + "\n";
                sendButton.IsEnabled = true;
            }
        }

        private void cancelButton_Click(object sender, RoutedEventArgs e)
        {
            bertCts.Cancel();
            bertCts.Dispose();
        }

        private void questionTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void chatTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {

        }
    }
}
