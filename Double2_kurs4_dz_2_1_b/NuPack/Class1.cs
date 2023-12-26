
using BERTTokenizers;
using System.Net;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NuPack
{
    public class MyPackedNetwork
    {
        private static InferenceSession session;
        

        private static string modelPath = "bert-large-uncased-whole-word-masking-finetuned-squad.onnx";
        CancellationToken ct;
        public MyPackedNetwork(CancellationToken ct1)
        {
            ct = ct1;
        }
        public Task<string> AnsweringAsync(string text, string question, CancellationToken ct)
        {
            return Task.Factory.StartNew(() => {
                try
                {
                    ct.ThrowIfCancellationRequested();
                    var sentence = "{\"question\": \"" + question + ", \"context\": \"@CTX\"}".Replace("@CTX", text);
                    var tokenizer = new BertUncasedLargeTokenizer();
                    var tokens = tokenizer.Tokenize(sentence);
                    var encoded = tokenizer.Encode(tokens.Count(), sentence);
                    var bertInput = new BertInput()
                    {
                        InputIds = encoded.Select(t => t.InputIds).ToArray(),
                        AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                        TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                    };

                    var input_ids = ConvertToTensor(bertInput.InputIds, bertInput.InputIds.Length);
                    var attention_mask = ConvertToTensor(bertInput.AttentionMask, bertInput.InputIds.Length);
                    var token_type_ids = ConvertToTensor(bertInput.TypeIds, bertInput.InputIds.Length);

                    var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids),
                                                    NamedOnnxValue.CreateFromTensor("input_mask", attention_mask),
                                                    NamedOnnxValue.CreateFromTensor("segment_ids", token_type_ids) };

                    ct.ThrowIfCancellationRequested();
                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue>? output;
                    lock (session)
                    {
                        output = session.Run(input);
                    }
                    ct.ThrowIfCancellationRequested();


                    List<float> startLogits = (output.ToList().First().Value as IEnumerable<float>).ToList();
                    List<float> endLogits = (output.ToList().Last().Value as IEnumerable<float>).ToList();

                    var startIndex = startLogits.ToList().IndexOf(startLogits.Max());
                    var endIndex = endLogits.ToList().IndexOf(endLogits.Max());

                    var predictedTokens = tokens
                                .Skip(startIndex)
                                .Take(endIndex + 1 - startIndex)
                                .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                                .ToList();

                    var ans = String.Join(" ", predictedTokens);
                    ct.ThrowIfCancellationRequested();
                    return ans;
                }
                catch (OperationCanceledException)
                {
                    return ("Cancelled.");
                }
                catch (Exception e)
                {
                    return e.Message;
                }
            }, ct, TaskCreationOptions.LongRunning, TaskScheduler.Current);
        }
        public async Task MakeSession()
        {
            if (!File.Exists(modelPath))
            {
                await Download_Network();
            }
            session = new InferenceSession(modelPath);
        }
        public async Task Download_Network()
        {
            string url = "https://storage.yandexcloud.net/dotnet4/bert-large-uncased-whole-word-masking-finetuned-squad.onnx"; // Замените на свою ссылку

            //string fileName = "C:\\Users\\Admin\\source\\repos\\ConsoleApp_Lab_1\\ConsoleApp_Lab_1\\bert-large-uncased-whole-word-masking-finetuned-squad.onnx"; // Замените на путь, куда сохранить файл

            string fileName = "bert-large-uncased-whole-word-masking-finetuned-squad.onnx"; // Замените на путь, куда сохранить файл

            if (File.Exists(fileName))
            {
                //Console.WriteLine($"Файл {fileName} уже существует.");
                return;
            }

            using (HttpClient client = new HttpClient())
            using (HttpResponseMessage response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead))
            using (Stream streamToReadFrom = await response.Content.ReadAsStreamAsync())
            using (Stream streamToWriteTo = File.Open(fileName, FileMode.Create))
            {
                byte[] buffer = new byte[8192];
                long totalBytesRead = 0;
                long totalBytes = response.Content.Headers.ContentLength.GetValueOrDefault();

                while (true)
                {
                    int bytesRead = await streamToReadFrom.ReadAsync(buffer, 0, buffer.Length);

                    if (bytesRead == 0)
                    {
                        break;
                    }

                    await streamToWriteTo.WriteAsync(buffer, 0, bytesRead);

                    totalBytesRead += bytesRead;

                    if (totalBytes > 0)
                    {
                        double percentage = (double)totalBytesRead / totalBytes * 100;
                        Console.WriteLine($"Загружено: {percentage:F2}%");
                    }
                }
            }

            //Console.WriteLine($"Файл {fileName} успешно скачан.");
        }

        public static Tensor<long> ConvertToTensor(long[] inputArray, int inputDimension)
        {
            // Create a tensor with the shape the model is expecting. Here we are sending in 1 batch with the inputDimension as the amount of tokens.
            Tensor<long> input = new DenseTensor<long>(new[] { 1, inputDimension });

            // Loop through the inputArray (InputIds, AttentionMask and TypeIds)
            for (var i = 0; i < inputArray.Length; i++)
            {
                // Add each to the input Tenor result.
                // Set index and array value of each input Tensor.
                input[0, i] = inputArray[i];
            }
            return input;
        }
        public class BertInput
        {
            public long[] InputIds { get; set; }
            public long[] AttentionMask { get; set; }
            public long[] TypeIds { get; set; }
        }
    }
}

