using BERTTokenizers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using System.IO;
using System;
using static System.Net.Mime.MediaTypeNames;
namespace NugPack
{
    public class BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public long[] TypeIds { get; set; }
    }
    public static class Class1
    {
        public static IEnumerable<string> Answer_Generate(string inputText, params string[] inputQuestions)
        {     

            
            foreach (var question in inputQuestions)
            {

                if (string.IsNullOrWhiteSpace(question))
                {
                    Console.WriteLine("Вы ввели пустую строку. На этом всё.");
                    break;
                }

                var sentence = "{\"question\": \"@CGI\", \"context\": \"@CTX\"}".Replace("@CTX", inputText).Replace("@CGI", question);

                
                Console.WriteLine(sentence);

                // Create Tokenizer and tokenize the sentence.
                var tokenizer = new BertUncasedLargeTokenizer();

                // Get the sentence tokens.
                var tokens = tokenizer.Tokenize(sentence);
                

                // Encode the sentence and pass in the count of the tokens in the sentence.
                var encoded = tokenizer.Encode(tokens.Count(), sentence);

                // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
                var bertInput = new BertInput()
                {
                    InputIds = encoded.Select(t => t.InputIds).ToArray(),
                    AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                    TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                };

                // Get path to model to create inference session.
                

                var modelPath = "C:\\Users\\Admin\\source\\repos\\ConsoleApp_Lab_1\\ConsoleApp_Lab_1\\bert-large-uncased-whole-word-masking-finetuned-squad.onnx";

                
                // Create input tensor.

                var input_ids = ConvertToTensor(bertInput.InputIds, bertInput.InputIds.Length);
                var attention_mask = ConvertToTensor(bertInput.AttentionMask, bertInput.InputIds.Length);
                var token_type_ids = ConvertToTensor(bertInput.TypeIds, bertInput.InputIds.Length);


                // Create input data for session.
                var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids),
                                                    NamedOnnxValue.CreateFromTensor("input_mask", attention_mask),
                                                    NamedOnnxValue.CreateFromTensor("segment_ids", token_type_ids) };

                // Create an InferenceSession from the Model Path.
                var session = new InferenceSession(modelPath);

                // Run session and send the input data in to get inference output. 
                var output = session.Run(input);

                // Call ToList on the output.
                // Get the First and Last item in the list.
                // Get the Value of the item and cast as IEnumerable<float> to get a list result.
                List<float> startLogits = (output.ToList().First().Value as IEnumerable<float>).ToList();
                List<float> endLogits = (output.ToList().Last().Value as IEnumerable<float>).ToList();

                // Get the Index of the Max value from the output lists.
                var startIndex = startLogits.ToList().IndexOf(startLogits.Max());
                var endIndex = endLogits.ToList().IndexOf(endLogits.Max());

                // From the list of the original tokens in the sentence
                // Get the tokens between the startIndex and endIndex and convert to the vocabulary from the ID of the token.
                var predictedTokens = tokens
                            .Skip(startIndex)
                            .Take(endIndex + 1 - startIndex)
                            .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                            .ToList();        
                yield return String.Join(" ", predictedTokens);
            }
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

       
        public static async void Download_Network()
        {
            string fileUrl = "https://storage.yandexcloud.net/dotnet4/bert-large-uncased-whole-word-masking-finetuned-squad.onnx"; // Замените на свою ссылку

            string destinationPath = "C:\\Users\\Admin\\source\\repos\\ConsoleApp_Lab_1\\ConsoleApp_Lab_1\\bert-large-uncased-whole-word-masking-finetuned-squad.onnx"; // Замените на путь, куда сохранить файл

            if (File.Exists(destinationPath))
            {
            }
            else
            {

                using (var httpClient = new HttpClient())
                {
                    using (var response = await httpClient.GetAsync(fileUrl, HttpCompletionOption.ResponseHeadersRead))
                    {
                        response.EnsureSuccessStatusCode();

                        long? contentLength = response.Content.Headers.ContentLength;
                        if (!contentLength.HasValue)
                        {
                            Console.WriteLine("Невозможно получить размер файла.");
                            return;
                        }

                        using (var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write))
                        {
                            using (var contentStream = await response.Content.ReadAsStreamAsync())
                            {
                                byte[] buffer = new byte[8192];
                                long downloadedBytes = 0;
                                int bytesRead;

                                while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                                {
                                    await fileStream.WriteAsync(buffer, 0, bytesRead);
                                    downloadedBytes += bytesRead;

                                    double percentage = (double)downloadedBytes / contentLength.Value * 100;
                                    Console.WriteLine($"Загружено: {percentage:F2}%");
                                }
                            }
                        }
                    }
                }
                Console.WriteLine("Скачивание завершено.");
            }

        }
    }



    }