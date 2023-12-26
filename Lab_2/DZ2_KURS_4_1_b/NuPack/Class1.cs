
using BERTTokenizers;
using System.Net;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NuPack
{
    public class MyPackedNetwork
    {
        private static InferenceSession session;

        public static string modelUrl = "https://storage.yandexcloud.net/dotnet4/bert-large-uncased-whole-word-masking-finetuned-squad.onnx";
        public static string modelPath = "bert-large-uncased-whole-word-masking-finetuned-squad.onnx";
        CancellationToken ct;
        public MyPackedNetwork(CancellationToken ct1)
        {
            ct = ct1;
        }
        public static Task<string> AnsweringAsync(string text, string question, CancellationToken ct)
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
                    return ("Отмена");
                }
                catch (Exception e)
                {
                    return e.Message;
                }
            }, ct, TaskCreationOptions.LongRunning, TaskScheduler.Current);
        }
        public async Task MakeSession()
        {
            await DownloadModel();
            if (!File.Exists(modelPath))
            {
                await DownloadModel();
            }
            session = new InferenceSession(modelPath);
            
        }
        public async Task DownloadModel()
        {
            int maxim = 5;
            int i = 0;
            while (i < maxim)
            {
                try
                {
                    using (WebClient client = new WebClient())
                    {
                        client.DownloadFile(modelUrl, modelPath);
                    }
                    return;
                }
                catch (WebException)
                {
                    i++;
                }
                catch (Exception ex)
                {
                    throw new Exception($"Ошибка загрузки: {ex.Message}");
                }
            }
        }

        public static Tensor<long> ConvertToTensor(long[] inputArray, int inputDimension)
        {
            // Create a tensor with the shape the model is expecting. Here we are sending in 1 batch with the inputDimension as the amount of tokens.
            Tensor<long> input = new DenseTensor<long>(new[] { 1, inputDimension });

            // Loop through the inputArray (InputIds, AttentionMask and TypeIds)
            for (var i = 0; i < inputArray.Length; i++)
            {
                
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

