using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.Text.Json;

public class FaqPlugin
{
    private const string CollectionName = "FAQ";
    private const float QuestionThreshold = 0.3f;
    private const float ResponseThreshold = 0.5f;
    private readonly IVectorStore _vectorStore;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;

    public FaqPlugin(IVectorStore vectorStore, IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator)
    {
        _vectorStore = vectorStore;
        _embeddingGenerator = embeddingGenerator;
    }

    public async Task InitializeAsync()
    {
        await GetFaqCollection();
    }

    private async Task<IVectorStoreRecordCollection<ulong, FaqModel>> GetFaqCollection()
    {
        var recordCollection = _vectorStore.GetCollection<ulong, FaqModel>(CollectionName);
        if (!await recordCollection.CollectionExistsAsync())
        {
            WriteLine("Criando a coleção (database)...");
            await recordCollection.CreateCollectionAsync();

            WriteLine("Carregando arquivo json");
            var faqs = JsonSerializer.Deserialize<FaqModel[]>(File.ReadAllText("Faq.json"))!;
            WriteLine("Gerando Embedding e inserindo registros na coleção...");
            var embeddingTasks = faqs.Select(faq => Task.Run(async () =>
            {
                faq.QuestionEmbedding = await _embeddingGenerator.GenerateEmbeddingVectorAsync(faq.Question);
                faq.ResponseEmbedding = await _embeddingGenerator.GenerateEmbeddingVectorAsync(faq.Response);
                await recordCollection.UpsertAsync(faq);
            }));
            await Task.WhenAll(embeddingTasks);
            WriteLine("Coleção pronta!");
        }

        return recordCollection;
    }

    [KernelFunction("perguntas_frequentes")]
    [Description("Retorna uma lista as perguntas frequentes mais semelhates a questão e suas respectivas respostas. O parâmetro question é obrigatório, só use essa função quando tiver uma questão fornecida pelo usuário!")]
    public async Task<List<FaqBasic>> GetFaq(string question)
    {
        WriteLine($"Buscando perguntas frequentes semelhantes a \"{question}\"...");

        var faqsResult = new List<FaqBasic>();
        if (string.IsNullOrWhiteSpace(question)) return faqsResult;

        var faqCollection = await GetFaqCollection();

        var questionEmbedding = await _embeddingGenerator.GenerateEmbeddingVectorAsync(question);

        var questionResults = await faqCollection.VectorizedSearchAsync(questionEmbedding, new VectorSearchOptions()
        {
            Top = 3,
            Skip = 0,
            VectorPropertyName = nameof(FaqModel.QuestionEmbedding)
        });

        await foreach (var result in questionResults.Results)
        {
            if (result.Score > QuestionThreshold)
            {
                WriteLine($"Pergunta semelhante encontrada: {result.Record.Question}");
                faqsResult.Add(new(result.Record));
            }
        }

        var responseResults = await faqCollection.VectorizedSearchAsync(questionEmbedding, new VectorSearchOptions()
        {
            Top = 5,
            Skip = 0,
            VectorPropertyName = nameof(FaqModel.ResponseEmbedding)
        });

        await foreach (var result in responseResults.Results)
        {
            if (result.Score > ResponseThreshold)
            {
                if (!faqsResult.Any(f => f.Question == result.Record.Question))
                {
                    WriteLine($"Resposta semelhante a pergunta encontrada: {result.Record.Question}");
                    faqsResult.Add(new(result.Record));
                }
            }
        }
        WriteLine($"{faqsResult.Count} perguntas frequentes encontradas!");
        return faqsResult;
    }

    private static void WriteLine(string text)
    {
        var color = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"Plugin     : {text}");
        Console.ForegroundColor = color;
    }
}
