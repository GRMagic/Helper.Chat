using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using OllamaSharp;

var endpoint = "http://localhost:11434";
var modelId = "llama3.2";
var systemMessage =
    """
    Você é um assistente virtual especializado em responder perguntas frequentes com base em uma lista predefinida. Seu objetivo é fornecer respostas diretas e precisas apenas para as perguntas que estão na lista.
    Regras de comportamento:
    - Sempre fale em português do Brasil
    - Se o usuário te cumprimentar, sempre responda com um cumprimento apropriado.
    - Se a pergunta do usuário corresponder exatamente ou for semelhante a uma das perguntas da lista, forneça a resposta correspondente.
    - Se a pergunta não estiver na lista ou não for reconhecida, responda com: "Desculpe, não sei responder a essa pergunta."
    - Não tente inventar respostas ou extrapolar informações além do que está na lista.
    - Caso necessário, reformule a resposta para melhor compreensão, mas sem alterar seu significado.
    """;

// Cria uma coleção e configura as dependencias necessárias

var serviceCollection = new ServiceCollection()
    .AddSingleton(new OllamaApiClient(new Uri(endpoint), modelId))                                                              // Usado para conectado com o servidor Ollama
    .AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(new OllamaEmbeddingGenerator(new Uri(endpoint), modelId))      // Usado para gerar embeddings de texto
    .AddOllamaChatCompletion()                                                                                                  // Usado para responder chat
    .AddInMemoryVectorStore()                                                                                                   // Usado para armazenar embeddings
    .AddSingleton<FaqPlugin>()                                                                                                  // Plugin de FAQ
    .AddSingleton<ImagePlugin>()                                                                                                // Plugin de análise de imagens
    .AddHttpClient();                                                                                                           // Usado para fazer requisições HTTP

// Cria um kernel com as dependencias configuradas

var kernelBuilder = serviceCollection.AddKernel();
var serviceProvider = serviceCollection.BuildServiceProvider();
var kernel = serviceProvider.GetRequiredService<Kernel>();

// Solicita o download do modelo caso necessário

var ollamaClient = serviceProvider.GetRequiredService<OllamaApiClient>();
var pulling = ollamaClient.PullModelAsync(new () { Model = modelId });
Console.WriteLine($"Preparando modelo {modelId}");
var emptyLine = new string(' ', Console.BufferWidth) + '\r';
await foreach (var progress in pulling)
    if (progress != null)
        Console.Write($"{emptyLine}{progress.Status} ({progress.Total:n0} bytes) : {progress.Percent:n2}%\r");
Console.WriteLine();

// Exibe algumas informações sobre o modelo que está sendo usado

var modelInfo = await ollamaClient.ShowModelAsync(new() { Model = modelId });
Console.WriteLine("Informações do modelo:");
Console.WriteLine($"{"Arquitetura",-30}{modelInfo.Info.Architecture}");
Console.WriteLine($"{"Parâmetros",-30}{modelInfo.Info.ParameterCount:n0}");
Console.WriteLine($"{"Quantização",-30}{modelInfo.Details.QuantizationLevel}");
var extraInfo = new Dictionary<string, string>()
{
    { "general.languages", "Linguagens" },
    { "llama.context_length", "Tamanho do Contexto" },
    { "llama.embedding_length", "Tamanho do Embedding"}
};
foreach(var info in extraInfo)
    if (modelInfo.Info.ExtraInfo?.ContainsKey(info.Key) ?? false)
        Console.WriteLine($"{info.Value,-30}{modelInfo.Info.ExtraInfo[info.Key]}");

// Adiciona um plugin de FAQ ao kernel

var faqPlugion = serviceProvider.GetRequiredService<FaqPlugin>();
await faqPlugion.InitializeAsync();
kernel.Plugins.AddFromObject(faqPlugion, "PerguntasFrequentes");

// Adiciona um plugin de imagem ao kernel

var imagePlugin = serviceProvider.GetRequiredService<ImagePlugin>();
await imagePlugin.InitializeAsync();
kernel.Plugins.AddFromObject(imagePlugin, "AnaliseDeImagens");

// Cria um serviço de completude de chat

var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

// Cria um chat

var history = new ChatHistory();
history.AddSystemMessage(systemMessage);

// Loop do chat

Console.WriteLine("Chat pronto para uso!");
string? userInput = string.Empty;
while (userInput is not null)
{
    // Solicita uma entrada do usuário
    Console.Write("Usuário    : ");
    userInput = Console.ReadLine();

    // Adiciona a mensagem do usuário ao histórico do chat
    history.AddUserMessage(userInput);

    // Obtém a resposta do agente
    var result = await chatCompletionService.GetChatMessageContentAsync(
        history,
        executionSettings: new OllamaPromptExecutionSettings()
        {
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() // Deixa o agente escolher automaticamente a função a ser executada
        },
        kernel: kernel);

    Console.WriteLine("Assistente : " + result);

    // Adiciona a mensagem do agente ao histórico do chat
    history.AddMessage(result.Role, result.Content ?? string.Empty);

}