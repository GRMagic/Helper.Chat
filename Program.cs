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
    .AddSingleton<FaqPlugin>();                                                                                                 // Plugin de FAQ

// Cria um kernel com as dependencias configuradas

var kernelBuilder = serviceCollection.AddKernel();
var serviceProvider = serviceCollection.BuildServiceProvider();
var kernel = serviceProvider.GetRequiredService<Kernel>();

// Pede para baixa o modelo caso necessário

var ollamaClient = serviceProvider.GetRequiredService<OllamaApiClient>();
var pulling = ollamaClient.PullModelAsync(new OllamaSharp.Models.PullModelRequest() { Model = modelId });
Console.WriteLine($"Preparando modelo {modelId}");
var emptyLine = new string(' ', Console.BufferWidth) + '\r';
await foreach (var progress in pulling)
    if (progress != null)
        Console.Write($"{emptyLine}{progress.Status} ({progress.Total:n0} bytes) : {progress.Percent:n2}%\r");
Console.WriteLine();

// Adiciona um plugin de FAQ ao kernel

var faqPlugion = serviceProvider.GetRequiredService<FaqPlugin>();
await faqPlugion.InitializeAsync();
kernel.Plugins.AddFromObject(faqPlugion, "PerguntasFrequentes");

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