using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using OllamaSharp;
using System.ComponentModel;

public class ImagePlugin : IDisposable
{
    private const string _modelId = "llava";
    private readonly OllamaApiClient _ollamaApiClient;
    private readonly HttpClient _httpClient;
    private OllamaChatClient? _ollamaChatClient;

    public ImagePlugin(OllamaApiClient ollamaApiClient, HttpClient httpClient)
    {
        _ollamaApiClient = ollamaApiClient;
        _httpClient = httpClient;
    }

    public async Task InitializeAsync()
    {
        var pulling = _ollamaApiClient.PullModelAsync(new() { Model = _modelId });
        WriteLine($"Preparando modelo {_modelId}");
        await foreach (var progress in pulling)
            if (progress != null)
                WriteFromBegin($"{progress.Status} ({progress.Total:n0} bytes) : {progress.Percent:n2}%\r");
        Console.WriteLine();

        _ollamaChatClient = new OllamaChatClient(_ollamaApiClient.Uri, _modelId);
    }

    [KernelFunction("analizar_imagem")]
    [Description("Use essa função para descrever o conteúdo de imagens sempre que você tiver um link de uma imagem (as vezes o link pode estar na FAQ, outras vezes na pergunta do usuário). Em alguns casos resposta para uma dúvida do usuário pode estar sendo respondida através de uma image, nesses casos o retorno dessa função pode ser útil para montar a sua resposta.")]
    public async Task<string?> AnalyzeImage(Uri imagePath)
    {
        WriteLine($"Analizando imagem '{imagePath}'...");

        if(_ollamaChatClient == null) throw new InvalidOperationException($"O plugin '{nameof(ImagePlugin)}' não foi inicializado.");

        var messageBody = """
            Descreva detalhadamente a imagem.
            Caso ela seja a tela de um sistema cite todos os componentes com seus tipos de descrições.
            É importante que alguém que não está vendo a image consiga entender tudo o que está nela apenas usando sua descrição.
            Se a imagem é um gráfico, descreva o que ele representa e como ele está organizado.
            Se a imagem é um diagrama, descreva o que ele representa e como ele está organizado.
            Se a imagem é um texto, transcreva o texto.
            """;

        var message = new ChatMessage(ChatRole.User, messageBody);

        if (imagePath.IsFile)
        {
            var imageData = await File.ReadAllBytesAsync(imagePath.LocalPath);  // Isso pode ser uma falha de segurança, mas aqui está sendo usado em um ambiente controlado apenas para testes.
            message.Contents.Add(new DataContent(imageData, "image/*"));
        }
        else
        {
            var downloadResult = await _httpClient.GetAsync(imagePath); // Isso pode ser uma falha de segurança, mas aqui está sendo usado em um ambiente controlado apenas para testes.
            downloadResult.EnsureSuccessStatusCode();
            var imageData = await downloadResult.Content.ReadAsByteArrayAsync();
            message.Contents.Add(new DataContent(imageData!, downloadResult.Content.Headers.ContentType?.MediaType));
        }
        

        var response = await _ollamaChatClient!.GetResponseAsync(message);
        WriteLine(response.Text ?? string.Empty);
        return response.Text;
    }

    private static void WriteLine(string text)
    {
        var color = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"Plugin     : {text}");
        Console.ForegroundColor = color;
    }

    private static void WriteFromBegin(string text)
    {
        var emptyLine = new string(' ', Console.BufferWidth) + '\r';

        var color = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.Write($"{emptyLine}Plugin     : {text}");
        Console.ForegroundColor = color;
    }

    public void Dispose() => _ollamaChatClient?.Dispose();
}
