using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Data;

public class FaqModel
{
    [VectorStoreRecordKey]
    [TextSearchResultName]
    public ulong Id { get; init; }

    [VectorStoreRecordData]
    public required string Question { get; init; }

    [VectorStoreRecordData]
    public required string Response { get; init; }

    [VectorStoreRecordVector(3072)]
    public ReadOnlyMemory<float> QuestionEmbedding { get; set; }

    [VectorStoreRecordVector(3072)]
    public ReadOnlyMemory<float> ResponseEmbedding { get; set; }
}
