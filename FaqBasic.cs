public record FaqBasic(string Question, string Response)
{
    public FaqBasic(FaqModel model) : this(model.Question, model.Response) { }
}