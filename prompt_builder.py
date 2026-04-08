def build_prompt(query: str, chunks: list[dict]) -> dict:
    """
    Assemble a prompt payload ready to send to your HuggingFace LLM.

    Returns a dict with:
      - system_prompt: context-injected instruction for the LLM
      - user_message:  the original user question
      - sources:       list of source references for the frontend to display
    """
    if not chunks:
        context = "No relevant context found in the uploaded documents."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source_label = f"[{i}] {chunk['filename']} (page {chunk['page']})"
            context_parts.append(f"{source_label}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the context provided below. If the answer is not in the context, say "
        "you don't know — do not make up information.\n\n"
        f"Context:\n{context}"
    )

    sources = [
        {"filename": c["filename"], "page": c["page"], "score": c["score"]}
        for c in chunks
    ]

    return {
        "system_prompt": system_prompt,
        "user_message": query,
        "sources": sources,
    }