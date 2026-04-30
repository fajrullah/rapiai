from app.services.generation import generate_answer


def build_prompt(query: str, chunks: list[dict]) -> dict:
    """Assemble a prompt from retrieved chunks and synthesize an answer via LLM."""
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

    answer = generate_answer(
        system_prompt=system_prompt,
        user_message=query,
    )

    return {
        "answer": answer,
        "user_message": query,
        "sources": sources,
    }

