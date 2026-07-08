import gradio as gr
import torch
import torch.nn as nn

def transformer_demo(text):

    text = text.lower()

    # STEP 1
    output = f"STEP 1: INPUT\n{text}\n\n"

    # STEP 2
    words = text.split()

    if len(words) < 2:
        return "Please enter at least 2 words."

    vocab = sorted(list(set(words)))

    word_to_idx = {
        word: i
        for i, word in enumerate(vocab)
    }

    idx_to_word = {
        i: word
        for word, i in word_to_idx.items()
    }

    tokens = [
        word_to_idx[word]
        for word in words
    ]

    output += f"STEP 2: TOKENS\n{tokens}\n\n"

    # STEP 3
    embedding_dim = 8

    embedding = nn.Embedding(
        len(vocab),
        embedding_dim
    )

    token_embeddings = embedding(
        torch.tensor(tokens)
    )

    output += f"STEP 3: TOKEN EMBEDDINGS\n{token_embeddings}\n\n"

    # STEP 4
    position_embedding = nn.Embedding(
        20,
        embedding_dim
    )

    positions = torch.arange(
        len(tokens)
    )

    position_vectors = position_embedding(
        positions
    )

    output += f"STEP 4: POSITION EMBEDDINGS\n{position_vectors}\n\n"

    # STEP 5
    x = token_embeddings + position_vectors

    output += f"STEP 5: TOKEN + POSITION\n{x}\n\n"

    # STEP 6
    query_layer = nn.Linear(
        embedding_dim,
        embedding_dim
    )

    key_layer = nn.Linear(
        embedding_dim,
        embedding_dim
    )

    value_layer = nn.Linear(
        embedding_dim,
        embedding_dim
    )

    Q = query_layer(x)
    K = key_layer(x)
    V = value_layer(x)

    output += f"STEP 6: QUERY\n{Q}\n\n"
    output += f"KEY\n{K}\n\n"
    output += f"VALUE\n{V}\n\n"

    # STEP 7
    scores = torch.matmul(
        Q,
        K.T
    )

    output += f"STEP 7: ATTENTION SCORES\n{scores}\n\n"

    # STEP 8
    attention_weights = torch.softmax(
        scores,
        dim=-1
    )

    output += f"STEP 8: ATTENTION WEIGHTS\n{attention_weights}\n\n"

    # STEP 9
    attention_output = torch.matmul(
        attention_weights,
        V
    )

    output += f"STEP 9: ATTENTION OUTPUT\n{attention_output}\n\n"

    # STEP 10
    ffn = nn.Sequential(
        nn.Linear(
            embedding_dim,
            32
        ),
        nn.ReLU(),
        nn.Linear(
            32,
            embedding_dim
        )
    )

    ffn_output = ffn(
        attention_output
    )

    output += f"STEP 10: FFN OUTPUT\n{ffn_output}\n\n"

    # STEP 11
    transformer_output = (
        attention_output
        +
        ffn_output
    )

    output += f"STEP 11: TRANSFORMER OUTPUT\n{transformer_output}\n\n"

    # STEP 12
    output_layer = nn.Linear(
        embedding_dim,
        len(vocab)
    )

    logits = output_layer(
        transformer_output[-1]
    )

    output += f"STEP 12: LOGITS\n{logits}\n\n"

    # STEP 13
    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    output += f"STEP 13: PROBABILITIES\n{probabilities}\n\n"

    # STEP 14
    top_k = min(3, len(vocab))

    top_probs, top_indices = torch.topk(
        probabilities,
        top_k
    )

    output += "TOP 3 PREDICTIONS\n\n"

    for i in range(top_k):

        word = idx_to_word[
            top_indices[i].item()
        ]

        prob = (
            top_probs[i].item()
            * 100
        )

        output += (
            f"{i+1}. {word} : "
            f"{prob:.2f}%\n"
        )

    return output


demo = gr.Interface(
    fn=transformer_demo,
    inputs=gr.Textbox(
        label="Enter Sentence",
        placeholder="Example: I love machine learning"
    ),
    outputs=gr.Textbox(
        label="Transformer Output",
        lines=30
    ),
    title="Mini Transformer Explorer",
    description="Visualize Embeddings, Attention, Softmax and Next Word Prediction"
)

demo.launch()