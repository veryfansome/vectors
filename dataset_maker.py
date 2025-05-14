from datasets import Dataset, load_dataset, load_from_disk
from openai import AsyncOpenAI
import asyncio
import faiss
import logging
import numpy as np
import os

import labels

EMBEDDING_DIMS = 3072  # Can be reduced down to 256
EMBEDDING_MODEL = "text-embedding-3-large"

async_openai_client = AsyncOpenAI()
logger = logging.getLogger(__name__)


async def batch_query_faiss_index(faiss_index, id_to_labels: dict[int, str], qry_texts: list[str], min_sim_score: float):
    results = await asyncio.gather(*[query_faiss_index(faiss_index, id_to_labels, t, min_sim_score) for t in qry_texts])
    return [(t, results[i]) for i, t in enumerate(qry_texts)]


async def build_ds_dict(words: list[str]):
    labels = []
    embeddings = []
    for word in words:
        logger.info(f"fetching vector for {word}")
        word_embedding = await get_text_embedding(word)
        labels.append(word)
        embeddings.append(word_embedding)
    return {"word": words, "embedding": embeddings}


async def faiss_vector_search(faiss_idx, query_vector, id_to_labels: dict[int, str], num_results: int, min_sim_score: float):
    results = []
    sim_scores, neighbors = await asyncio.to_thread(faiss_idx.search, query_vector, num_results)
    for rank, (vector_id, sim_score) in enumerate(zip(neighbors[0], sim_scores[0]), 1):
        if vector_id != -1 and sim_score > min_sim_score:  # -1 means no match
            results.append((rank, id_to_labels[vector_id], sim_score))
    return results


async def get_text_embedding(text: str):
    response = await async_openai_client.embeddings.create(
        dimensions=EMBEDDING_DIMS,
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding


async def query_faiss_index(faiss_index, id_to_labels: dict[int, str], qry_text: str, min_sim_score: float):
    qry_embedding = await get_text_embedding(qry_text)
    qry_vector = np.array([qry_embedding], dtype=np.float32)
    await asyncio.to_thread(faiss.normalize_L2, qry_vector)
    return await faiss_vector_search(faiss_index, qry_vector, id_to_labels, num_results=3, min_sim_score=min_sim_score)


async def test_classification(test_ds: Dataset, cases: list[str], min_sim_score: float):
    logger.info(test_ds)
    test_faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIMS))
    word_id_to_word: dict[int, str] = {}
    for word_id in range(len(test_ds)):
        word = test_ds[word_id]['word']
        word_embedding = test_ds[word_id]['embedding']
        word_vector = np.array([word_embedding], dtype=np.float32)
        faiss.normalize_L2(word_vector)  # Important for cosine search
        logging.info(f"indexing {word}: {word_vector}")
        test_faiss_index.add_with_ids(word_vector, np.array([word_id], dtype=np.int64))
        word_id_to_word[word_id] = word

    batch_results = await batch_query_faiss_index(test_faiss_index, word_id_to_word, cases, min_sim_score)
    for case, results in batch_results:
        logger.info(case)
        for result in results:
            rank, label, sim_score = result
            logger.info(f"{rank:>2} sim={sim_score:.4f} label={label}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_cases = [
        "Steph Curry's stamina is legendary and a sight to behold.",
        "Did you see how I dunked over that guy?",
        "I feel gross!",
        "I can't wait to try my new red sports car!",
        "When the hearst came for my mother, I struggled to watch her go.",
        "I think that would probably work.",
        "Hello, I need some help.",
        "Ok, that worked.",
        "I think you're a total waste of space.",
        "Tell me a joke",
        "I'm so bored! Tell me a joke",
        "There's nothing to do around here",
        "You're so boring!",
        "Doctor Green is very experienced and will take great care of you.",
        "Some mechanics will take advantage of you but Bill is reliable.",
        "I'm constantly worried that my son will injure himself the moment I look away.",
        "I think I hurt his feeling. I feel pretty bad about it.",
        "I broke his leg. I feel pretty bad about it.",
        "Please explain e = m x c^2",
        "What is 1024 times 1024?",
        "Write me a sonnet",
        "What are the symptoms of attention deficit disorder?",
        "I'm experiencing heart palpitations",
        "Have you been to Hawaii?",
    ]

    if not os.path.exists("topic_embeddings"):
        topic_dict_to_save = asyncio.run(build_ds_dict(labels.topics))
        topic_ds_to_save = Dataset.from_dict(topic_dict_to_save)
        topic_ds_to_save.save_to_disk("topic_embeddings")

    if not os.path.exists("emotion_embeddings"):
        emo_dict_to_save = asyncio.run(build_ds_dict(labels.emotions))
        emo_ds_to_save = Dataset.from_dict(emo_dict_to_save)
        emo_ds_to_save.save_to_disk("emotion_embeddings")

    asyncio.run(test_classification(load_from_disk("./topic_embeddings"),
                                    test_cases, min_sim_score=0.10))

    #asyncio.run(test_classification(load_from_disk("./emotion_embeddings"),
    #                                test_cases, min_sim_score=0.10))

