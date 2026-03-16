import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def load_documents(directory_path="docs"):
    print("Loading documents from source docs folder")

    try:
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        documents = loader.load()

        print(f"Total documents loaded: {len(documents)}")

        if len(documents) == 0:
            print(f"No text documents present in the directory: {directory_path}")

        for i, doc in enumerate(documents):
            print(f"\nDocument details {i+1}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Content preview: {doc.page_content[:100]}")
            print(f"Metadata: {doc.metadata}")

        return documents

    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def chunk_documents(documents, chunk_size=100, chunk_overlap=10):
    print("Chunking documents into smaller pieces")

    try:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Total chunks created: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"\nChunk details {i+1}")
            print(f"Metadata: {chunk.metadata}")
            print(f"Chunk length: {len(chunk.page_content)}")
            print("Chunk content:")
            print(chunk.page_content)

        return chunks

    except Exception as e:
        print(f"Error chunking documents: {e}")
        return []


def embedding_vector_store(text_chunks, persist_directory="db/chroma"):
    print("Embedding and storing chunks in Chroma DB")

    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )

        print("Finished embedding and storing chunks in Chroma DB")

        return vector_store

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it in a .env file or environment variable."
        )

    # Load documents
    documents = load_documents("docs")

    # Chunk documents
    text_chunks = chunk_documents(documents)

    # Create embeddings and store
    vector_store = embedding_vector_store(text_chunks)


if __name__ == "__main__":
    main()