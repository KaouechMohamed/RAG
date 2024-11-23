import os
import pandas as pd
import chromadb
import streamlit as st
from langchain_groq import ChatGroq
from langchain_nomic.embeddings import NomicEmbeddings
from dotenv import load_dotenv



def prepare_data(path, embedding, collection):
    data = pd.read_csv(path)
    texts = data['Summary '].tolist()
    actions = data["Action"].tolist()
    expected_results = data["Expected Result"].tolist()
    metadata = [
        {"action": actions[i], "expected_result": expected_results[i]}
        for i in range(len(texts))
    ]
    embeddings = embedding.embed(texts, task_type="search_query")
    for i, text in enumerate(texts):
        collection.add(
            ids=[str(i)],
            embeddings=[embeddings[i]],
            documents=[text],  
            metadatas=[metadata[i]],
        )   
    return collection

def view_collection(collection):
    data = collection.get()
    print("Collection Contents:")
    for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
        print(f"Entry {i + 1}:")
        print("Document (Summary):", doc)
        print("Metadata:", meta)
        print("-" * 50)

    
def query_with_llm(collection, query, embedding, llm, n_results=3):
    query_embedding = embedding.embed([query], task_type="search_query")[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    query_results = []
    
    for i, result in enumerate(results["documents"]):
        document = result
        metadata = results["metadatas"][i] 
        if isinstance(metadata, list):
            metadata = metadata[0] 
        if isinstance(metadata, dict):
            prompt = f"""
            Query: {query}

            Document Summary: {document}

            Action to take: {metadata.get('action', 'No action found')}

            Expected Result: {metadata.get('expected_result', 'No expected result found')}

            Based on the above information, generate a concise and informative summary or suggestion.
            """
            response = llm.invoke(prompt)
            
            # Collect the data to return
            query_results.append({
                "document": document,
                "action": metadata.get('action', 'No action found'),
                "expected_result": metadata.get('expected_result', 'No expected result found'),
                "llm_response": response
            })
        else:
            print(f"Metadata for result {i + 1} is not in the correct format: {metadata}")
    
    return query_results

if __name__ == "__main__":
    client = chromadb.Client()   
    collection = client.create_collection(name="Test_Cases")
    embedding = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    data_path = "data/datacleaned.csv"
    collection = prepare_data(data_path, embedding, collection)    
    
    
    load_dotenv()
    api_key = os.getenv("api_key")
    llm = ChatGroq(
        temperature=0,
        groq_api_key=api_key,
        model_name="llama-3.1-70b-versatile"
    )
    
    st.title("RAG System Interface")
    user_query = st.text_input("Enter your query:")
    
    if st.button("Search"):
        if user_query:
            query_results = query_with_llm(collection, user_query, embedding, llm)
            st.subheader("Search Results")
            for i, result in enumerate(query_results):
                st.write(f"### Result {i + 1}")
                st.write(f"*Document Summary*: {result['document']}")
                st.write(f"*Action*: {result['action']}")
                st.write(f"*Expected Result*: {result['expected_result']}")
                st.write(f"*LLM Response*: {result['llm_response']}")
                st.markdown("---")
        else:
            st.warning("Please enter a query.")
