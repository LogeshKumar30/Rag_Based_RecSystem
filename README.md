# RAG based RecSys using LLMs
[![Static Badge](https://img.shields.io/badge/Demo-blue)](https://huggingface.co/spaces/Logesh30/Rag_based_Recsystem)

This work presents a real-time product recommendation system based on **Retrieval-Augmented Generation (RAG)**, designed to engage users with insightful recommendations. The system leverages the **ChromaDB** vector database, supported with the **BAAI/bge-large-en-v1.5** model, and employs a RAG-integrated **LangChain** pipeline for advanced semantic search. Powered by the open-source **LLama-3.1-70b-instruct model**, this solution is deployed as a **Streamlit API**, enabling efficient and interactive querying.


### Design overview
#### Vector-database
- A vector database in RAG stores document embeddings and enables semantic similarity search to retrieve relevant context for LLM response generation.
- We utilise the Chromadb package to generate a localised version of vector-database. To generate the database, we create the embbeddings of the product database using an open-source **BAAI/bge-large-en-v1.5** embedding generator with embedding dimension of 1024. The vector database are further stored in chunks of 32 for faster retreival.
- Based on the user-query, a total of n recommendations can be extracted.
**  need of vector db**
  1.Retrieving relevant data from a vector DB
  2.Augmenting the prompt with that data
  3.Generating a grounded, accurate answer

#### Langchain
- A customised prompt that analyses chat history and current user query which further powers the LLM to provide valuable suggestions to the user.
- The RecSys utilizes the **RetrievalQAWithSourcesChain** module alongside **ConversationSummaryMemory** to preserve chat history, integrating it with retrieved product documents from ChromaDB database to effectively address follow-up queries.
- We introduced a feature that allows seamless **context switching** within the same window through an **auxiliary agent**, which detects when the user seeks recommendations in a different category. This functionality leverages a secondary prompt to analyze chat history and the current query, determining contextual similarity. If a shift is detected, the system resets memory and begins anew, ensuring smooth transitions without disrupting the LLM's behavior.
  
**Qdrant** 
 It is an open-source vector database used to store and search embedding vectors efficiently for AI applications.

**Need:**
Qdrant is needed to quickly retrieve the most relevant data based on semantic similarity, enabling accurate search, recommendation systems, and Retrieval-Augmented Generation (RAG) with large language models.

#### Streamlit Frontend
- A web-interface is designed to abstract the underlying framework and provides the user the with a conversational recommendation system.
- The search parameter K can be adjusted in the web-interface, to limit the maximum number of products retrieved by the chain.
- The streamlit app can be run locally using the following command:
      ```
        streamlit run app.py
      ```

### Setup
1. Install the required packages:\
      `pip install -r requirements.txt`
2. Create your account in `DeepInfra` and generate your API key for utilising the embedding and LLM models. Update required feilds in `config.yaml`
3. Create your own local vector-database in chromaDB by running the following:\
      `python create_vectorStore.py`
4. Run your streamlit application
   `streamlit run app.py`

### Sample conversation
```
[USER] : Suggest some good cookies
[AGENT]: Here are some good cookie options you might enjoy:

      1. **Good Day Butter Cookies** by Britannia: Classic buttery cookies with a rich taste, perfect for tea-time snacks.
      2. **Hide & Seek Fab Choco Fills** by Parle: Crunchy cookies filled with smooth chocolate cream, loved by both kids and adults.
      3. **Oreo Original Cream Biscuits** by Cadbury: Chocolate-flavoured biscuits with a sweet vanilla cream filling, great for quick munching.
      
      Note that **Marie Gold Biscuits** by Britannia are a lighter and healthier option compared to cream-based cookies, making them suitable for daily consumption.
      
      Also, **Milk Bikis** by Britannia pairs well with milk and is a popular choice for children.

[USER] : "Okay, give me some good recommendations for Baby Care products"
[AGENT]: Here are some baby care products that might interest you:

      1. **Baby Gentle Shampoo** by Johnson’s: Mild and tear-free formula that cleans baby’s hair gently without irritating the eyes.
      2. **Baby Lotion** by Himalaya: Enriched with natural ingredients to keep baby’s skin soft, moisturized, and healthy.
      3. **Baby Massage Oil** by Dabur Lal Tail: Made with Ayurvedic ingredients to support baby’s bone and muscle development.
      4. **Baby Soap – Mild & Gentle** by Sebamed: Maintains skin’s natural pH and is suitable for babies with sensitive skin.
      
      These products are commonly used for maintaining baby hygiene, skin care, and overall comfort.

[USER] : "Tell me more about the baby lotion"
[AGENT]: Sure! Here are some additional details about the **Baby Lotion** by Himalaya:

      1. The lotion is enriched with herbal ingredients that help nourish and protect baby’s skin.
      2. It provides long-lasting moisturization and prevents dryness.
      3. The formula is mild, non-greasy, and easily absorbed.
      4. It is suitable for daily use and safe for delicate baby skin.
      
      Please note that the information shared is strictly based on the available product description, without adding external assumptions.
      
      If you would like details about other baby care products, feel free to ask!

[USER] : "Compare all the baby care products and give me a good recommendation with reason"
[AGENT]: Based on the provided product descriptions, here’s a comparison and recommendation:

      1. **Baby Gentle Shampoo** by Johnson’s: Tear-free and mild, ideal for regular hair cleansing without causing irritation.
      
      2. **Baby Lotion** by Himalaya: Focuses on skin hydration and nourishment using natural ingredients, suitable for daily moisturizing.
      
      3. **Baby Massage Oil** by Dabur Lal Tail: Designed to support physical development with traditional Ayurvedic formulation.
      
      4. **Baby Soap – Mild & Gentle** by Sebamed: Helps maintain skin’s natural pH and is especially good for sensitive skin.
      
      Recommendation: I would recommend the **Baby Lotion** by Himalaya as the best overall choice. It offers daily skin protection, hydration, and nourishment, making it suitable for all babies, especially those with dry or sensitive skin.
      
      While the shampoo, soap, and oil serve specific purposes, the lotion provides continuous skin care throughout the day, which is essential for a baby’s comfort and skin health.

```
### Upcoming updates
- Reducing overall latency in the framework
  - Retaining vector-database results if the consequitive context are similar.
  - Replacing auxiliary intent matching agent with pretrained BERT for intent similarity identification (or) Langraph.
